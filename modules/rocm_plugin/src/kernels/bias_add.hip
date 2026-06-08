// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Broadcast bias add: output[n,k,h,w] += bias[k]
// Supports f16 and f32.

#include "bias_add.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
__global__ void bias_add_nchw(T* output, const T* bias,
                               int N, int K, int H, int W) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K * H * W;
    if (idx >= total) return;
    // Decompose idx → (n, k, h, w)
    const int HW = H * W;
    const int w  = idx % W;
    const int h  = (idx / W) % H;
    const int k  = (idx / HW) % K;
    output[idx] = output[idx] + bias[k];
}

void launch_bias_add(void* output, const void* bias,
                      int N, int K, int H, int W,
                      bool fp16, hipStream_t stream) {
    const int total = N * K * H * W;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;
    if (fp16) {
        hipLaunchKernelGGL(bias_add_nchw<__half>, dim3(grid), dim3(BLOCK), 0, stream,
                           (__half*)output, (const __half*)bias, N, K, H, W);
    } else {
        hipLaunchKernelGGL(bias_add_nchw<float>, dim3(grid), dim3(BLOCK), 0, stream,
                           (float*)output, (const float*)bias, N, K, H, W);
    }
}

// ── Vectorized pe 3-way add: attn_out += conv_out + bias ─────────────────────
// WHY NOT WMMA: WMMA computes C += A×B (matrix multiply). Element-wise addition
// (c[i] = a[i] + b[i]) has NO matrix multiply structure — it's purely memory-bound.
// The right optimization is __half2 vectorization using v_pk_add_f16 instruction,
// which processes 2 FP16 values per instruction = 2× memory throughput vs scalar.
//
// This replaces the linalg.generic in the pe MLIR kernel which:
//   1. Uses same grid as rock.conv → 172,032 threads for 153,600 elements (18,432 OOB!)
//   2. Generates scalar v_add_f16 (1 FP16/instruction) not v_pk_add_f16 (2 FP16)
//   3. rocMLIR linalg.generic lacks bounds check when grid is inherited from conv
//
// This kernel: proper bounds check + __half2 vectorization.
// Computes: attn[i] += conv[i] + bias[i/HW % K]  (in-place, 3-way add)
__global__ void pe_add_vectorized_h2(
        __half2* __restrict__ out_h2,      // attn_out in-place [(K*HW)/2 pairs]
        const __half* __restrict__ conv,   // conv(V) output, flat [K*HW]
        const __half* __restrict__ bias,   // bias [K]
        int K, int HW) {
    const int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_h2 = (K * HW) / 2;
    if (idx2 >= total_h2) return;  // bounds check prevents OOB writes

    const int i0 = idx2 * 2;
    const int i1 = i0 + 1;
    const int k0 = (i0 / HW) % K;
    const int k1 = (i1 / HW) % K;

    // Load 2 FP16 from attn_out at once (coalesced 32-bit load)
    __half2 attn = out_h2[idx2];

    // Load conv values and bias
    __half cb0 = __hadd(conv[i0], bias[k0]);  // conv + bias for elem 0
    __half cb1 = __hadd(conv[i1], bias[k1]);  // conv + bias for elem 1

    // Vectorized add: attn += (conv + bias) using v_pk_add_f16
    __half2 result = __hadd2(attn, __halves2half2(cb0, cb1));

    // Coalesced 32-bit store
    out_h2[idx2] = result;
}

void launch_pe_add_fp16(void* output, const void* conv_out, const void* bias_k,
                        int K, int HW, hipStream_t stream) {
    const int total_elems = K * HW;
    if (total_elems % 2 != 0 || total_elems == 0) return;
    const int total_h2 = total_elems / 2;
    constexpr int BLOCK = 256;
    const int grid = (total_h2 + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(pe_add_vectorized_h2, dim3(grid), dim3(BLOCK), 0, stream,
                       (__half2*)output, (const __half*)conv_out,
                       (const __half*)bias_k, K, HW);
}

// ── 3-way add: dst = src + conv + bias  (single-pass, no D2D copy) ───────────
// Replaces: hipMemcpyAsync(dst, src) + launch_pe_add_fp16(dst, conv, bias)
// Saves one memory traversal over the data.
__global__ void pe_add3_vectorized_h2(
        __half2* __restrict__ dst_h2,
        const __half2* __restrict__ src_h2,    // AV attn result [K*HW/2 pairs]
        const __half* __restrict__ conv,        // conv(V) output [K*HW]
        const __half* __restrict__ bias,        // bias [K]
        int K, int HW) {
    const int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_h2 = (K * HW) / 2;
    if (idx2 >= total_h2) return;

    const int i0 = idx2 * 2;
    const int k0 = (i0 / HW) % K;
    const int k1 = ((i0 + 1) / HW) % K;

    __half2 src = src_h2[idx2];  // load AV result pair
    __half cb0 = __hadd(conv[i0],   bias[k0]);
    __half cb1 = __hadd(conv[i0+1], bias[k1]);
    // dst = src + (conv + bias)
    dst_h2[idx2] = __hadd2(src, __halves2half2(cb0, cb1));
}

void launch_pe_add3_fp16(void* dst, const void* src, const void* conv_out,
                          const void* bias_k, int K, int HW, hipStream_t stream) {
    const int total_elems = K * HW;
    if (total_elems % 2 != 0 || total_elems == 0) return;
    const int total_h2 = total_elems / 2;
    constexpr int BLOCK = 256;
    const int grid = (total_h2 + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(pe_add3_vectorized_h2, dim3(grid), dim3(BLOCK), 0, stream,
                       (__half2*)dst, (const __half2*)src,
                       (const __half*)conv_out, (const __half*)bias_k, K, HW);
}

// ── Fused depthwise conv3×3 + bias + skip add ────────────────────────────────
// Single-pass kernel: reads V and AV_out once, writes fgc_out once.
// Replaces 2-kernel approach (pe_conv + pe_add3), saving 1 memory traversal.
__global__ void fused_pe_conv_add_kernel(
    __half* __restrict__ fgc_out,
    const __half* __restrict__ V,
    const __half* __restrict__ filter,
    const __half* __restrict__ AV_out,
    const __half* __restrict__ bias,
    int K, int H, int W)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = K * H * W;
    if (idx >= total) return;

    const int w = idx % W;
    const int h = (idx / W) % H;
    const int k = idx / (H * W);

    // Depthwise conv3×3 with padding=1
    float acc = 0.f;
    const int fbase = k * 9;
    for (int fy = 0; fy < 3; fy++) {
        const int ih = h + fy - 1;
        if (ih < 0 || ih >= H) continue;
        for (int fx = 0; fx < 3; fx++) {
            const int iw = w + fx - 1;
            if (iw < 0 || iw >= W) continue;
            acc += (float)V[k*H*W + ih*W + iw] * (float)filter[fbase + fy*3 + fx];
        }
    }
    acc += (float)bias[k];      // bias add
    acc += (float)AV_out[idx];  // attn skip add
    fgc_out[idx] = (__half)acc;
}

void launch_fused_pe_conv_add(void* fgc_out, const void* V, const void* filter,
                               const void* AV_out, const void* bias,
                               int K, int H, int W, hipStream_t stream) {
    const int total = K * H * W;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(fused_pe_conv_add_kernel, dim3(grid), dim3(BLOCK), 0, stream,
                       (__half*)fgc_out, (const __half*)V, (const __half*)filter,
                       (const __half*)AV_out, (const __half*)bias, K, H, W);
}

} // namespace kernel
} // namespace rocm_gpu
} // namespace ov
