// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Fused Reduce+Pointwise kernels for OV ROCm Plugin.
//
// Design inspired by MIGraphX's fuse_pointwise_reduce, re-implemented natively:
// - One workgroup (block) processes one "row" (outer dimensions)
// - Block-level parallel reduction using warp shuffles + shared memory
// - All intermediate values stay in registers/shared memory
// - No intermediate global memory writes between operations
//
// Supports:
//   LayerNorm: reduce_mean → sub → mul(sq) → reduce_mean → add(eps) → rsqrt → mul(gamma) → add(beta)
//   FC+GELU:   rocBLAS GEMM → bias_gelu_fused_kernel (bias add + GELU elementwise)
//   MaskedSoftmax: add(mask) → reduce_max → sub → exp → reduce_sum → div
//
// Reference: MIGraphX src/targets/gpu/jit/reduce.cpp + kernels/reduce.hpp

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// ── Warp-level reduction ──────────────────────────────────────────────────────
__device__ inline float warp_reduce_sum(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor(v, mask);
    return v;
}

__device__ inline float warp_reduce_max(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor(v, mask));
    return v;
}

// Block-level reduction using shared memory for cross-warp communication.
// smem must have at least (blockDim.x / 32) elements.
__device__ float block_reduce_sum(float v, volatile float* smem) {
    int warp  = threadIdx.x >> 5;
    int lane  = threadIdx.x & 31;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : 0.f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ float block_reduce_max(float v, volatile float* smem) {
    int warp  = threadIdx.x >> 5;
    int lane  = threadIdx.x & 31;
    v = warp_reduce_max(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : -1e30f;
    if (warp == 0) v = warp_reduce_max(v);
    return v;
}

// ── Fused LayerNorm kernel ────────────────────────────────────────────────────
// Computes: y = ((x + skip) - mean) / sqrt(var + eps) * gamma + beta
// skip may be nullptr; when non-null the residual is added before normalization.
// One block per row. Threads stride across the hidden dimension.
__global__ void __launch_bounds__(256)
layernorm_fused_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ skip,   // optional residual; may be nullptr
        const __half* __restrict__ gamma,
        const __half* __restrict__ beta,
        __half* __restrict__ y,
        int cols,
        float eps)
{
    extern __shared__ float smem[];

    const int row = blockIdx.x;
    const __half* row_in   = x    + (size_t)row * cols;
    const __half* row_skip = skip ? skip + (size_t)row * cols : nullptr;
    __half*       row_out  = y    + (size_t)row * cols;

    // ── Pass 1: Compute mean (of x + skip) ────────────────────────────────────
    float sum = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = __half2float(row_in[i]);
        if (row_skip) xi += __half2float(row_skip[i]);
        sum += xi;
    }
    sum = block_reduce_sum(sum, smem);
    __syncthreads();
    const float mean = sum / (float)cols;

    // ── Pass 2: Compute variance ──────────────────────────────────────────────
    float var_sum = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = __half2float(row_in[i]);
        if (row_skip) xi += __half2float(row_skip[i]);
        float d = xi - mean;
        var_sum += d * d;
    }
    var_sum = block_reduce_sum(var_sum, smem);
    __syncthreads();
    const float inv_std = rsqrtf(var_sum / (float)cols + eps);

    // ── Pass 3: Normalize + scale + shift ─────────────────────────────────────
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = __half2float(row_in[i]);
        if (row_skip) xi += __half2float(row_skip[i]);
        float gi   = __half2float(gamma[i]);
        float bi   = __half2float(beta[i]);
        float norm = (xi - mean) * inv_std * gi + bi;
        row_out[i] = __float2half(norm);
    }
}

// ── Fused Masked Softmax kernel ───────────────────────────────────────────────
// Computes: softmax(scores + mask) along the last dimension (sk)
// Numerically stable: subtract max before exp.
// One block per [batch, head, sq] row.
// Args:
//   scores[1, heads, sq, sk] f16
//   mask[sq, sk] f16 (2D, broadcast over batch and heads)
//   out[1, heads, sq, sk] f16
__global__ void __launch_bounds__(256)
masked_softmax_fused_kernel(
        const __half* __restrict__ scores,  // [heads, sq, sk] (batch=1 squeezed)
        const __half* __restrict__ mask,    // [sq, sk]
        __half* __restrict__ out,           // [heads, sq, sk]
        int heads, int sq, int sk)
{
    extern __shared__ float smem[];

    // One block per (head, sq) combination
    const int total_rows = heads * sq;
    if (blockIdx.x >= total_rows) return;

    const int head = blockIdx.x / sq;
    const int seq  = blockIdx.x % sq;

    const __half* row_scores = scores + ((size_t)head * sq + seq) * sk;
    const __half* row_mask   = mask   + (size_t)seq * sk;
    __half*       row_out    = out    + ((size_t)head * sq + seq) * sk;

    // ── Pass 1: Find max (for numerical stability) ────────────────────────────
    float row_max = -1e30f;
    for (int i = threadIdx.x; i < sk; i += blockDim.x) {
        float s = __half2float(row_scores[i]) + __half2float(row_mask[i]);
        row_max = fmaxf(row_max, s);
    }
    row_max = block_reduce_max(row_max, smem);
    __syncthreads();

    // ── Pass 2: exp(x - max) and sum ─────────────────────────────────────────
    float exp_sum = 0.f;
    for (int i = threadIdx.x; i < sk; i += blockDim.x) {
        float s = __half2float(row_scores[i]) + __half2float(row_mask[i]);
        exp_sum += expf(s - row_max);
    }
    exp_sum = block_reduce_sum(exp_sum, smem);
    __syncthreads();

    // ── Pass 3: Normalize ─────────────────────────────────────────────────────
    const float inv_sum = 1.f / exp_sum;
    for (int i = threadIdx.x; i < sk; i += blockDim.x) {
        float s = __half2float(row_scores[i]) + __half2float(row_mask[i]);
        row_out[i] = __float2half(expf(s - row_max) * inv_sum);
    }
}

// ── Fused FC+GELU kernel (GEMM part handled by rocBLAS, GELU part here) ───────
// Note: Full FC+GELU fusion (matmul inside kernel) is complex.
// This kernel handles the GELU+bias part after rocBLAS GEMM.
// GELU (erf form): y = x * 0.5 * (1 + erf(x / sqrt(2)))
__global__ void __launch_bounds__(256)
bias_gelu_fused_kernel(
        const __half* __restrict__ gemm_out,  // [seq, out_dim] from rocBLAS
        const __half* __restrict__ bias,      // [out_dim]
        __half* __restrict__ out,             // [seq, out_dim]
        int seq, int out_dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)seq * out_dim) return;

    const int col = idx % out_dim;
    float x = __half2float(gemm_out[idx]) + __half2float(bias[col]);
    // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    float gelu = x * 0.5f * (1.f + erff(x * 0.7071067811865476f));
    out[idx] = __float2half(gelu);
}

// ── Launch helpers ─────────────────────────────────────────────────────────────

void launch_layernorm_fused(
        hipStream_t stream,
        const void* x, const void* skip, const void* gamma, const void* beta, void* y,
        int rows, int cols, float eps) {
    int threads = 32;
    while (threads < cols && threads < 256) threads += 32;
    threads = (threads > 256) ? 256 : threads;
    int smem_floats = (threads + 31) / 32;

    hipLaunchKernelGGL(
        layernorm_fused_kernel,
        dim3(rows), dim3(threads), smem_floats * sizeof(float), stream,
        static_cast<const __half*>(x),
        static_cast<const __half*>(skip),
        static_cast<const __half*>(gamma),
        static_cast<const __half*>(beta),
        static_cast<__half*>(y),
        cols, eps);
}

void launch_masked_softmax_fused(
        hipStream_t stream,
        const void* scores, const void* mask, void* out,
        int heads, int sq, int sk) {
    int threads = 32;
    while (threads < sk && threads < 256) threads += 32;
    threads = (threads > 256) ? 256 : threads;
    int smem_floats = (threads + 31) / 32;
    int total_rows  = heads * sq;

    hipLaunchKernelGGL(
        masked_softmax_fused_kernel,
        dim3(total_rows), dim3(threads), smem_floats * sizeof(float), stream,
        static_cast<const __half*>(scores),
        static_cast<const __half*>(mask),
        static_cast<__half*>(out),
        heads, sq, sk);
}

void launch_bias_gelu_fused(
        hipStream_t stream,
        const void* gemm_out, const void* bias, void* out,
        int seq, int out_dim) {
    int total = seq * out_dim;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    hipLaunchKernelGGL(
        bias_gelu_fused_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        static_cast<const __half*>(gemm_out),
        static_cast<const __half*>(bias),
        static_cast<__half*>(out),
        seq, out_dim);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
