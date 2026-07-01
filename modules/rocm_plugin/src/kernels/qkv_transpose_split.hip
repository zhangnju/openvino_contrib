// QKV transpose+split kernel for Swin attention.
// Input:  x [nW, sq, 3*H*hd] — packed QKV from linear projection
// Output: Q, K, V each [nW, H, sq, hd]
//
// Replaces: Reshape [nW,sq,3,H,hd] → Transpose [3,nW,H,sq,hd] → 3×Gather
// The 3 Gather ops copy the entire tensor 3 times; this kernel reads once, writes 3 outputs.

#include "qkv_transpose_split.hpp"
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Each thread handles one element in ONE of the 3 outputs (Q/K/V).
// Total threads = nW * H * sq * hd * 3
// Thread maps to (qkv_idx, w, h, s, d) where qkv_idx ∈ {0,1,2}
static __global__ void qkv_transpose_split_fp16(
    const __half* __restrict__ x,  // [nW, sq, 3*H*hd]
    __half* __restrict__ q,         // [nW, H, sq, hd]
    __half* __restrict__ k,
    __half* __restrict__ v,
    int nW, int sq, int H, int hd)
{
    const int total_per_output = nW * H * sq * hd;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_per_output) return;

    // Decompose output index: [nW, H, sq, hd] layout
    const int d = idx % hd;
    const int s = (idx / hd) % sq;
    const int h = (idx / (hd * sq)) % H;
    const int w = idx / (hd * sq * H);

    // Input layout: x[w, s, 3*H*hd] where the last dim packs [qkv=3, H, hd]
    // x[w][s][qkv * H * hd + h * hd + d]
    const int stride_w = sq * 3 * H * hd;
    const int stride_s = 3 * H * hd;
    const int base = w * stride_w + s * stride_s;

    __half vq = x[base + 0 * H * hd + h * hd + d];
    __half vk = x[base + 1 * H * hd + h * hd + d];
    __half vv = x[base + 2 * H * hd + h * hd + d];

    q[idx] = vq;
    k[idx] = vk;
    v[idx] = vv;
}

// With optional Q/K L2 normalization (swin_v2)
static __global__ void qkv_transpose_split_norm_fp16(
    const __half* __restrict__ x,
    __half* __restrict__ q, __half* __restrict__ k, __half* __restrict__ v,
    int nW, int sq, int H, int hd, float eps)
{
    // Each warp handles one (w, h, s) row of hd elements
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int total_rows = nW * H * sq;
    if (warp_id >= total_rows) return;

    const int s = warp_id % sq;
    const int h = (warp_id / sq) % H;
    const int w = warp_id / (sq * H);

    const int stride_w = sq * 3 * H * hd;
    const int stride_s = 3 * H * hd;
    const int base = w * stride_w + s * stride_s;
    const int out_base = ((w * H + h) * sq + s) * hd;

    // Load and write V (no normalization needed)
    for (int d = lane; d < hd; d += 32) {
        v[out_base + d] = x[base + 2 * H * hd + h * hd + d];
    }

    // Load Q, compute L2 norm, normalize, write
    float sq_sum_q = 0.0f, sq_sum_k = 0.0f;
    for (int d = lane; d < hd; d += 32) {
        float fq = __half2float(x[base + 0 * H * hd + h * hd + d]);
        float fk = __half2float(x[base + 1 * H * hd + h * hd + d]);
        sq_sum_q += fq * fq;
        sq_sum_k += fk * fk;
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq_sum_q += __shfl_xor(sq_sum_q, offset);
        sq_sum_k += __shfl_xor(sq_sum_k, offset);
    }
    float inv_norm_q = rsqrtf(fmaxf(sq_sum_q, eps));
    float inv_norm_k = rsqrtf(fmaxf(sq_sum_k, eps));

    for (int d = lane; d < hd; d += 32) {
        float fq = __half2float(x[base + 0 * H * hd + h * hd + d]);
        float fk = __half2float(x[base + 1 * H * hd + h * hd + d]);
        q[out_base + d] = __float2half(fq * inv_norm_q);
        k[out_base + d] = __float2half(fk * inv_norm_k);
    }
}

void launchQKVTransposeSplit(hipStream_t stream,
                             const void* x, void* q, void* k, void* v,
                             int nW, int sq, int H, int head, bool is_fp16,
                             bool norm_q, bool norm_k, float eps) {
    if ((norm_q || norm_k) && head <= 64) {
        const long long rows = (long long)nW * H * sq;
        const int threads = 256;
        const int warps_per_block = threads / 32;
        const int grid = (int)((rows + warps_per_block - 1) / warps_per_block);
        hipLaunchKernelGGL(qkv_transpose_split_norm_fp16,
                           dim3(grid), dim3(threads), 0, stream,
                           (const __half*)x, (__half*)q, (__half*)k, (__half*)v,
                           nW, sq, H, head, eps);
    } else {
        const int total = nW * H * sq * head;
        const int threads = 256;
        const int grid = (total + threads - 1) / threads;
        hipLaunchKernelGGL(qkv_transpose_split_fp16,
                           dim3(grid), dim3(threads), 0, stream,
                           (const __half*)x, (__half*)q, (__half*)k, (__half*)v,
                           nW, sq, H, head);
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
