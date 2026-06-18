// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Warp-level parallel sum reduction (all threads in warp participate).
__device__ inline float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

// Block-level sum using shared memory for cross-warp communication.
// Returns the SAME total to ALL threads (broadcast via smem[0]) — required because
// every thread uses the result (mean/var) to normalize its own elements.
__device__ float block_reduce_sum(float v, float* smem, int tid) {
    int warp_id = tid >> 5;         // tid / 32
    int lane_id = tid & 31;         // tid % 32
    int nwarps  = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum(v);
    if (lane_id == 0) smem[warp_id] = v;
    __syncthreads();
    // Warp 0 reduces the per-warp partial sums and writes the total to smem[0].
    if (warp_id == 0) {
        float t = (lane_id < nwarps) ? smem[lane_id] : 0.f;
        t = warp_reduce_sum(t);
        if (lane_id == 0) smem[0] = t;
    }
    __syncthreads();
    return smem[0];  // broadcast total to every thread
}

// One block handles one row of [outer, inner].
// blockDim.x = min(inner, 1024), rounded to nearest warp size.
// Each thread processes multiple elements in the inner dimension (strided).
template<typename T>
__global__ void layer_norm_kernel(
        const T* __restrict__ x,
        T*       __restrict__ y,
        const T* __restrict__ scale,    // [inner], nullable (gamma)
        const T* __restrict__ bias,     // [inner], nullable (beta)
        const T* __restrict__ add_bias, // [inner], nullable: pre-LN additive bias (broadcast over rows)
        const T* __restrict__ residual, // [outer,inner], nullable: pre-LN residual/skip
        int inner,
        float eps)
{
    extern __shared__ float smem[];  // blockDim.x/32 floats for cross-warp reduce

    int row = blockIdx.x;
    const T* row_in  = x + (size_t)row * inner;
    const T* row_res = residual ? residual + (size_t)row * inner : nullptr;
    T*       row_out = y + (size_t)row * inner;

    // Effective input = x (+ add_bias) (+ residual). Computed on the fly.
    auto load = [&](int i) -> float {
        float v = (float)row_in[i];
        if (add_bias) v += (float)add_bias[i];
        if (row_res)  v += (float)row_res[i];
        return v;
    };

    // Pass 1: mean
    float sum = 0.f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) sum += load(i);
    sum = block_reduce_sum(sum, smem, threadIdx.x);
    __syncthreads();
    float mean = sum / (float)inner;

    // Pass 2: variance
    float var_sum = 0.f;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float diff = load(i) - mean;
        var_sum += diff * diff;
    }
    var_sum = block_reduce_sum(var_sum, smem, threadIdx.x);
    __syncthreads();
    float inv_std = rsqrtf(var_sum / (float)inner + eps);

    // Pass 3: normalize (+ optional scale/bias)
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float norm = (load(i) - mean) * inv_std;
        if (scale) norm *= (float)scale[i];
        if (bias)  norm += (float)bias[i];
        row_out[i] = (T)norm;
    }
}

void launchLayerNorm(
    hipStream_t stream,
    const void* x,
    void*       y,
    const void* scale,
    const void* bias,
    const void* add_bias,
    const void* residual,
    size_t      outer,
    size_t      inner,
    float       epsilon,
    bool        is_fp16)
{
    // Choose threads per block: next multiple of 32 >= min(inner, 1024)
    int threads = 32;
    while (threads < (int)inner && threads < 1024) threads = (threads << 1);
    threads = (threads > 1024) ? 1024 : threads;

    // Shared memory: one float per warp
    int num_warps  = (threads + 31) / 32;
    size_t smem_sz = num_warps * sizeof(float);

    dim3 blocks(static_cast<unsigned>(outer));
    dim3 block_dim(static_cast<unsigned>(threads));

    if (is_fp16) {
        hipLaunchKernelGGL(
            layer_norm_kernel<__half>,
            blocks, block_dim, smem_sz, stream,
            static_cast<const __half*>(x),
            static_cast<__half*>(y),
            static_cast<const __half*>(scale),
            static_cast<const __half*>(bias),
            static_cast<const __half*>(add_bias),
            static_cast<const __half*>(residual),
            static_cast<int>(inner),
            epsilon);
    } else {
        hipLaunchKernelGGL(
            layer_norm_kernel<float>,
            blocks, block_dim, smem_sz, stream,
            static_cast<const float*>(x),
            static_cast<float*>(y),
            static_cast<const float*>(scale),
            static_cast<const float*>(bias),
            static_cast<const float*>(add_bias),
            static_cast<const float*>(residual),
            static_cast<int>(inner),
            epsilon);
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
