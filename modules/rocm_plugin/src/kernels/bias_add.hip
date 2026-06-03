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

} // namespace kernel
} // namespace rocm_gpu
} // namespace ov
