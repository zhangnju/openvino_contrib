// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
__global__ void transposeKernel(const T* __restrict__ src,
                                 T* __restrict__ dst,
                                 const int64_t* __restrict__ src_strides,
                                 const int64_t* __restrict__ dst_strides,
                                 const int32_t* __restrict__ perm,
                                 int64_t total_elements,
                                 int32_t ndim) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose flat dst index into nd coords
    int64_t remaining = idx;
    int64_t src_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        int64_t dim_coord = remaining / dst_strides[d];
        remaining -= dim_coord * dst_strides[d];
        src_offset += dim_coord * src_strides[perm[d]];
    }
    dst[idx] = src[src_offset];
}

void launchTranspose(const void* src,
                     void* dst,
                     const int64_t* src_shape,
                     const int32_t* perm,
                     int32_t ndim,
                     size_t element_size,
                     hipStream_t stream) {
    // Compute strides
    int64_t src_strides[8], dst_strides[8];
    int64_t dst_shape[8];

    // dst_shape[i] = src_shape[perm[i]]
    for (int i = 0; i < ndim; ++i) dst_shape[i] = src_shape[perm[i]];

    // row-major strides
    src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    dst_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];

    int64_t total = 1;
    for (int i = 0; i < ndim; ++i) total *= dst_shape[i];

    // Copy strides/perm to device
    int64_t *d_src_strides, *d_dst_strides;
    int32_t *d_perm;
    hipMalloc(&d_src_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_dst_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_perm, ndim * sizeof(int32_t));
    hipMemcpyAsync(d_src_strides, src_strides, ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_dst_strides, dst_strides, ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_perm, perm, ndim * sizeof(int32_t), hipMemcpyHostToDevice, stream);

    constexpr int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;

    if (element_size == 2) {
        hipLaunchKernelGGL(transposeKernel<__half>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const __half*)src, (__half*)dst, d_src_strides, d_dst_strides, d_perm, total, ndim);
    } else if (element_size == 4) {
        hipLaunchKernelGGL(transposeKernel<float>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const float*)src, (float*)dst, d_src_strides, d_dst_strides, d_perm, total, ndim);
    } else if (element_size == 8) {
        hipLaunchKernelGGL(transposeKernel<double>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const double*)src, (double*)dst, d_src_strides, d_dst_strides, d_perm, total, ndim);
    } else if (element_size == 1) {
        hipLaunchKernelGGL(transposeKernel<uint8_t>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const uint8_t*)src, (uint8_t*)dst, d_src_strides, d_dst_strides, d_perm, total, ndim);
    }

    // Free temporary buffers after kernel submission (stream-ordered)
    hipFreeAsync(d_src_strides, stream);
    hipFreeAsync(d_dst_strides, stream);
    hipFreeAsync(d_perm, stream);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
