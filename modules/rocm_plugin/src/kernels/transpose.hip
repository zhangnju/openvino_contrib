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

// Layout of device_buf: [src_strides (ndim*8B) | dst_strides (ndim*8B) | perm (ndim*4B)]
// Allocated once at construction; no hipMalloc at Execute time → hipGraph-safe.
void* allocTransposeDeviceBuffers(const int64_t* src_shape, const int32_t* perm, int32_t ndim) {
    int64_t src_strides[8], dst_strides[8];
    int64_t dst_shape[8];
    for (int i = 0; i < ndim; ++i) dst_shape[i] = src_shape[perm[i]];
    src_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    dst_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];

    const size_t sz = 2 * ndim * sizeof(int64_t) + ndim * sizeof(int32_t);
    void* buf = nullptr;
    hipMalloc(&buf, sz);
    char* p = static_cast<char*>(buf);
    hipMemcpy(p,                           src_strides, ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(p + ndim * sizeof(int64_t),  dst_strides, ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(p + 2 * ndim * sizeof(int64_t), perm,    ndim * sizeof(int32_t), hipMemcpyHostToDevice);
    return buf;
}

void freeTransposeDeviceBuffers(void* device_buf) {
    if (device_buf) hipFree(device_buf);
}

void launchTranspose(const void* src,
                     void* dst,
                     void* device_buf,
                     int64_t total_elements,
                     int32_t ndim,
                     size_t element_size,
                     hipStream_t stream) {
    char* p = static_cast<char*>(device_buf);
    const int64_t* d_src_strides = reinterpret_cast<const int64_t*>(p);
    const int64_t* d_dst_strides = reinterpret_cast<const int64_t*>(p + ndim * sizeof(int64_t));
    const int32_t* d_perm        = reinterpret_cast<const int32_t*>(p + 2 * ndim * sizeof(int64_t));

    constexpr int BLOCK = 256;
    int grid = static_cast<int>((total_elements + BLOCK - 1) / BLOCK);

    if (element_size == 2) {
        hipLaunchKernelGGL(transposeKernel<__half>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const __half*)src, (__half*)dst, d_src_strides, d_dst_strides, d_perm, total_elements, ndim);
    } else if (element_size == 4) {
        hipLaunchKernelGGL(transposeKernel<float>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const float*)src, (float*)dst, d_src_strides, d_dst_strides, d_perm, total_elements, ndim);
    } else if (element_size == 8) {
        hipLaunchKernelGGL(transposeKernel<double>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const double*)src, (double*)dst, d_src_strides, d_dst_strides, d_perm, total_elements, ndim);
    } else if (element_size == 1) {
        hipLaunchKernelGGL(transposeKernel<uint8_t>, dim3(grid), dim3(BLOCK), 0, stream,
                           (const uint8_t*)src, (uint8_t*)dst, d_src_strides, d_dst_strides, d_perm, total_elements, ndim);
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
