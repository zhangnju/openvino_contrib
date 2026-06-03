// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements.hpp"
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// GatherElements: out[i0,i1,...,axis=indices[i0,i1,...,in,...],in,...] = data[i0,i1,...,indices[i0,...],in,...]
__global__ void gatherElementsKernel(const char* __restrict__ data,
                                      const int64_t* __restrict__ indices,
                                      char* __restrict__ out,
                                      const int64_t* __restrict__ data_strides,
                                      const int64_t* __restrict__ out_strides,
                                      int32_t axis,
                                      int32_t ndim,
                                      int64_t total_out,
                                      size_t element_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    // decompose out_idx into coords
    int64_t remaining = out_idx;
    int64_t data_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        int64_t coord = remaining / out_strides[d];
        remaining -= coord * out_strides[d];
        if (d == axis) {
            // use indices value instead of coord
            coord = indices[out_idx];
        }
        data_offset += coord * data_strides[d];
    }

    const char* src = data + data_offset * element_size;
    char* dst = out + out_idx * element_size;
    for (size_t b = 0; b < element_size; ++b) dst[b] = src[b];
}

void launchGatherElements(const void* data,
                          const void* indices,
                          void* out,
                          const int64_t* data_strides,
                          const int64_t* out_strides,
                          int32_t axis,
                          int32_t ndim,
                          int64_t total_out,
                          size_t element_size,
                          hipStream_t stream) {
    int64_t *d_data_strides, *d_out_strides;
    hipMalloc(&d_data_strides, ndim * sizeof(int64_t));
    hipMalloc(&d_out_strides, ndim * sizeof(int64_t));
    hipMemcpyAsync(d_data_strides, data_strides, ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_out_strides, out_strides, ndim * sizeof(int64_t), hipMemcpyHostToDevice, stream);

    constexpr int BLOCK = 256;
    int grid = (total_out + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(gatherElementsKernel, dim3(grid), dim3(BLOCK), 0, stream,
                       (const char*)data, (const int64_t*)indices,
                       (char*)out, d_data_strides, d_out_strides,
                       axis, ndim, total_out, element_size);

    hipFreeAsync(d_data_strides, stream);
    hipFreeAsync(d_out_strides, stream);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
