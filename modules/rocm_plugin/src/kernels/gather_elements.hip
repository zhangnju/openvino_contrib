// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements.hpp"
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// GatherElements: out[i0,i1,...,axis=indices[i0,i1,...,in,...],in,...] = data[i0,i1,...,indices[i0,...],in,...]
// indices_element_size: size in bytes of each index (4=int32, 8=int64, 2=f16/int16)
__global__ void gatherElementsKernel(const char* __restrict__ data,
                                      const char* __restrict__ indices_bytes,
                                      char* __restrict__ out,
                                      const int64_t* __restrict__ data_strides,
                                      const int64_t* __restrict__ out_strides,
                                      int32_t axis,
                                      int32_t ndim,
                                      int64_t total_out,
                                      size_t element_size,
                                      size_t indices_element_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    // Read index value (handles int32 or int64 indices)
    auto read_index = [&](int64_t i) -> int64_t {
        const char* p = indices_bytes + i * indices_element_size;
        if (indices_element_size == 4) {
            int32_t v; __builtin_memcpy(&v, p, 4); return static_cast<int64_t>(v);
        } else if (indices_element_size == 8) {
            int64_t v; __builtin_memcpy(&v, p, 8); return v;
        } else {
            // Fallback: treat as int64 (may be wrong for f16 indices)
            int64_t v; __builtin_memcpy(&v, p, 8); return v;
        }
    };

    // decompose out_idx into coords
    int64_t remaining = out_idx;
    int64_t data_offset = 0;
    for (int d = 0; d < ndim; ++d) {
        int64_t coord = remaining / out_strides[d];
        remaining -= coord * out_strides[d];
        if (d == axis) {
            coord = read_index(out_idx);
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
                          size_t indices_element_size,
                          hipStream_t stream) {
    // data_strides and out_strides are already GPU pointers (pre-uploaded in InitSharedImmutableWorkbuffers)
    const int64_t* d_data_strides = data_strides;
    const int64_t* d_out_strides  = out_strides;

    constexpr int BLOCK = 256;
    int grid = (total_out + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(gatherElementsKernel, dim3(grid), dim3(BLOCK), 0, stream,
                       (const char*)data, (const char*)indices,
                       (char*)out, d_data_strides, d_out_strides,
                       axis, ndim, total_out, element_size, indices_element_size);

    // No cleanup needed: strides are owned by immutable workbuffers
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
