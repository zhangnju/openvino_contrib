// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile.hpp"
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

__global__ void tileKernel(const char* __restrict__ src,
                            char* __restrict__ dst,
                            const int64_t* __restrict__ out_strides,
                            const int64_t* __restrict__ in_shape,
                            int32_t ndim,
                            int64_t total_out,
                            size_t element_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_out) return;

    // compute in_idx by taking each dim coord modulo in_shape[d]
    int64_t remaining = out_idx;
    int64_t in_idx = 0;
    int64_t in_stride = 1;
    // process dims from last to first for stride computation
    // we need in strides: in_stride[d] = product of in_shape[d+1..ndim-1]
    // pre-compute in strides on-the-fly:
    // Actually, compute by decomposing out_idx using out_strides
    // then fold each coord mod in_shape[d]
    for (int d = 0; d < ndim; ++d) {
        int64_t coord = remaining / out_strides[d];
        remaining -= coord * out_strides[d];
        coord = coord % in_shape[d];
        in_idx = in_idx * in_shape[d] + coord;
    }

    const char* src_elem = src + in_idx * element_size;
    char* dst_elem = dst + out_idx * element_size;
    for (size_t b = 0; b < element_size; ++b) dst_elem[b] = src_elem[b];
}

// device_buf layout: [out_strides (ndim*8B) | in_shape (ndim*8B)]
void* allocTileDeviceBuffers(const int64_t* in_shape, const int64_t* out_shape, int32_t ndim) {
    int64_t out_strides[8];
    out_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    const size_t sz = 2 * ndim * sizeof(int64_t);
    void* buf = nullptr;
    hipMalloc(&buf, sz);
    char* p = static_cast<char*>(buf);
    hipMemcpy(p,                         out_strides, ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    hipMemcpy(p + ndim * sizeof(int64_t), in_shape,   ndim * sizeof(int64_t), hipMemcpyHostToDevice);
    return buf;
}

void freeTileDeviceBuffers(void* device_buf) {
    if (device_buf) hipFree(device_buf);
}

void launchTile(const void* src,
                void* dst,
                void* device_buf,
                int64_t total_out,
                int32_t ndim,
                size_t element_size,
                hipStream_t stream) {
    char* p = static_cast<char*>(device_buf);
    const int64_t* d_out_strides = reinterpret_cast<const int64_t*>(p);
    const int64_t* d_in_shape    = reinterpret_cast<const int64_t*>(p + ndim * sizeof(int64_t));

    constexpr int BLOCK = 256;
    int grid = static_cast<int>((total_out + BLOCK - 1) / BLOCK);
    hipLaunchKernelGGL(tileKernel, dim3(grid), dim3(BLOCK), 0, stream,
                       (const char*)src, (char*)dst, d_out_strides, d_in_shape, ndim, total_out, element_size);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
