// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// OneHot kernel: given indices tensor of shape [d0,...,dk] and depth N,
// produces output of shape [d0,...,dk,N] (axis=-1 / last axis).
// Each output[...][i] = on_val if i == indices[...] else off_val.

#include "hip/hip_runtime.h"
#include <hip/hip_fp16.h>
#include <cstdint>
#include "kernels/one_hot.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
static __global__ void oneHotKernel(
        const int64_t* indices,
        T on_val,
        T off_val,
        int64_t depth,
        size_t num_indices,
        T* output) {
    // Each thread handles one output row (length = depth)
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= num_indices) return;

    int64_t idx = indices[row];
    // Handle negative indices (wrap-around)
    if (idx < 0) idx += depth;

    T* out_row = output + row * depth;
    for (int64_t d = 0; d < depth; ++d) {
        out_row[d] = (d == idx) ? on_val : off_val;
    }
}

// Specialization for int32 indices
template <typename T>
static __global__ void oneHotKernelI32(
        const int32_t* indices,
        T on_val,
        T off_val,
        int64_t depth,
        size_t num_indices,
        T* output) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= num_indices) return;
    int64_t idx = static_cast<int64_t>(indices[row]);
    if (idx < 0) idx += depth;
    T* out_row = output + row * depth;
    for (int64_t d = 0; d < depth; ++d) {
        out_row[d] = (d == idx) ? on_val : off_val;
    }
}

void launchOneHot(
        const hipStream_t stream,
        const void* indices,
        bool indices_i32,
        float on_val_f,
        float off_val_f,
        int64_t depth,
        size_t num_indices,
        void* output,
        Type_t out_type) {
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((num_indices + BLOCK - 1) / BLOCK);
    if (grid == 0) return;

    switch (out_type) {
        case Type_t::f16: {
            __half on  = __float2half(on_val_f);
            __half off = __float2half(off_val_f);
            if (indices_i32)
                oneHotKernelI32<__half><<<grid, BLOCK, 0, stream>>>(
                    (const int32_t*)indices, on, off, depth, num_indices, (__half*)output);
            else
                oneHotKernel<__half><<<grid, BLOCK, 0, stream>>>(
                    (const int64_t*)indices, on, off, depth, num_indices, (__half*)output);
            break;
        }
        case Type_t::f32: {
            if (indices_i32)
                oneHotKernelI32<float><<<grid, BLOCK, 0, stream>>>(
                    (const int32_t*)indices, on_val_f, off_val_f, depth, num_indices, (float*)output);
            else
                oneHotKernel<float><<<grid, BLOCK, 0, stream>>>(
                    (const int64_t*)indices, on_val_f, off_val_f, depth, num_indices, (float*)output);
            break;
        }
        case Type_t::i32: {
            int32_t on  = static_cast<int32_t>(on_val_f);
            int32_t off = static_cast<int32_t>(off_val_f);
            if (indices_i32)
                oneHotKernelI32<int32_t><<<grid, BLOCK, 0, stream>>>(
                    (const int32_t*)indices, on, off, depth, num_indices, (int32_t*)output);
            else
                oneHotKernel<int32_t><<<grid, BLOCK, 0, stream>>>(
                    (const int64_t*)indices, on, off, depth, num_indices, (int32_t*)output);
            break;
        }
        case Type_t::i64: {
            int64_t on  = static_cast<int64_t>(on_val_f);
            int64_t off = static_cast<int64_t>(off_val_f);
            if (indices_i32)
                oneHotKernelI32<int64_t><<<grid, BLOCK, 0, stream>>>(
                    (const int32_t*)indices, on, off, depth, num_indices, (int64_t*)output);
            else
                oneHotKernel<int64_t><<<grid, BLOCK, 0, stream>>>(
                    (const int64_t*)indices, on, off, depth, num_indices, (int64_t*)output);
            break;
        }
        default:
            break;
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
