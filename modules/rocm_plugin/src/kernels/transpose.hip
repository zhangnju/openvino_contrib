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

// Fast path for perm[ndim-1] == ndim-1 (last axis unchanged): the transpose is a
// permutation of the leading dims with the contiguous last axis (C) copied as a block.
// Vectorized: each thread copies VEC elements (half4 = 4×f16) along the contiguous C axis.
// This gives 4× fewer threads and coalesced 8-byte reads/writes.
// Vectorized transpose: each thread copies 4 f16 (8 bytes) along contiguous C axis.
// src_base and C are in f16 elements; src layout has C contiguous.
__global__ void transposeKernelLastFixedVec4(const __half* __restrict__ src,
                                             __half* __restrict__ dst,
                                             const int64_t* __restrict__ src_strides,
                                             const int64_t* __restrict__ dst_strides,
                                             const int32_t* __restrict__ perm,
                                             int64_t C,
                                             int64_t total_rows,
                                             int64_t total_vec_elems,
                                             int32_t n_lead) {
    const int64_t vid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_vec_elems) return;
    constexpr int VEC = 4;
    const int64_t C_vec = C / VEC;
    const int64_t vec_col = vid % C_vec;
    const int64_t row = vid / C_vec;
    int64_t rem = row;
    int64_t src_base = 0;
    #pragma unroll 1
    for (int d = 0; d < n_lead; ++d) {
        int64_t row_stride = dst_strides[d] / C;
        int64_t c = rem / row_stride;
        rem -= c * row_stride;
        src_base += c * src_strides[perm[d]];
    }
    const uint2* src4 = reinterpret_cast<const uint2*>(src + src_base);
    uint2* dst4 = reinterpret_cast<uint2*>(dst + row * C);
    dst4[vec_col] = src4[vec_col];
}

// Scalar fallback for non-vectorizable cases
template <typename T>
__global__ void transposeKernelLastFixed(const T* __restrict__ src,
                                         T* __restrict__ dst,
                                         const int64_t* __restrict__ src_strides,
                                         const int64_t* __restrict__ dst_strides,
                                         const int32_t* __restrict__ perm,
                                         int64_t C,
                                         int64_t total_elements,
                                         int32_t n_lead) {
    const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total_elements) return;
    const int64_t col = id % C;
    int64_t rem = id - col;
    int64_t src_base = 0;
    #pragma unroll 1
    for (int d = 0; d < n_lead; ++d) {
        int64_t c = rem / dst_strides[d];
        rem -= c * dst_strides[d];
        src_base += c * src_strides[perm[d]];
    }
    dst[id] = src[src_base + col];
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
                     hipStream_t stream,
                     int64_t last_fixed_C,
                     int32_t n_lead) {
    char* p = static_cast<char*>(device_buf);
    const int64_t* d_src_strides = reinterpret_cast<const int64_t*>(p);
    const int64_t* d_dst_strides = reinterpret_cast<const int64_t*>(p + ndim * sizeof(int64_t));
    const int32_t* d_perm        = reinterpret_cast<const int32_t*>(p + 2 * ndim * sizeof(int64_t));

    if (last_fixed_C > 0 && ndim >= 2 && n_lead > 0) {
        constexpr int BLOCK = 256;
        int64_t total_rows = total_elements / last_fixed_C;

        // Vectorized path for fp16: copy 4 halfs (8 bytes) per thread
        if (element_size == 2 && last_fixed_C % 4 == 0) {
            constexpr int VEC = 4;
            int64_t vec_elems = total_rows * (last_fixed_C / VEC);
            int grid = (int)((vec_elems + BLOCK - 1) / BLOCK);
            hipLaunchKernelGGL(transposeKernelLastFixedVec4, dim3(grid), dim3(BLOCK), 0, stream,
                               (const __half*)src, (__half*)dst, d_src_strides, d_dst_strides, d_perm,
                               last_fixed_C, total_rows, vec_elems, n_lead);
            return;
        }

        // Scalar fallback
        int grid = (int)((total_elements + BLOCK - 1) / BLOCK);
        if (element_size == 2) {
            hipLaunchKernelGGL(transposeKernelLastFixed<__half>, dim3(grid), dim3(BLOCK), 0, stream,
                               (const __half*)src, (__half*)dst, d_src_strides, d_dst_strides, d_perm, last_fixed_C, total_elements, n_lead);
        } else if (element_size == 4) {
            hipLaunchKernelGGL(transposeKernelLastFixed<float>, dim3(grid), dim3(BLOCK), 0, stream,
                               (const float*)src, (float*)dst, d_src_strides, d_dst_strides, d_perm, last_fixed_C, total_elements, n_lead);
        } else if (element_size == 8) {
            hipLaunchKernelGGL(transposeKernelLastFixed<double>, dim3(grid), dim3(BLOCK), 0, stream,
                               (const double*)src, (double*)dst, d_src_strides, d_dst_strides, d_perm, last_fixed_C, total_elements, n_lead);
        } else if (element_size == 1) {
            hipLaunchKernelGGL(transposeKernelLastFixed<uint8_t>, dim3(grid), dim3(BLOCK), 0, stream,
                               (const uint8_t*)src, (uint8_t*)dst, d_src_strides, d_dst_strides, d_perm, last_fixed_C, total_elements, n_lead);
        }
        return;
    }

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
