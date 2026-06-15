#include "hip/hip_runtime.h"
#include <vector>
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_runtime.h>
#include <fmt/format.h>

#include <rocm/float16.hpp>

//#include "details/type_validator.hpp"
#include "gather.hpp"

namespace ov {
namespace rocm_gpu {

namespace kernel {

//template <typename DataType, typename IndexType>
static inline __device__ void gather(unsigned data_length,
                                     size_t index_range,
                                     unsigned els_per_thread,
                                     unsigned indices_size,
                                     unsigned indices_index,
                                     unsigned dict,
                                     unsigned chunk,
                                     const float* src_dict,
                                     const int* src_index,
                                     float* dst_data) {
    auto dict_index = src_index[indices_index];
    if (dict_index < 0) {
        dict_index += index_range;
    }
    unsigned thread_offset;
    for (int el = 0; el < els_per_thread; ++el) {
        thread_offset = chunk + el;
        if (thread_offset >= data_length) {
            return;
        }
        const auto dst_index = data_length * (indices_index + dict * indices_size) + thread_offset;
        if (dict_index < index_range) {
            const auto src_index = data_length * (dict_index + dict * index_range) + thread_offset;
            dst_data[dst_index] = src_dict[src_index];
        } else {
            dst_data[dst_index] = 0;
        }
    }
}

//template <typename DataType, typename IndexType>
static __global__ void chunks_gather(unsigned data_length,
                                     unsigned indices_size,
                                     size_t index_range,
                                     unsigned dicts_batch_stride,
                                     unsigned indices_batch_stride,
                                     unsigned out_batch_stride,
                                     unsigned els_per_thread,
                                     const float* src_dict,
                                     const int* src_index,
                                     float* dst_data) {
    const auto dict = blockIdx.y;
    const auto indices_index = blockIdx.x % indices_size;
    const auto batch = blockIdx.x / indices_size;
    const auto chunk = (blockIdx.z * blockDim.x + threadIdx.x) * els_per_thread;
    gather(data_length,
           index_range,
           els_per_thread,
           indices_size,
           indices_index,
           dict,
           chunk,
           src_dict + batch * dicts_batch_stride,
           src_index + batch * indices_batch_stride,
           dst_data + batch * out_batch_stride);
}

//template <typename DataType, typename IndexType>
static __global__ void dicts_gather(unsigned num_dicts,
                                    unsigned indices_size,
                                    size_t index_range,
                                    unsigned dicts_batch_stride,
                                    unsigned indices_batch_stride,
                                    unsigned out_batch_stride,
                                    unsigned els_per_thread,
                                    const float* src_dict,
                                    const int* src_index,
                                    float* dst_data) {
    const auto data_length = gridDim.y;
    const auto chunk = blockIdx.y * els_per_thread;
    const auto dict = blockIdx.z * blockDim.x + threadIdx.x;
    if (dict >= num_dicts) {
        return;
    }
    const auto indices_index = blockIdx.x % indices_size;
    const auto batch = blockIdx.x / indices_size;
    gather(data_length,
           index_range,
           els_per_thread,
           indices_size,
           indices_index,
           dict,
           chunk,
           src_dict + batch * dicts_batch_stride,
           src_index + batch * indices_batch_stride,
           dst_data + batch * out_batch_stride);
}

Gather::Gather(Type_t element_type,
               Type_t indices_type,
               unsigned num_dicts,
               unsigned index_range,
               unsigned data_length,
               unsigned indices_size,
               bool gather_chunks,
               unsigned blocks_per_grid,
               unsigned threads_per_block,
               unsigned grid_dim_x,
               unsigned grid_dim_y,
               unsigned dicts_batch_stride,
               unsigned indices_batch_stride,
               unsigned out_batch_stride,
               unsigned els_per_thread_chunks,
               unsigned els_per_thread_dicts)
    : element_type_(element_type),
      indices_type_(indices_type),
      num_dicts_(num_dicts),
      index_range_(index_range),
      data_length_(data_length),
      indices_size_(indices_size),
      gather_chunks_(gather_chunks),
      blocks_per_grid_(blocks_per_grid),
      threads_per_block_(threads_per_block),
      grid_dim_x_(grid_dim_x),
      grid_dim_y_(grid_dim_y),
      dicts_batch_stride_(dicts_batch_stride),
      indices_batch_stride_(indices_batch_stride),
      out_batch_stride_(out_batch_stride),
      els_per_thread_chunks_(els_per_thread_chunks),
      els_per_thread_dicts_(els_per_thread_dicts) {
    //TypeValidator<AllElementTypesSwitch>::check(element_type_);
    //TypeValidator<ElementTypesSwitch<Type_t::i64, Type_t::i32>>::check(indices_type_);
}

void Gather::operator()(const hipStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const {
    switch (indices_type_) {
        case Type_t::i64: {
            // Cast i64 indices to i32 temporarily — safe as long as index values fit in int32
            size_t n = indices_size_;
            std::vector<int32_t> idx32(n);
            hipStream_t s = const_cast<hipStream_t>(stream);
            std::vector<int64_t> host_idx(n);
            hipMemcpyAsync(host_idx.data(), src_index, n * sizeof(int64_t), hipMemcpyDeviceToHost, s);
            hipStreamSynchronize(s);
            for (size_t k = 0; k < n; ++k) idx32[k] = static_cast<int32_t>(host_idx[k]);
            int32_t* d_idx32 = nullptr;
            hipMalloc(&d_idx32, n * sizeof(int32_t));
            hipMemcpyAsync(d_idx32, idx32.data(), n * sizeof(int32_t), hipMemcpyHostToDevice, s);
            CallByDataType(stream, src_dict, d_idx32, dst_data);
            hipFreeAsync(d_idx32, s);
            return;
        }
        case Type_t::u32:
        case Type_t::i32:
            return CallByDataType(stream, src_dict, src_index, dst_data);
        default:
            throw_ov_exception(
                fmt::format("Index element type = {} is not supported by Gather operation !!", indices_type_));
    }
}

//template <typename IndexType>
void Gather::CallByDataType(const hipStream_t stream,
                            const void* src_dict,
                            const void* src_index,
                            void* dst_data) const {
    const float* src_dict_fp32 = nullptr;
    const int* src_index_int= nullptr;
    float* dst_data_fp32= nullptr;
    switch (element_type_) {
        case Type_t::f16:
            // FP16 gather: vectorized wavefront-per-row kernel (float4/half2 loads)
            return CallFP16(stream, (const __half*)src_dict, (const int32_t*)src_index, (__half*)dst_data);
        case Type_t::f32:
            return Call(stream, (const float*)src_dict, (const int*)src_index, (float*)dst_data);
        case Type_t::i64:
            return CallGeneric(stream, src_dict, (const int32_t*)src_index, dst_data, sizeof(int64_t));
        case Type_t::i32:
            return CallGeneric(stream, src_dict, (const int32_t*)src_index, dst_data, sizeof(int32_t));
        case Type_t::u8:
            return CallGeneric(stream, src_dict, (const int32_t*)src_index, dst_data, sizeof(uint8_t));
        default:
            throw_ov_exception(
                fmt::format("Data element type = {} is not supported by Gather operation !!", element_type_));
    }
}

//template <typename DataType, typename IndexType>
void Gather::Call(const hipStream_t stream, const float* src_dict, const int* src_index, float* dst_data) const {
    dim3 grid{grid_dim_x_, grid_dim_y_, blocks_per_grid_};

    const auto src_dict_typed = static_cast<const float*>(src_dict);
    const auto src_index_typed = static_cast<const int*>(src_index);
    auto dst_data_typed = static_cast<float*>(dst_data);

    if (gather_chunks_) {
        kernel::chunks_gather<<<grid, threads_per_block_, 0, stream>>>(data_length_,
                                                                       indices_size_,
                                                                       index_range_,
                                                                       dicts_batch_stride_,
                                                                       indices_batch_stride_,
                                                                       out_batch_stride_,
                                                                       els_per_thread_chunks_,
                                                                       src_dict_typed,
                                                                       src_index_typed,
                                                                       dst_data_typed);
    } else {
        kernel::dicts_gather<<<grid, threads_per_block_, 0, stream>>>(num_dicts_,
                                                                      indices_size_,
                                                                      index_range_,
                                                                      dicts_batch_stride_,
                                                                      indices_batch_stride_,
                                                                      out_batch_stride_,
                                                                      els_per_thread_dicts_,
                                                                      src_dict_typed,
                                                                      src_index_typed,
                                                                      dst_data_typed);
    }
    // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly in rocm Plugin
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw_ov_exception(hipGetErrorString(err));
    }
}

// Optimized FP16 gather kernel: vectorized float4 loads (16 bytes = 8 halves per load)
// Each wavefront (64 threads on RDNA) handles one (dict, indices_index) pair — one row of data_length elements.
// float4 loads give coalesced 16-byte reads vs the generic_gather byte-at-a-time approach.
//
// Grid: x = ceil(indices_size / waves_per_block), y = num_dicts, z = batch_count
// Block: 256 threads = 4 wavefronts of 64
static __global__ void gather_fp16_vectorized(const __half* __restrict__ src_dict,
                                               const int32_t* __restrict__ src_index,
                                               __half* __restrict__ dst_data,
                                               unsigned data_length,      // hidden dim (e.g. 768)
                                               size_t index_range,        // vocab size (axis dim)
                                               unsigned indices_size,     // seq length
                                               unsigned dicts_batch_stride,
                                               unsigned indices_batch_stride,
                                               unsigned out_batch_stride) {
    // 4 wavefronts of 64 threads per block = 256 threads
    const int waves_per_block = blockDim.x / 64;
    const int wave_id = threadIdx.x / 64;
    const int lane    = threadIdx.x % 64;
    const int indices_index = blockIdx.x * waves_per_block + wave_id;
    const int dict    = blockIdx.y;
    const int batch   = blockIdx.z;

    if (indices_index >= indices_size) return;

    int32_t dict_idx = src_index[batch * indices_batch_stride + indices_index];
    if (dict_idx < 0) dict_idx += (int32_t)index_range;

    // Output: dst[batch][dict * indices_size + indices_index][data_length]
    __half* dst = dst_data + (size_t)batch * out_batch_stride
                           + (size_t)(indices_index + dict * indices_size) * data_length;

    if (dict_idx >= 0 && (size_t)dict_idx < index_range) {
        // Source: src_dict[batch][dict_idx + dict * index_range][data_length]
        const __half* src = src_dict + (size_t)batch * dicts_batch_stride
                                     + (size_t)(dict_idx + dict * index_range) * data_length;

        // Use float4 (16-byte) vectorized loads when data_length is divisible by 8
        if ((data_length & 7) == 0) {
            const unsigned len8 = data_length >> 3;
            const float4* src4 = reinterpret_cast<const float4*>(src);
            float4* dst4 = reinterpret_cast<float4*>(dst);
            for (unsigned j = lane; j < len8; j += 64)
                dst4[j] = src4[j];
        }
        // Use half2 (4-byte) vectorized loads when data_length is divisible by 2
        else if ((data_length & 1) == 0) {
            const unsigned len2 = data_length >> 1;
            const __half2* src2 = reinterpret_cast<const __half2*>(src);
            __half2* dst2 = reinterpret_cast<__half2*>(dst);
            for (unsigned j = lane; j < len2; j += 64)
                dst2[j] = src2[j];
        }
        // Scalar fallback for odd data_length
        else {
            for (unsigned j = lane; j < data_length; j += 64)
                dst[j] = src[j];
        }
    } else {
        // Out of range: zero fill
        if ((data_length & 7) == 0) {
            const unsigned len8 = data_length >> 3;
            float4* dst4 = reinterpret_cast<float4*>(dst);
            float4 zero4 = {0.f, 0.f, 0.f, 0.f};
            for (unsigned j = lane; j < len8; j += 64)
                dst4[j] = zero4;
        } else {
            for (unsigned j = lane; j < data_length; j += 64)
                dst[j] = __float2half(0.f);
        }
    }
}

// Generic byte-level gather for any data type with int32 indices
static __global__ void generic_gather(const char* __restrict__ src_dict,
                                       const int32_t* __restrict__ src_index,
                                       char* __restrict__ dst_data,
                                       unsigned data_length_bytes,
                                       size_t index_range,
                                       unsigned indices_size,
                                       unsigned dicts_batch_stride_elems,
                                       unsigned indices_batch_stride,
                                       unsigned out_batch_stride_elems,
                                       size_t elem_bytes) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= indices_size) return;
    const auto batch = blockIdx.y;
    int32_t dict_idx = src_index[batch * indices_batch_stride + idx];
    if (dict_idx < 0) dict_idx += (int32_t)index_range;
    const char* src = src_dict + batch * dicts_batch_stride_elems * elem_bytes + dict_idx * data_length_bytes;
    char* dst = dst_data + (batch * out_batch_stride_elems + idx * (data_length_bytes / elem_bytes)) * elem_bytes;
    for (unsigned b = threadIdx.y; b < data_length_bytes; b += blockDim.y) {
        dst[b] = (dict_idx >= 0 && (size_t)dict_idx < index_range) ? src[b] : 0;
    }
}

void Gather::CallFP16(const hipStream_t stream, const __half* src_dict, const int32_t* src_index,
                       __half* dst_data) const {
    // 4 wavefronts of 64 threads = 256 threads per block
    constexpr int waves_per_block = 4;
    constexpr int threads = waves_per_block * 64;  // 256
    // grid_dim_x_ = indices_size * batch_count (from the op constructor)
    unsigned batch_count = grid_dim_x_ / indices_size_;
    if (batch_count == 0) batch_count = 1;
    // Grid: x = ceil(indices_size / waves_per_block), y = num_dicts, z = batch_count
    dim3 block(threads, 1, 1);
    dim3 grid((indices_size_ + waves_per_block - 1) / waves_per_block, num_dicts_, batch_count);
    hipLaunchKernelGGL(gather_fp16_vectorized, grid, block, 0, stream,
                       src_dict, src_index, dst_data,
                       data_length_, (size_t)index_range_,
                       indices_size_, dicts_batch_stride_, indices_batch_stride_, out_batch_stride_);
}

void Gather::CallGeneric(const hipStream_t stream, const void* src_dict, const int32_t* src_index,
                          void* dst_data, size_t elem_bytes) const {
    dim3 block(std::min((unsigned)256u, (unsigned)indices_size_), 1, 1);
    dim3 grid((indices_size_ + block.x - 1) / block.x, 1, 1);
    hipLaunchKernelGGL(generic_gather, grid, block, 0, stream,
                       (const char*)src_dict, src_index, (char*)dst_data,
                       (unsigned)(data_length_ * elem_bytes), (size_t)index_range_,
                       indices_size_, dicts_batch_stride_, indices_batch_stride_, out_batch_stride_,
                       elem_bytes);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
