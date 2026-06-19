// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <fmt/format.h>

#include <rocm/float16.hpp>

#include "gather_nd.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename DataType, typename IndexType>
static inline __device__ void gather_nd(const size_t indices_last_dim,
                                        const size_t num_of_gather_elements,
                                        const size_t* data_dim_padding,
                                        const size_t gather_element,
                                        const size_t gather_chunk,
                                        const IndexType* indices,
                                        const DataType* data,
                                        DataType* output) {
    const auto begin = gather_chunk * indices_last_dim;
    const auto end = begin + indices_last_dim;
    size_t in_index{};
    for (size_t j{begin}, k{}; j < end; ++j, ++k) in_index += indices[j] * data_dim_padding[k];

    output[gather_chunk * num_of_gather_elements + gather_element] = data[in_index + gather_element];
}

template <typename DataType, typename IndexType>
static __global__ void gather_thread_per_element(const size_t indices_last_dim,
                                                 const size_t num_of_gather_elements,
                                                 const size_t* data_dim_padding,
                                                 const IndexType* indices,
                                                 const DataType* data,
                                                 DataType* output) {
    const auto gather_element = blockIdx.y * blockDim.x + threadIdx.x;
    const auto gather_chunk = blockIdx.x;

    if (gather_element >= num_of_gather_elements) return;

    gather_nd<DataType, IndexType>(indices_last_dim,
                                   num_of_gather_elements,
                                   data_dim_padding,
                                   gather_element,
                                   gather_chunk,
                                   indices,
                                   data,
                                   output);
}

template <typename DataType, typename IndexType>
static __global__ void gather_thread_per_chunk(const size_t indices_last_dim,
                                               const size_t num_of_gather_elements,
                                               const size_t num_of_gather_chunks,
                                               const size_t* data_dim_padding,
                                               const IndexType* indices,
                                               const DataType* data,
                                               DataType* output) {
    const auto gather_chunk = blockIdx.y * blockDim.x + threadIdx.x;
    const auto gather_element = blockIdx.x;

    if (gather_chunk >= num_of_gather_chunks) return;

    gather_nd<DataType, IndexType>(indices_last_dim,
                                   num_of_gather_elements,
                                   data_dim_padding,
                                   gather_element,
                                   gather_chunk,
                                   indices,
                                   data,
                                   output);
}

GatherND::GatherND(Type_t data_type,
                   Type_t indices_type,
                   size_t indices_last_dim,
                   size_t num_of_gather_elements,
                   size_t num_of_gather_chunks,
                   size_t num_of_blocks,
                   size_t num_of_threads,
                   bool thread_per_element)
    : data_type_(data_type),
      indices_type_(indices_type),
      indices_last_dim_(indices_last_dim),
      num_of_gather_elements_(num_of_gather_elements),
      num_of_gather_chunks_(num_of_gather_chunks),
      num_of_blocks_(num_of_blocks),
      num_of_threads_(num_of_threads),
      thread_per_element_(thread_per_element) {}

template <typename DataType, typename IndexType>
void GatherND::Call(const hipStream_t stream,
                    const void* data,
                    const void* indices,
                    const size_t* data_dim_padding,
                    void* output) const {
    const auto data_typed = static_cast<const DataType*>(data);
    const auto indices_typed = static_cast<const IndexType*>(indices);
    auto output_typed = static_cast<DataType*>(output);

    if (thread_per_element_) {
        dim3 grid{static_cast<unsigned int>(num_of_gather_chunks_), static_cast<unsigned int>(num_of_blocks_)};
        gather_thread_per_element<DataType, IndexType><<<grid, num_of_threads_, 0, stream>>>(
            indices_last_dim_, num_of_gather_elements_, data_dim_padding, indices_typed, data_typed, output_typed);
    } else {
        dim3 grid{static_cast<unsigned int>(num_of_gather_elements_), static_cast<unsigned int>(num_of_blocks_)};
        gather_thread_per_chunk<DataType, IndexType><<<grid, num_of_threads_, 0, stream>>>(indices_last_dim_,
                                                                                           num_of_gather_elements_,
                                                                                           num_of_gather_chunks_,
                                                                                           data_dim_padding,
                                                                                           indices_typed,
                                                                                           data_typed,
                                                                                           output_typed);
    }
}

void GatherND::operator()(const hipStream_t stream,
                          const void* data,
                          const void* indices,
                          const size_t* data_dim_padding,
                          void* output) const {
    switch (indices_type_) {
        case Type_t::i32:
        case Type_t::u32:
            switch (data_type_) {
                case Type_t::f32:
                    return Call<float, int32_t>(stream, data, indices, data_dim_padding, output);
                case Type_t::f16:
                    return Call<__half, int32_t>(stream, data, indices, data_dim_padding, output);
                case Type_t::i32:
                    return Call<int32_t, int32_t>(stream, data, indices, data_dim_padding, output);
                case Type_t::i8:
                case Type_t::u8:
                    return Call<int8_t, int32_t>(stream, data, indices, data_dim_padding, output);
                default:
                    throw_ov_exception(fmt::format(
                        "Data element type = {} is not supported by GatherND operation !!", data_type_));
            }
        case Type_t::i64:
        case Type_t::u64:
            switch (data_type_) {
                case Type_t::f32:
                    return Call<float, int64_t>(stream, data, indices, data_dim_padding, output);
                case Type_t::f16:
                    return Call<__half, int64_t>(stream, data, indices, data_dim_padding, output);
                case Type_t::i32:
                    return Call<int32_t, int64_t>(stream, data, indices, data_dim_padding, output);
                case Type_t::i8:
                case Type_t::u8:
                    return Call<int8_t, int64_t>(stream, data, indices, data_dim_padding, output);
                default:
                    throw_ov_exception(fmt::format(
                        "Data element type = {} is not supported by GatherND operation !!", data_type_));
            }
        default:
            throw_ov_exception(
                fmt::format("Index element type = {} is not supported by GatherND operation !!", indices_type_));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
