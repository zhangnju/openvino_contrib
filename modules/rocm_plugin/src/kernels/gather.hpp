// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//#include <driver_types.h>
#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp" 
#include "details/rocm_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class Gather {
public:
    Gather(Type_t element_type,
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
           unsigned els_per_thread_dicts);

    void operator()(const hipStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const;

private:
    //template <typename IndexType>
    void CallByDataType(const hipStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const;

    //template <typename DataType, typename IndexType>
    void Call(const hipStream_t stream, const float* src_dict, const int* src_index, float* dst_data) const;
    void CallGeneric(const hipStream_t stream, const void* src_dict, const int32_t* src_index,
                     void* dst_data, size_t elem_bytes) const;

    Type_t element_type_;
    Type_t indices_type_;
    unsigned num_dicts_;
    unsigned index_range_;
    unsigned data_length_;
    unsigned indices_size_;
    bool gather_chunks_;
    unsigned blocks_per_grid_;
    unsigned threads_per_block_;
    unsigned grid_dim_x_;
    unsigned grid_dim_y_;
    unsigned dicts_batch_stride_;
    unsigned indices_batch_stride_;
    unsigned out_batch_stride_;
    unsigned els_per_thread_chunks_;
    unsigned els_per_thread_dicts_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
