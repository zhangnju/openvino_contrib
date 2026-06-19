// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class GatherND {
public:
    GatherND(Type_t data_type,
             Type_t indices_type,
             size_t indices_last_dim,
             size_t num_of_gather_elements,
             size_t num_of_gather_chunks,
             size_t num_of_blocks,
             size_t num_of_threads,
             bool thread_per_element);

    void operator()(const hipStream_t stream,
                    const void* data,
                    const void* indices,
                    const size_t* data_dim_padding,
                    void* output) const;

private:
    template <typename DataType, typename IndexType>
    void Call(const hipStream_t stream,
              const void* data,
              const void* indices,
              const size_t* data_dim_padding,
              void* output) const;

    Type_t data_type_;
    Type_t indices_type_;
    size_t indices_last_dim_;
    size_t num_of_gather_elements_;
    size_t num_of_gather_chunks_;
    size_t num_of_blocks_;
    size_t num_of_threads_;
    bool thread_per_element_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
