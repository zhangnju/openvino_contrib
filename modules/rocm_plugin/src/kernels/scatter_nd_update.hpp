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
#include "details/numpy_broadcast_mapper.hpp" 
#include "details/error.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class ScatterNDUpdate {
public:
    ScatterNDUpdate(Type_t data_type,
                    Type_t indices_type,
                    size_t indices_last_dim,
                    size_t num_of_update_elements,
                    size_t num_of_elements,
                    size_t num_of_update_chunks,
                    size_t num_of_blocks,
                    size_t num_of_threads,
                    bool thread_per_element);

    void operator()(const hipStream_t stream,
                    const void* input,
                    const void* indices,
                    const void* updates,
                    const size_t* input_data_dim_pading,
                    void* output) const;


    void Call(const hipStream_t stream,
              const float* input,
              const unsigned int* indices,
              const float* updates,
              const size_t* input_data_dim_pading,
              float* output) const;


    void CallByDataType(const hipStream_t stream,
                        const void* input,
                        const unsigned int* indices,
                        const void* updates,
                        const size_t* input_data_dim_pading,
                        void* output) const;

private:
    Type_t data_type_;
    Type_t indices_type_;
    size_t indices_last_dim_;
    size_t num_of_update_elements_;
    size_t num_of_input_elements_;
    size_t num_of_update_chunks_;
    size_t num_of_blocks_;
    size_t num_of_threads_;
    bool thread_per_element_;

    size_t num_elements_;
    size_t max_threads_per_block_;

    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
