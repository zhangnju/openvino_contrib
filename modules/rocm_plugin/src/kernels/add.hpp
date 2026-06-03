// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp" 
#include "details/elementwise_binary.hpp"
namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
struct AddOpImpl;

/**
 * Performs element-wise addition operation with two given tensors applying
 * broadcasting if needed.
 */
class Add {
public:
    Add(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block);

    void operator()(hipStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out) const;

private:
    ElementwiseBinary<AllElementTypesSwitch, AddOpImpl> impl_;
    /*
    Type_t element_type_;
    size_t num_elements_;
    size_t max_threads_per_block_;

    size_t num_blocks_;
    size_t threads_per_block_;
    */
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
