// Copyright (C) 2021-2023 Intel Corporation
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
//#include "details/elementwise_unary.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
struct EluOpImpl;

class Elu {
public:
    Elu(Type_t element_type, size_t max_threads_per_block, size_t num_elements, float alpha);

    void operator()(hipStream_t stream, const void* in, void* out) const;

private:
    //ElementwiseUnary<FloatElementTypesSwitch, EluOpImpl> impl_;
    float alpha_;

    Type_t element_type_;
    size_t num_elements_;
    size_t max_threads_per_block_;

    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
