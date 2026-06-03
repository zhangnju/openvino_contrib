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
#include "details/rocm_type_traits.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class VarianceNormalizationFactor {
public:
    VarianceNormalizationFactor(unsigned blocks_number,
                                unsigned threads_per_block,
                                double epsilon,
                                size_t data_size,
                                Type_t data_type,
                                bool epsilon_inside_sqrt);

    void operator()(hipStream_t stream, void* data) const;

private:
    unsigned blocks_number_;
    unsigned threads_per_block_;
    double epsilon_;
    size_t data_size_;
    bool epsilon_inside_sqrt_;
    using TFuncPtr = void (*)(hipStream_t, unsigned, unsigned, double, size_t, float *,bool);
    TFuncPtr func_ptr_;
};

}  // namespace kernel

}  // namespace rocm_gpu
}  // namespace ov
