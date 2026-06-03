// Copyright (C) 2018-2021 Intel Corporation
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
#include "details/error.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class Split {
public:
    Split(Type_t element_type,
          size_t num_splits,
          size_t num_split_chunks,
          size_t split_step_size,
          unsigned num_blocks,
          unsigned threads_per_block);
    Split(Split&&) = default;
    Split& operator=(Split&&) = default;

    void operator()(hipStream_t stream, const void* src, void** dst) const;

    [[nodiscard]] size_t mutableWbSize() const { return sizeof(float*) * num_splits_; }

private:
    //template <typename T>
    void Call(hipStream_t stream, const float* src, float** dst) const;

    Type_t element_type_{};
    size_t num_splits_{};
    size_t num_split_chunks_{};
    size_t split_step_size_{};
    unsigned num_blocks_{};
    unsigned threads_per_block_{};
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
