// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>

#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp" 

#include "details/rocm_type_traits.hpp"
#include "details/numpy_broadcast_mapper.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class FakeQuantize {
public:
    FakeQuantize(Type_t element_type, std::size_t max_size, std::size_t threads_per_block, std::size_t levels);
    FakeQuantize(FakeQuantize&&) = default;
    FakeQuantize& operator=(FakeQuantize&&) = default;

    void operator()(const hipStream_t stream,
                    const void* arg,
                    const void* in_low,
                    const void* in_high,
                    const void* out_low,
                    const void* out_high,
                    const NumpyBroadcastMapper& in_low_mapper,
                    const NumpyBroadcastMapper& in_high_mapper,
                    const NumpyBroadcastMapper& out_low_mapper,
                    const NumpyBroadcastMapper& out_high_mapper,
                    void* out) const;

private:
    template <typename T>
    void Call(const hipStream_t stream,
              const void* arg,
              const void* in_low,
              const void* in_high,
              const void* out_low,
              const void* out_high,
              const NumpyBroadcastMapper& in_low_mapper,
              const NumpyBroadcastMapper& in_high_mapper,
              const NumpyBroadcastMapper& out_low_mapper,
              const NumpyBroadcastMapper& out_high_mapper,
              T levels_1,
              void* out) const;

    Type_t element_type_{};
    std::size_t max_size_{};
    std::size_t num_blocks_{};
    std::size_t threads_per_block_{};
    std::size_t levels_{};
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
