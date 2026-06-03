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
#include "details/rocm_type_traits.hpp"
#include "details/element_types_switch.hpp"
#include "details/numpy_broadcast_mapper.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class Broadcast {
public:
    Broadcast(ov::rocm_gpu::kernel::Type_t element_type, size_t dst_num_elements, size_t max_threads_per_block);

    void operator()(const hipStream_t stream,
                    const void* src,
                    const NumpyBroadcastMapper& src_mapper,
                    void* dst) const;

private:
    friend AllElementTypesSwitch;

    template <typename T>
    constexpr void case_(hipStream_t, const void*, const NumpyBroadcastMapper&, void*) const noexcept;

    template <typename T>
    void default_(T t, hipStream_t, const void*, const NumpyBroadcastMapper&, void*) const noexcept;

private:
    Type_t element_type_;
    size_t dst_num_elements_;
    size_t num_blocks_;
    size_t threads_per_block_;

};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
