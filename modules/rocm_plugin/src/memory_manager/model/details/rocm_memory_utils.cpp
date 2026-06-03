// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/runtime.hpp>

#include "memory_manager/model/rocm_memory_model.hpp"

namespace ov {
namespace rocm_gpu {

size_t applyAllignment(size_t value) {
    constexpr size_t allignment = rocm::memoryAlignment;
    return (value % allignment) == 0 ? value : value - (value % allignment) + allignment;
}

}  // namespace rocm_gpu
}  // namespace ov
