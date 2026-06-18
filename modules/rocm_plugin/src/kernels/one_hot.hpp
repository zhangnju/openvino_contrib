// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "details/rocm_type_traits.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

void launchOneHot(
    hipStream_t stream,
    const void* indices,
    bool indices_i32,
    float on_val,
    float off_val,
    int64_t depth,
    size_t num_indices,
    void* output,
    Type_t out_type);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
