// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ov {
namespace rocm_gpu {
namespace kernel {

void launchGatherElements(const void* data,
                          const void* indices,
                          void* out,
                          const int64_t* data_strides,
                          const int64_t* out_strides,
                          int32_t axis,
                          int32_t ndim,
                          int64_t total_out,
                          size_t element_size,
                          size_t indices_element_size,
                          hipStream_t stream);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
