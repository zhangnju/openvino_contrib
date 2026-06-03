// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ov {
namespace rocm_gpu {
namespace kernel {

void launchTranspose(const void* src,
                     void* dst,
                     const int64_t* src_shape,
                     const int32_t* perm,
                     int32_t ndim,
                     size_t element_size,
                     hipStream_t stream);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
