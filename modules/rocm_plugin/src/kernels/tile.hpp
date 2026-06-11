// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Allocate device buffers for [out_strides | in_shape] (call once at construction).
void* allocTileDeviceBuffers(const int64_t* in_shape, const int64_t* out_shape, int32_t ndim);
void freeTileDeviceBuffers(void* device_buf);

// hipGraph-safe launch using pre-allocated device buffers.
// device_buf layout: [out_strides (ndim*8B) | in_shape (ndim*8B)]
void launchTile(const void* src,
                void* dst,
                void* device_buf,
                int64_t total_out,
                int32_t ndim,
                size_t element_size,
                hipStream_t stream);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
