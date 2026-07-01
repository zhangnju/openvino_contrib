// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Pre-allocate device buffers for strides/perm (call once at construction time).
// Returns combined allocation: [src_strides | dst_strides | perm] on device.
void* allocTransposeDeviceBuffers(const int64_t* src_shape, const int32_t* perm, int32_t ndim);
void freeTransposeDeviceBuffers(void* device_buf);

// Launch transpose using pre-allocated device stride buffers (hipGraph-safe: no hipMalloc).
// device_buf must be the pointer returned by allocTransposeDeviceBuffers.
// last_fixed_C: if > 0, perm keeps the last axis fixed and last_fixed_C is its size (C);
// the fast row-copy kernel (coalesced, no per-element int division) is used. If 0, the
// generic per-element kernel is used.
void launchTranspose(const void* src,
                     void* dst,
                     void* device_buf,       // pre-allocated: [src_strides|dst_strides|perm]
                     int64_t total_elements,
                     int32_t ndim,
                     size_t element_size,
                     hipStream_t stream,
                     int64_t last_fixed_C = 0,
                     int32_t n_lead = 0);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
