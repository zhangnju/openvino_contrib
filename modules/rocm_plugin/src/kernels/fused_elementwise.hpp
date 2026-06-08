// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Op kind enum — values are stored in a GPU buffer as uint8_t.
// Order must match fused_ew_apply() switch in fused_elementwise.hip.
enum class FusedEwOp : uint8_t {
    Relu        = 0,
    Sigmoid     = 1,
    SiLU        = 2,   // x * sigmoid(x)
    Tanh        = 3,
    Gelu        = 4,
    HardSwish   = 5,
    Abs         = 6,
    Neg         = 7,
    Sqrt        = 8,
    Exp         = 9,
    Log         = 10,
    Erf         = 11,
    Add         = 12,  // binary, reads aux_ptrs[step]
    Sub         = 13,
    Mul         = 14,
    Div         = 15,
    LeakyRelu   = 16,  // param = alpha
};

static constexpr int kFusedEwMaxChain = 16;

// Launch the fused elementwise kernel.
// All GPU pointer args must be on device.
// ops_device:    array of kFusedEwOp values (uint8_t), length = chain_len
// params_device: float parameters per op (0.0f if unused), length = chain_len
// aux_ptrs_device: device pointer to array of device pointers (const void*[chain_len])
//                  aux_ptrs_device[s] == nullptr for unary ops
void launch_fused_elementwise(
    const void* primary_in,
    const void* const* aux_ptrs_device,
    void* out,
    int64_t n,
    const uint8_t* ops_device,
    const float* params_device,
    int chain_len,
    bool is_fp16,
    hipStream_t stream);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
