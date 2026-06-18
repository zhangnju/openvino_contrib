// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Native FP16/FP32 LayerNorm kernel.
// Fuses: ReduceMean + Sub + Mul(square) + ReduceMean + Add(eps) + Rsqrt + Mul(scale) + Add(bias)
// Uses one threadblock per row (the normalized dimension), warp-level parallel reductions.

#pragma once
#include <cstddef>
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Launch LayerNorm for a tensor of shape [outer, inner]:
//   For each row r in [0, outer):
//     mean = sum(x[r, :]) / inner
//     var  = sum((x[r, :] - mean)^2) / inner
//     y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * scale[:] + bias[:]
//
// scale and bias are optional (pass nullptr to skip).
// T = float or __half (FP16).
// Accumulates mean/variance in float for numerical stability regardless of T.
void launchLayerNorm(
    hipStream_t stream,
    const void* x,        // input  [outer, inner], type T
    void*       y,        // output [outer, inner], type T
    const void* scale,    // optional scale (gamma) [inner], type T (nullptr = skip)
    const void* bias,     // optional bias  (beta)  [inner], type T (nullptr = skip)
    const void* add_bias, // optional pre-LN additive bias [inner] broadcast over rows (nullptr = skip)
    const void* residual, // optional pre-LN residual/skip [outer,inner] (nullptr = skip)
    size_t      outer,
    size_t      inner,
    float       epsilon,
    bool        is_fp16);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
