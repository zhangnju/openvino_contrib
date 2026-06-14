// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Native fused reduce+pointwise kernel launchers.
// Re-implemented from MIGraphX's design principles without MIGraphX API calls.
#pragma once
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Fused LayerNorm: (x+skip)→mean→var→rsqrt→scale+shift in single kernel
// skip may be nullptr (no residual add). x[rows,cols], gamma[cols], beta[cols] → y[rows,cols]
void launch_layernorm_fused(
    hipStream_t stream,
    const void* x, const void* skip, const void* gamma, const void* beta, void* y,
    int rows, int cols, float eps);

// Fused masked softmax: add(mask) → softmax along last dim
// scores[heads, sq, sk], mask[sq, sk] → out[heads, sq, sk]
void launch_masked_softmax_fused(
    hipStream_t stream,
    const void* scores, const void* mask, void* out,
    int heads, int sq, int sk);

// Fused bias+GELU: add bias then apply GELU elementwise
// gemm_out[seq, out_dim], bias[out_dim] → out[seq, out_dim]
void launch_bias_gelu_fused(
    hipStream_t stream,
    const void* gemm_out, const void* bias, void* out,
    int seq, int out_dim);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
