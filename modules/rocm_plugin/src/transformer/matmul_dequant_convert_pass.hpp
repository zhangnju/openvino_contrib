// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Retypes the INT8 GEMM (MatMulInteger, i32 output) directly to f32 and elides
// the immediately following Convert(i32->f32) of the dequant epilogue.
//
// Background: the plugin's MatMul executor never computes i32×i32→i32 — for an
// i32-input MatMul it casts the operands to f16 and accumulates in f32 (see
// matmul.cpp). It then casts the f32 result BACK to i32 (cast_f32_to_i32), only
// for the very next dequant op to Convert it i32->f32 again. That f32->i32->f32
// round-trip is pure overhead (one full-tensor cast kernel each way per GEMM).
//
// The executor already has the machinery to skip it: when the MatMul's OUTPUT
// element type is f32 it sets i32_out_is_f32_ and writes the f32 GEMM result
// straight to the op output (no cast_f32_to_i32). This pass is what makes that
// trigger — it rewrites each i32-output MatMul whose sole consumer is a
// Convert(i32->f32) into a TypeRelaxed<MatMul> with f32 output, then replaces
// the Convert with the MatMul's f32 output.
//
// Mirrors ForceF16MatMulOutput (force_f16_matmul_pass.cpp) which does the same
// i32->f16 trick for the fp16 path. Gated by ROCM_DISABLE_DEQUANT_CONVERT.

#pragma once

#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class MatMulDequantConvertPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MatMulDequantConvertPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
