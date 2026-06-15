// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// ForceF16MatMulOutput: convert MatMul/FullyConnected outputs from f32 to f16
// when both inputs are already f16.
//
// MIGraphX --fp16 does this globally, eliminating the f32 accumulation boundary
// that OV's ConvertPrecision inserts. This removes ~32 Convert kernels per
// BERT inference and makes downstream FusedElementwise run in f16 instead of f32.
//
// Trade-off: f32 accumulation → f16 output (acceptable for BERT fp16 inference).
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class ForceF16MatMulOutput : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ForceF16MatMulOutput", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// EliminateF16ToF32Convert: remove Convert(f16->f32) nodes when running in f16
// inference mode. These "decompression" converts widen f16 MatMul outputs back
// to f32, causing downstream FusedElementwise chains to run in f32 instead of f16.
// The fix: bypass the Convert so consumers receive f16 directly.
class EliminateF16ToF32Convert : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("EliminateF16ToF32Convert", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
