// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

// Detects BERT-style LayerNorm patterns and replaces with a single LayerNorm op.
// Matches: ReduceMean → Sub → Mul(sq) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul
//          with optional Identity/StopGrad nodes and optional Mul(scale) + Add(bias).
class LayerNormFusionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("LayerNormFusionPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
