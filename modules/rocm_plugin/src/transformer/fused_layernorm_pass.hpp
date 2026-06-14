// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Detects LayerNorm+Residual patterns and replaces with MIGraphX-compiled kernel.
// Targets the pattern: Add(x, skip) → LayerNorm(gamma, beta) → output
// This is the residual connection + layer normalization seen in BERT encoder layers.

#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class FusedLayerNormPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FusedLayerNormPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
