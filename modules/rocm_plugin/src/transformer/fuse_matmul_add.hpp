// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::rocm_gpu::pass {

class FullyConnectedTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedTransformation", "0");
    FullyConnectedTransformation();
};

// Removes MatMul nodes that became dead (no consumers) after FullyConnected fusion.
class DeadMatMulElimination : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("DeadMatMulElimination", "0");
    DeadMatMulElimination();
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::rocm_gpu::pass
