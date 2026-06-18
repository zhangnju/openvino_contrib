// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Custom OV node fusing the global min/max statistics of the onnx
// DynamicQuantizeLinear decomposition. After OV's CommonOptimizations folds the
// QDQ chain, each activation-quant point still computes:
//   xmax = Maximum(0, ReduceMax(x));  xmin = Minimum(0, ReduceMin(x))
//   span = Subtract(xmax, xmin)
// i.e. TWO full-tensor reductions feeding a Subtract. This node replaces that
// sub-graph with a single op whose kernel makes ONE pass over x computing both
// min and max, then emits span = max(0,max) - min(0,min) as a scalar.

#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

// Fused dynamic-quant statistics node.
// Input:  [0] x     — activation tensor, f32, any shape
// Outputs:
//   [0] span  — scalar f32, = max(0,max(x)) - min(0,min(x))
//   [1] xmin  — scalar f32, = min(0,min(x))   (reused by the zero_point path)
// Emitting both lets the pass eliminate BOTH the ReduceMax and ReduceMin
// reductions (xmin is otherwise still needed for zero_point = round(-xmin/scale)).
class DynamicQuantizeStats : public ov::op::Op {
public:
    OPENVINO_OP("DynamicQuantizeStats", "rocm_gpu");

    DynamicQuantizeStats() = default;

    explicit DynamicQuantizeStats(const ov::Output<ov::Node>& x) : Op({x}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::f32, ov::PartialShape{1});
        set_output_type(1, ov::element::f32, ov::PartialShape{1});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<DynamicQuantizeStats>(inputs[0]);
    }

    bool visit_attributes(ov::AttributeVisitor&) override { return true; }
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
