// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Custom OV node representing a fused LayerNorm operation.
// Created by LayerNormFusionPass.

#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

// Fused LayerNorm node.
// Inputs:
//   [0] x      — input tensor, shape [..., inner], dtype FP16 or FP32
//   [1] scale  — optional, shape [inner], same dtype as x
//   [2] bias   — optional, shape [inner], same dtype as x
// Output:
//   [0]        — normalized tensor, same shape and dtype as x
class LayerNorm : public ov::op::Op {
public:
    OPENVINO_OP("LayerNorm", "rocm_gpu");

    LayerNorm() = default;

    // Constructor with scale and bias
    LayerNorm(const ov::Output<ov::Node>& x,
              const ov::Output<ov::Node>& scale,
              const ov::Output<ov::Node>& bias,
              double epsilon, int64_t axis)
        : Op({x, scale, bias}), epsilon_(epsilon), axis_(axis) {
        constructor_validate_and_infer_types();
    }

    // Constructor without scale/bias
    LayerNorm(const ov::Output<ov::Node>& x, double epsilon, int64_t axis)
        : Op({x}), epsilon_(epsilon), axis_(axis) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        if (inputs.size() >= 3)
            return std::make_shared<LayerNorm>(inputs[0], inputs[1], inputs[2], epsilon_, axis_);
        return std::make_shared<LayerNorm>(inputs[0], epsilon_, axis_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("epsilon", epsilon_);
        visitor.on_attribute("axis", axis_);
        return true;
    }

    double  get_epsilon() const { return epsilon_; }
    int64_t get_axis()    const { return axis_; }

private:
    double  epsilon_{1e-5};
    int64_t axis_{-1};
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
