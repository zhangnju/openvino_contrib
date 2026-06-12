// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Custom OpenVINO node representing a chain of elementwise ops fused into
// a single GPU kernel. Created by the ElementwiseFusionTransformation pass.

#pragma once

#include <openvino/op/op.hpp>
#include <string>
#include <vector>

namespace ov {
namespace rocm_gpu {
namespace nodes {

// Each sub-op in the fused chain
struct FusedEwStep {
    std::string op_type;  // "Swish", "Sigmoid", "Mul", "Add", "Relu", etc.
    float       param;    // extra parameter (e.g., alpha for LeakyRelu, 0.f otherwise)
    bool        has_aux;  // true if this step needs a secondary input tensor
};

// FusedElementwise: custom OpenVINO node
//
// Inputs:
//   input 0:      primary tensor (the data stream being transformed)
//   input 1..N:   auxiliary tensors (for binary steps, in order of appearance)
//
// Outputs:
//   output 0: result tensor (same shape and type as input 0)
//
// Attribute: _steps — vector of FusedEwStep describing the chain
class FusedElementwise : public ov::op::Op {
public:
    OPENVINO_OP("FusedElementwise", "rocm_gpu");

    static constexpr int kMaxChain = 16;

    FusedElementwise() = default;

    FusedElementwise(const ov::OutputVector& inputs,
                     std::vector<FusedEwStep> steps)
        : Op(inputs), steps_(std::move(steps)) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // Output shape = BROADCAST of all inputs, same element type as primary input.
        // Bug fix: the original code used only input[0] shape, which is wrong when
        // the primary input is a scalar/constant and aux inputs have larger shapes.
        // Example: Sub(scalar_1.0, mask_tensor[1,1,256]) → output must be [1,1,256],
        // not [] or [1,1,1,1] (shape of the scalar). This caused BERT attention mask
        // reshapes to fail because FusedElementwise reported the wrong output shape.
        OPENVINO_ASSERT(get_input_size() >= 1, "FusedElementwise: at least one input required");
        ov::PartialShape out_shape = get_input_partial_shape(0);
        for (size_t i = 1; i < get_input_size(); ++i) {
            const auto& aux = get_input_partial_shape(i);
            if (!ov::PartialShape::broadcast_merge_into(
                    out_shape, aux, ov::op::AutoBroadcastType::NUMPY)) {
                // If shapes are incompatible, fall back to primary shape
                // (will produce a runtime error if truly incompatible)
                break;
            }
        }
        set_output_type(0, get_input_element_type(0), out_shape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_inputs) const override {
        return std::make_shared<FusedElementwise>(new_inputs, steps_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        // steps_ serialization omitted for now (not needed for single-session inference)
        return true;
    }

    const std::vector<FusedEwStep>& steps() const { return steps_; }

private:
    std::vector<FusedEwStep> steps_;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
