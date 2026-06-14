// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedLayerNorm: custom OV node for MIGraphX-compiled fused LayerNorm+Residual.
//
// Inputs (has_residual=true):
//   [0] x      [seq, hidden] f16
//   [1] skip   [seq, hidden] f16  — residual input
//   [2] gamma  [hidden]      f16
//   [3] beta   [hidden]      f16
//
// Inputs (has_residual=false):
//   [0] x      [seq, hidden] f16
//   [1] gamma  [hidden]      f16
//   [2] beta   [hidden]      f16
//
// Output: [seq, hidden] f16

#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class FusedLayerNorm : public ov::op::Op {
public:
    OPENVINO_OP("FusedLayerNorm", "rocm_gpu");

    FusedLayerNorm() = default;

    // With residual
    FusedLayerNorm(const ov::Output<ov::Node>& x,
                const ov::Output<ov::Node>& skip,
                const ov::Output<ov::Node>& gamma,
                const ov::Output<ov::Node>& beta,
                int64_t seq_len, int64_t hidden)
        : Op({x, skip, gamma, beta})
        , seq_len_(seq_len), hidden_(hidden), has_residual_(true) {
        constructor_validate_and_infer_types();
    }

    // Without residual
    FusedLayerNorm(const ov::Output<ov::Node>& x,
                const ov::Output<ov::Node>& gamma,
                const ov::Output<ov::Node>& beta,
                int64_t seq_len, int64_t hidden)
        : Op({x, gamma, beta})
        , seq_len_(seq_len), hidden_(hidden), has_residual_(false) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto dtype = get_input_element_type(0);
        set_output_type(0, dtype, ov::PartialShape{seq_len_, hidden_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        if (has_residual_ && inputs.size() == 4)
            return std::make_shared<FusedLayerNorm>(inputs[0], inputs[1], inputs[2], inputs[3],
                                                 seq_len_, hidden_);
        return std::make_shared<FusedLayerNorm>(inputs[0], inputs[1], inputs[2],
                                             seq_len_, hidden_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("seq_len",      seq_len_);
        visitor.on_attribute("hidden",       hidden_);
        visitor.on_attribute("has_residual", has_residual_);
        return true;
    }

    int64_t get_seq_len()      const { return seq_len_; }
    int64_t get_hidden()       const { return hidden_; }
    bool    get_has_residual() const { return has_residual_; }

private:
    int64_t seq_len_{256};
    int64_t hidden_{768};
    bool    has_residual_{true};
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
