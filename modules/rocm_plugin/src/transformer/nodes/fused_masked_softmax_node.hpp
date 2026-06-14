// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// FusedMaskedSoftmax: attention scores + mask → Softmax in one kernel.
// Inputs: [0]=scores[1,heads,sq,sk]  [1]=mask[1,1,sq,sk]
// Output: [0]=softmax(scores+mask)[1,heads,sq,sk]
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class FusedMaskedSoftmax : public ov::op::Op {
public:
    OPENVINO_OP("FusedMaskedSoftmax", "rocm_gpu");

    FusedMaskedSoftmax() = default;

    FusedMaskedSoftmax(const ov::Output<ov::Node>& scores,
                       const ov::Output<ov::Node>& mask,
                       int64_t heads, int64_t sq, int64_t sk)
        : Op({scores, mask}), heads_(heads), sq_(sq), sk_(sk) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0),
                        ov::PartialShape{1, heads_, sq_, sk_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<FusedMaskedSoftmax>(inputs[0], inputs[1],
                                                     heads_, sq_, sk_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("heads", heads_);
        visitor.on_attribute("sq",    sq_);
        visitor.on_attribute("sk",    sk_);
        return true;
    }

    int64_t get_heads() const { return heads_; }
    int64_t get_sq()    const { return sq_; }
    int64_t get_sk()    const { return sk_; }

private:
    int64_t heads_{0}, sq_{0}, sk_{0};
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
