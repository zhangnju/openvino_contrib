// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedFCGELU: FullyConnected + GELU in a single GPU kernel.
// Inputs: [0]=x[seq,in_dim]  [1]=W[in_dim,out_dim]  [2]=bias[out_dim]
// Output: [0]=GELU(x×W+bias) [seq,out_dim]
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class FusedFCGELU : public ov::op::Op {
public:
    OPENVINO_OP("FusedFCGELU", "rocm_gpu");

    FusedFCGELU() = default;

    FusedFCGELU(const ov::Output<ov::Node>& x,
                const ov::Output<ov::Node>& W,
                const ov::Output<ov::Node>& bias,
                int64_t seq, int64_t in_dim, int64_t out_dim)
        : Op({x, W, bias}), seq_(seq), in_dim_(in_dim), out_dim_(out_dim) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0),
                        ov::PartialShape{seq_, out_dim_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<FusedFCGELU>(inputs[0], inputs[1], inputs[2],
                                             seq_, in_dim_, out_dim_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("seq",     seq_);
        visitor.on_attribute("in_dim",  in_dim_);
        visitor.on_attribute("out_dim", out_dim_);
        return true;
    }

    int64_t get_seq()     const { return seq_; }
    int64_t get_in_dim()  const { return in_dim_; }
    int64_t get_out_dim() const { return out_dim_; }

private:
    int64_t seq_{0};
    int64_t in_dim_{0};
    int64_t out_dim_{0};
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
