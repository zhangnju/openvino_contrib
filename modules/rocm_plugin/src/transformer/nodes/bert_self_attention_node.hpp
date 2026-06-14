// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Custom OV node for BERT-style self-attention.
// Fuses: Q×K^T/scale → mask+softmax → ×V into a single rocMLIR kernel.
//
// Inputs:
//   [0] Q_flat   [seq, num_heads*head_dim]  — Q FC projection output (interleaved)
//   [1] K_flat   [seq, num_heads*head_dim]  — K FC projection output (interleaved)
//   [2] V_flat   [seq, num_heads*head_dim]  — V FC projection output (interleaved)
//   [3] attn_bias [1, num_heads, seq, seq]  — attention mask (already scaled, broadcasted)
//
// Output:
//   [0] context  [1, seq, num_heads*head_dim]  — attended context (contiguous)
//
// Attribute storage:
//   seq_len, num_heads, head_dim

#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class BertSelfAttention : public ov::op::Op {
public:
    OPENVINO_OP("BertSelfAttention", "rocm_gpu");

    BertSelfAttention() = default;

    BertSelfAttention(const ov::Output<ov::Node>& q_flat,
                      const ov::Output<ov::Node>& k_flat,
                      const ov::Output<ov::Node>& v_flat,
                      const ov::Output<ov::Node>& attn_bias,
                      int64_t seq_len,
                      int64_t num_heads,
                      int64_t head_dim)
        : Op({q_flat, k_flat, v_flat, attn_bias})
        , seq_len_(seq_len), num_heads_(num_heads), head_dim_(head_dim) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto dtype = get_input_element_type(0);
        // Output: [1, seq_len, num_heads * head_dim]
        set_output_type(0, dtype, ov::PartialShape{1, seq_len_, num_heads_ * head_dim_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<BertSelfAttention>(
            inputs[0], inputs[1], inputs[2], inputs[3],
            seq_len_, num_heads_, head_dim_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("seq_len",   seq_len_);
        visitor.on_attribute("num_heads", num_heads_);
        visitor.on_attribute("head_dim",  head_dim_);
        return true;
    }

    int64_t get_seq_len()   const { return seq_len_; }
    int64_t get_num_heads() const { return num_heads_; }
    int64_t get_head_dim()  const { return head_dim_; }

private:
    int64_t seq_len_   = 256;
    int64_t num_heads_ = 12;
    int64_t head_dim_  = 64;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
