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

    // Standard mode: separate Q, K, V inputs
    BertSelfAttention(const ov::Output<ov::Node>& q_flat,
                      const ov::Output<ov::Node>& k_flat,
                      const ov::Output<ov::Node>& v_flat,
                      const ov::Output<ov::Node>& attn_bias,
                      int64_t seq_len, int64_t num_heads, int64_t head_dim)
        : Op({q_flat, k_flat, v_flat, attn_bias})
        , seq_len_(seq_len), num_heads_(num_heads), head_dim_(head_dim)
        , combined_qkv_(false) {
        constructor_validate_and_infer_types();
    }

    // Combined QKV mode: single QKV_combined[seq, 3*heads*dim] input (no D2D copy)
    BertSelfAttention(const ov::Output<ov::Node>& qkv_combined,
                      const ov::Output<ov::Node>& attn_bias,
                      int64_t seq_len, int64_t num_heads, int64_t head_dim)
        : Op({qkv_combined, attn_bias})
        , seq_len_(seq_len), num_heads_(num_heads), head_dim_(head_dim)
        , combined_qkv_(true) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto dtype = get_input_element_type(0);
        set_output_type(0, dtype, ov::PartialShape{1, seq_len_, num_heads_ * head_dim_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& v) const override {
        if (combined_qkv_)
            return std::make_shared<BertSelfAttention>(v[0], v[1], seq_len_, num_heads_, head_dim_);
        return std::make_shared<BertSelfAttention>(v[0], v[1], v[2], v[3], seq_len_, num_heads_, head_dim_);
    }

    bool visit_attributes(ov::AttributeVisitor& vis) override {
        vis.on_attribute("seq_len",      seq_len_);
        vis.on_attribute("num_heads",    num_heads_);
        vis.on_attribute("head_dim",     head_dim_);
        vis.on_attribute("combined_qkv", combined_qkv_);
        return true;
    }

    int64_t get_seq_len()      const { return seq_len_; }
    int64_t get_num_heads()    const { return num_heads_; }
    int64_t get_head_dim()     const { return head_dim_; }
    bool    is_combined_qkv()  const { return combined_qkv_; }

private:
    int64_t seq_len_   = 256;
    int64_t num_heads_ = 12;
    int64_t head_dim_  = 64;
    bool    combined_qkv_ = false;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
