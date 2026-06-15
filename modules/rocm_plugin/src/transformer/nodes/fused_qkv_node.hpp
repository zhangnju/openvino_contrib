// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedQKVProjection: merges Q, K, V FullyConnected projections into a
// single 256×2304×768 hipBLASLt GEMM+bias kernel.
//
// Inputs: [0]=x[seq,H]  [1]=W_qkv[3H,H]  [2]=b_qkv[3H]
// Output: [0]=QKV_combined[seq,3H]  — contiguous, Q at [0], K at [H], V at [2H]
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class FusedQKVProjection : public ov::op::Op {
public:
    OPENVINO_OP("FusedQKVProjection", "rocm_gpu");

    FusedQKVProjection() = default;

    FusedQKVProjection(const ov::Output<ov::Node>& x,
                       const ov::Output<ov::Node>& W_qkv,
                       const ov::Output<ov::Node>& b_qkv,
                       int64_t seq_len, int64_t hidden)
        : Op({x, W_qkv, b_qkv}), seq_len_(seq_len), hidden_(hidden) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // Output: [seq, 3*hidden] — Q, K, V concatenated
        set_output_type(0, get_input_element_type(0),
                        ov::PartialShape{seq_len_, 3 * hidden_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& v) const override {
        return std::make_shared<FusedQKVProjection>(v[0], v[1], v[2], seq_len_, hidden_);
    }

    bool visit_attributes(ov::AttributeVisitor& vis) override {
        vis.on_attribute("seq_len", seq_len_);
        vis.on_attribute("hidden",  hidden_);
        return true;
    }

    int64_t get_seq_len() const { return seq_len_; }
    int64_t get_hidden()  const { return hidden_; }

private:
    int64_t seq_len_{256};
    int64_t hidden_{768};
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
