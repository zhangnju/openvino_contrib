// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fused_qkv_pass.hpp"
#include "nodes/fused_qkv_node.hpp"
#include "nodes/bert_self_attention_node.hpp"

#include <openvino/op/constant.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

namespace {

// Get the underlying FullyConnected node from a BertSelfAttention input.
// BertSelfAttentionFusion stored FC outputs as node inputs.
static std::shared_ptr<ov::Node> get_fc(const std::shared_ptr<ov::Node>& attn,
                                         size_t input_idx) {
    auto n = attn->get_input_node_shared_ptr(input_idx);
    // skip trivial wrappers
    for (int d = 0; d < 3 && n; ++d) {
        const std::string& t = n->get_type_info().name;
        if (t == "FullyConnected") return n;
        if (t == "Convert" || t == "Identity") n = n->get_input_node_shared_ptr(0);
        else break;
    }
    return nullptr;
}

// Concatenate three Constant tensors along axis 0, return new Constant.
// Handles both f16 and f32 element types via cast_vector<float>.
static std::shared_ptr<ov::op::v0::Constant> concat_const(
        const std::shared_ptr<ov::op::v0::Constant>& a,
        const std::shared_ptr<ov::op::v0::Constant>& b,
        const std::shared_ptr<ov::op::v0::Constant>& c) {
    auto et = a->get_element_type();
    auto sa = a->get_shape(), sb = b->get_shape(), sc = c->get_shape();
    if (et != b->get_element_type() || et != c->get_element_type()) return nullptr;
    if (sa.size() != 1 && sa.size() != 2) return nullptr;

    // Use cast_vector for f16 compatibility (get_vector rejects sizeof(float) > sizeof(f16))
    auto va = a->cast_vector<float>();
    auto vb = b->cast_vector<float>();
    auto vc = c->cast_vector<float>();
    std::vector<float> combined;
    combined.reserve(va.size() + vb.size() + vc.size());
    combined.insert(combined.end(), va.begin(), va.end());
    combined.insert(combined.end(), vb.begin(), vb.end());
    combined.insert(combined.end(), vc.begin(), vc.end());

    ov::Shape out_shape;
    if (sa.size() == 1) {
        out_shape = {sa[0] + sb[0] + sc[0]};
    } else {
        // 2D: concatenate along axis 0 (rows)
        if (sa[1] != sb[1] || sa[1] != sc[1]) return nullptr;
        out_shape = {sa[0] + sb[0] + sc[0], sa[1]};
    }
    return ov::op::v0::Constant::create(et, out_shape, combined);
}

// Squeeze bias from [1, H] to [H] for hipBLASLt EPILOGUE_BIAS compatibility.
static void squeeze_bias_if_needed(std::shared_ptr<ov::op::v0::Constant>& c) {
    auto s = c->get_shape();
    if (s.size() == 2 && s[0] == 1) {
        auto data = c->cast_vector<float>();
        c = ov::op::v0::Constant::create(c->get_element_type(), ov::Shape{s[1]}, data);
    }
}

}  // namespace

bool FusedQKVProjectionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    // Collect BertSelfAttention nodes
    std::vector<std::shared_ptr<nodes::BertSelfAttention>> attns;
    for (const auto& node : model->get_ordered_ops())
        if (auto a = std::dynamic_pointer_cast<nodes::BertSelfAttention>(node))
            if (!a->is_combined_qkv())  // skip already-fused
                attns.push_back(a);

    for (auto& attn : attns) {
        try {
            // BertSelfAttention inputs: [Q_fc_out, K_fc_out, V_fc_out, attn_bias]
            if (attn->get_input_size() < 4) continue;

            auto q_fc = get_fc(attn, 0);
            auto k_fc = get_fc(attn, 1);
            auto v_fc = get_fc(attn, 2);
            if (!q_fc || !k_fc || !v_fc) continue;

            // All 3 FCs must share the same primary input x
            auto x_q = q_fc->get_input_node_ptr(0);
            auto x_k = k_fc->get_input_node_ptr(0);
            auto x_v = v_fc->get_input_node_ptr(0);
            if (x_q != x_k || x_q != x_v) continue;

            // Extract weights and biases (input[1] = W, input[2] = bias)
            auto Wq = std::dynamic_pointer_cast<ov::op::v0::Constant>(q_fc->get_input_node_shared_ptr(1));
            auto Wk = std::dynamic_pointer_cast<ov::op::v0::Constant>(k_fc->get_input_node_shared_ptr(1));
            auto Wv = std::dynamic_pointer_cast<ov::op::v0::Constant>(v_fc->get_input_node_shared_ptr(1));
            auto bq = std::dynamic_pointer_cast<ov::op::v0::Constant>(q_fc->get_input_node_shared_ptr(2));
            auto bk = std::dynamic_pointer_cast<ov::op::v0::Constant>(k_fc->get_input_node_shared_ptr(2));
            auto bv = std::dynamic_pointer_cast<ov::op::v0::Constant>(v_fc->get_input_node_shared_ptr(2));
            if (!Wq || !Wk || !Wv || !bq || !bk || !bv) continue;

            // Verify shapes: W[hidden, hidden], b[hidden]
            auto ws = Wq->get_shape();
            if (ws.size() != 2 || ws != Wk->get_shape() || ws != Wv->get_shape()) continue;
            int64_t hidden = static_cast<int64_t>(ws[0]);  // out_dim for FC weight

            // Squeeze bias from [1, H] to [H] if needed (TF-origin models use [1, H])
            squeeze_bias_if_needed(bq);
            squeeze_bias_if_needed(bk);
            squeeze_bias_if_needed(bv);

            auto bs = bq->get_shape();
            if (bs.size() != 1 || bs[0] != (size_t)hidden) continue;

            // Get seq_len from Q_fc output shape
            auto q_shape = q_fc->get_output_partial_shape(0);
            if (!q_shape.rank().is_static()) continue;
            auto rank = q_shape.rank().get_length();
            if (!q_shape[rank - 2].is_static()) continue;
            int64_t seq_len = q_shape[rank - 2].get_length();

            // Concatenate weights: [3H, H] and biases: [3H]
            auto W_qkv = concat_const(Wq, Wk, Wv);
            auto b_qkv = concat_const(bq, bk, bv);
            if (!W_qkv || !b_qkv) continue;

            W_qkv->set_friendly_name(Wq->get_friendly_name() + "_W_qkv");
            b_qkv->set_friendly_name(bq->get_friendly_name() + "_b_qkv");

            // Create FusedQKVProjection node
            auto x_val = q_fc->input_value(0);
            auto qkv_node = std::make_shared<nodes::FusedQKVProjection>(
                x_val, W_qkv->output(0), b_qkv->output(0), seq_len, hidden);
            qkv_node->set_friendly_name(attn->get_friendly_name() + "/QKV");

            // Replace BertSelfAttention with combined-QKV variant
            // Keep attn_bias (input[3]) unchanged
            auto attn_bias = attn->input_value(3);
            auto new_attn = std::make_shared<nodes::BertSelfAttention>(
                qkv_node->output(0),  // QKV_combined[seq, 3H]
                attn_bias,
                attn->get_seq_len(), attn->get_num_heads(), attn->get_head_dim());
            new_attn->set_friendly_name(attn->get_friendly_name());
            ov::copy_runtime_info(attn, new_attn);
            ov::replace_node(attn, new_attn);

            fprintf(stderr, "[FusedQKVPass] Fused Q+K+V: seq=%lld hidden=%lld -> QKV[%lld,%lld]\n",
                    (long long)seq_len, (long long)hidden,
                    (long long)seq_len, (long long)(3 * hidden));
            changed = true;

        } catch (...) {}
    }
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
