// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// BERT self-attention pattern detection and fusion.
//
// Pattern (after FullyConnectedTransformation and LayerNormFusion):
//   Q = FullyConnected(x, W_q, b_q)  → [seq, hidden=12*64]
//   K = FullyConnected(x, W_k, b_k)  → [seq, hidden]
//   V = FullyConnected(x, W_v, b_v)  → [seq, hidden]
//   scores = MatMul(Reshape+Transpose(Q), Reshape+Transpose+Transpose(K))
//   scores = scores * scale_factor
//   scores = scores + attn_mask
//   weights = Softmax(scores)
//   context = MatMul(weights, Reshape+Transpose(V))
//   context_flat = Reshape(Transpose(context))   → [1, seq, hidden]
//
// This pass looks for the Softmax node (unique to each attention layer)
// and traces backwards to find Q, K, V and forward to get context output.

#include "bert_self_attention_fusion.hpp"
#include "nodes/bert_self_attention_node.hpp"

#include <optional>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>
#include <vector>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

namespace {

// Skip through passthrough ops (Identity, Convert, Reshape with single consumer)
static std::shared_ptr<ov::Node> skip_trivial(std::shared_ptr<ov::Node> n, int depth = 0) {
    if (depth > 4) return n;
    if (!n) return n;
    const std::string& t = n->get_type_info().name;
    if ((t == "Identity" || t == "Convert") && n->inputs().size() == 1)
        return skip_trivial(n->get_input_node_shared_ptr(0), depth + 1);
    return n;
}

static std::shared_ptr<ov::Node> single_consumer(const std::shared_ptr<ov::Node>& n) {
    const auto ts = n->output(0).get_target_inputs();
    if (ts.size() != 1) return nullptr;
    return ts.begin()->get_node()->shared_from_this();
}


// Trace from a FullyConnected/MatMul output back to the flat [seq, hidden] buffer.
// Returns the FC/MatMul node if found within 2 hops.
static std::shared_ptr<ov::Node> find_qkv_source(const std::shared_ptr<ov::Node>& transpose_node) {
    // Transpose ← Reshape ← FC/MatMul
    auto pre_t = transpose_node->get_input_node_shared_ptr(0);
    auto pre_pre = pre_t->get_input_node_shared_ptr(0);
    if (pre_pre->get_type_info().name == std::string("FullyConnected"))
        return pre_pre;
    if (pre_pre->get_type_info().name == std::string("MatMul"))
        return pre_pre;
    return nullptr;
}

// Detect the Q×K^T MatMul that feeds into Softmax (via scale + mask add).
// Returns: {qkt_matmul, scale_multiply, mask_add, Q_fc, K_fc}
struct AttentionInfo {
    std::shared_ptr<ov::Node> qkt_matmul;
    std::shared_ptr<ov::Node> q_fc;   // Q FullyConnected output
    std::shared_ptr<ov::Node> k_fc;   // K FullyConnected output
    std::shared_ptr<ov::Node> v_fc;   // V FullyConnected output
    std::shared_ptr<ov::Node> attn_bias;  // attention mask (already scaled)
    std::shared_ptr<ov::Node> context_out;  // final context [1, seq, hidden]
    int64_t seq_len   = 0;
    int64_t num_heads = 0;
    int64_t head_dim  = 0;
};

// Try to match BERT attention from a Softmax node (v1 or v8).
static std::optional<AttentionInfo> match_attention(
        const std::shared_ptr<ov::Node>& softmax_node) {
    AttentionInfo info;
    const std::shared_ptr<ov::Node>& softmax = softmax_node;

    // Softmax input: mask_add (scores + attention_mask)
    auto sm_input = softmax_node->get_input_node_shared_ptr(0);
    fprintf(stderr, "[BertAttnFuse] Softmax input: %s = %s\n",
            sm_input->get_type_name(), sm_input->get_friendly_name().c_str());
    auto mask_add = std::dynamic_pointer_cast<ov::op::v1::Add>(sm_input);
    if (!mask_add) {
        fprintf(stderr, "[BertAttnFuse] Not Add, is %s\n", sm_input->get_type_name());
        return std::nullopt;
    }

    // mask_add inputs: one is the Q×K^T result (possibly scaled), other is attention mask
    // Pattern 1: MatMul(Q, K^T) → Add(mask)  [scale already in Q]
    // Pattern 2: Multiply(MatMul, scale) → Add(mask)  [separate scale]
    auto in0 = mask_add->get_input_node_shared_ptr(0);
    auto in1 = mask_add->get_input_node_shared_ptr(1);

    std::shared_ptr<ov::op::v0::MatMul> qkt_matmul;
    std::shared_ptr<ov::Node> attn_bias_node;

    // Try to find MatMul directly or via Multiply
    auto try_find_matmul = [&](const std::shared_ptr<ov::Node>& n) -> bool {
        if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(n)) {
            qkt_matmul = mm; return true;
        }
        // Via Multiply (scale factor)
        if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(n)) {
            for (int i = 0; i < 2; ++i) {
                if (auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(mul->get_input_node_shared_ptr(i))) {
                    qkt_matmul = mm; return true;
                }
            }
        }
        return false;
    };

    if (try_find_matmul(in0)) {
        attn_bias_node = in1;
    } else if (try_find_matmul(in1)) {
        attn_bias_node = in0;
    }

    if (!qkt_matmul) {
        fprintf(stderr, "[BertAttnFuse]   Can't find Q×K^T MatMul: in0=%s in1=%s\n",
                in0->get_type_name(), in1->get_type_name());
        return std::nullopt;
    }

    info.attn_bias = attn_bias_node;

    // QKT matmul inputs: Q (possibly Multiply(Transpose(Q), scale) or just Transpose(Q))
    // and K^T (double Transpose)
    auto mm_in0_raw = qkt_matmul->get_input_node_shared_ptr(0);  // Q path
    auto mm_in1 = qkt_matmul->get_input_node_shared_ptr(1);  // K^T path
    fprintf(stderr, "[BertAttnFuse]   QKT matmul: in0=%s in1=%s\n",
            mm_in0_raw->get_type_name(), mm_in1->get_type_name());

    // Skip through Multiply (scale factor) if Q is Multiply(Transpose(Q), scale)
    std::shared_ptr<ov::Node> mm_in0 = mm_in0_raw;
    if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(mm_in0)) {
        for (int i = 0; i < 2; ++i) {
            auto inp = mul->get_input_node_shared_ptr(i);
            if (inp->get_type_info().name == std::string("Transpose")) {
                mm_in0 = inp; break;
            }
        }
    }

    // Q: Reshape→Transpose([0,2,1,3])
    if (mm_in0->get_type_info().name != std::string("Transpose")) {
        fprintf(stderr, "[BertAttnFuse]   Q not Transpose (is %s)\n", mm_in0->get_type_name());
        return std::nullopt;
    }
    // Skip through Multiply (scale applied to K^T) to find the Transpose
    std::shared_ptr<ov::Node> mm_in1_eff = mm_in1;
    if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(mm_in1)) {
        for (int i = 0; i < 2; ++i) {
            auto inp = mul->get_input_node_shared_ptr(i);
            if (inp->get_type_info().name == std::string("Transpose")) {
                mm_in1_eff = inp; break;
            }
        }
    }

    if (mm_in1_eff->get_type_info().name != std::string("Transpose")) {
        fprintf(stderr, "[BertAttnFuse]   K not Transpose (is %s)\n", mm_in1_eff->get_type_name());
        return std::nullopt;
    }
    mm_in1 = mm_in1_eff;

    auto q_fc = find_qkv_source(mm_in0);
    if (!q_fc) { fprintf(stderr, "[BertAttnFuse]   Q FC not found\n"); return std::nullopt; }

    // For K: may be Transpose(Transpose(Reshape(K_fc)))
    auto k_inner = mm_in1->get_input_node_shared_ptr(0);
    if (k_inner->get_type_info().name == std::string("Transpose")) {
        // Double transpose path
        auto k_fc = find_qkv_source(k_inner);
        if (!k_fc) return std::nullopt;
        info.k_fc = k_fc;
    } else {
        auto k_fc = find_qkv_source(mm_in1);
        if (!k_fc) return std::nullopt;
        info.k_fc = k_fc;
    }

    info.q_fc = q_fc;
    // info.attn_bias was already set above
    info.qkt_matmul = qkt_matmul;

    // Now find A×V MatMul: Softmax → MatMul(softmax, V_transposed)
    if (softmax_node->output(0).get_target_inputs().size() != 1) return std::nullopt;
    auto av_matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
        single_consumer(softmax_node));
    if (!av_matmul) return std::nullopt;

    // V: Reshape→Transpose([0,2,1,3]) → MatMul input 1
    auto av_in1 = av_matmul->get_input_node_shared_ptr(1);
    if (av_in1->get_type_info().name != std::string("Transpose")) return std::nullopt;
    auto v_fc = find_qkv_source(av_in1);
    if (!v_fc) return std::nullopt;
    info.v_fc = v_fc;

    // Context: A×V MatMul output → Transpose([0,2,1,3]) → Reshape → output
    if (av_matmul->output(0).get_target_inputs().size() != 1) return std::nullopt;
    auto ctx_transpose = single_consumer(av_matmul);
    if (!ctx_transpose || ctx_transpose->get_type_info().name != std::string("Transpose"))
        return std::nullopt;
    if (ctx_transpose->output(0).get_target_inputs().size() != 1) return std::nullopt;
    auto ctx_reshape = single_consumer(ctx_transpose);
    if (!ctx_reshape || ctx_reshape->get_type_info().name != std::string("Reshape"))
        return std::nullopt;

    info.context_out = ctx_reshape;

    // Extract dimensions from Q FC output shape
    auto q_shape = q_fc->get_output_partial_shape(0);
    if (!q_shape.rank().is_static() || q_shape.rank().get_length() < 2) return std::nullopt;
    int64_t rank = q_shape.rank().get_length();
    if (!q_shape[rank-1].is_static()) return std::nullopt;
    int64_t hidden = q_shape[rank-1].get_length();

    // Get num_heads from the Reshape output shape
    auto reshape_shape = mm_in0->get_input_node_shared_ptr(0)->get_output_partial_shape(0);
    if (!reshape_shape.rank().is_static() || reshape_shape.rank().get_length() != 4) return std::nullopt;
    if (!reshape_shape[2].is_static() || !reshape_shape[3].is_static()) return std::nullopt;
    info.num_heads = reshape_shape[2].get_length();
    info.head_dim  = reshape_shape[3].get_length();

    // Verify: hidden = num_heads * head_dim
    if (hidden != info.num_heads * info.head_dim) return std::nullopt;

    // Get seq_len from Q FC output (first dim)
    if (!q_shape[rank-2].is_static()) return std::nullopt;
    info.seq_len = q_shape[rank-2].get_length();

    return info;
}

}  // namespace

bool BertSelfAttentionFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    // Collect Softmax nodes (v1 and v8 — ONNX export uses v8)
    std::vector<std::shared_ptr<ov::Node>> softmaxes;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v1::Softmax>(node) ||
            std::dynamic_pointer_cast<ov::op::v8::Softmax>(node))
            softmaxes.push_back(node);
    }

    fprintf(stderr, "[BertAttnFuse] Running on %zu Softmax nodes\n", softmaxes.size());
    if (!softmaxes.empty() && std::getenv("ATTN_DEBUG")) {
        // Backward BFS dump from the first Softmax to reveal the INT8 attention structure.
        auto sm0 = softmaxes.front();
        fprintf(stderr, "[ATTN_DEBUG] backward from Softmax '%s'\n", sm0->get_friendly_name().c_str());
        std::vector<std::pair<std::shared_ptr<ov::Node>,int>> st{{sm0,0}};
        int printed = 0;
        while (!st.empty() && printed < 40) {
            auto [n,d] = st.back(); st.pop_back();
            fprintf(stderr, "  d%-2d %-16s '%s'\n", d, n->get_type_name(), n->get_friendly_name().c_str());
            ++printed;
            if (d >= 12) continue;
            for (size_t i=0;i<n->get_input_size();++i)
                st.push_back({n->input_value(i).get_node_shared_ptr(), d+1});
        }
    }
    for (auto& sm : softmaxes) {
        try {
            auto info = match_attention(sm);
            if (!info) {
                fprintf(stderr, "[BertAttnFuse] No match for Softmax: %s\n",
                        sm->get_friendly_name().c_str());
                continue;
            }

            // Flatten bias from [1, 1, seq, seq] to [seq*seq] so the kernel receives
            // a flat buffer of exactly seq*seq f16 elements that it broadcasts over heads.
            int64_t bias_flat = info->seq_len * info->seq_len;
            auto bias_shape_const = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{1}, std::vector<int64_t>{bias_flat});
            auto bias_flat_node = std::make_shared<ov::op::v1::Reshape>(
                info->attn_bias->output(0), bias_shape_const->output(0), false);
            bias_flat_node->set_friendly_name(sm->get_friendly_name() + "/bias_flat");

            // Build BertSelfAttention node
            auto attn_node = std::make_shared<nodes::BertSelfAttention>(
                info->q_fc->output(0),
                info->k_fc->output(0),
                info->v_fc->output(0),
                bias_flat_node->output(0),
                info->seq_len, info->num_heads, info->head_dim);
            attn_node->set_friendly_name(sm->get_friendly_name() + "/BertSelfAttn");

            // Replace the context output with the fused attention output
            ov::replace_node(info->context_out, attn_node);

            fprintf(stderr, "[BertAttnFuse] Fused layer: seq=%lld heads=%lld dim=%lld\n",
                    (long long)info->seq_len, (long long)info->num_heads,
                    (long long)info->head_dim);
            changed = true;
        } catch (const std::exception& e) {
            fprintf(stderr, "[BertAttnFuse] Skipped (exception): %s\n", e.what());
        } catch (...) {}
    }
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
