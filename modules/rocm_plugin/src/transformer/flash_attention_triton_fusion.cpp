#include "flash_attention_triton_fusion.hpp"
#include "transformer/nodes/flash_attention_triton_node.hpp"
#include "transformer/nodes/wmma_attention_node.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <openvino/op/matmul.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/core/graph_util.hpp>

namespace ov { namespace rocm_gpu { namespace pass {

namespace {
static bool g_trace = false;
static std::vector<int64_t> get_shape(const ov::Output<ov::Node>& o) {
    const auto& ps = o.get_partial_shape();
    if (ps.is_dynamic()) return {};
    std::vector<int64_t> d;
    for (auto& x : ps) d.push_back(x.get_length());
    return d;
}

// Insert Transpose [B,H,S,D] → [B,S,H,D]
static ov::Output<ov::Node> insert_bhsd_to_bshd(const ov::Output<ov::Node>& input) {
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(input, perm);
    return transpose->output(0);
}

// Insert Transpose [B,S,H,D] → [B,H,S,D]
static ov::Output<ov::Node> insert_bshd_to_bhsd(
    const ov::Output<ov::Node>& input) {
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(input, perm);
    return transpose->output(0);
}
// Transpose last two dims: [..., A, B] → [..., B, A]
static ov::Output<ov::Node> transpose_last2(const ov::Output<ov::Node>& input) {
    auto shape = get_shape(input);
    int rank = (int)shape.size();
    std::vector<int64_t> perm(rank);
    for (int i = 0; i < rank; i++) perm[i] = i;
    std::swap(perm[rank-1], perm[rank-2]);
    auto perm_const = ov::op::v0::Constant::create(ov::element::i64, {(size_t)rank}, perm);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(input, perm_const);
    return transpose->output(0);
}

} // anon

bool FlashAttentionTritonFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (std::getenv("ROCM_DISABLE_TRITON_FA")) return false;
    g_trace = std::getenv("ROCM_TRACE_TRITON_FA") != nullptr;

    bool changed = false;
    int fired = 0;

    std::vector<std::shared_ptr<ov::Node>> softmaxes;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Softmax>(node) || ov::is_type<ov::op::v8::Softmax>(node))
            softmaxes.push_back(node);
    }

    if (g_trace)
        fprintf(stderr, "[TritonFA] found %zu Softmax nodes\n", softmaxes.size());

    for (auto& sm : softmaxes) {
        auto sm_shape = get_shape(sm->output(0));
        if (sm_shape.empty()) continue;
        int rank = (int)sm_shape.size();

        int axis = -1;
        if (auto v1 = std::dynamic_pointer_cast<ov::op::v1::Softmax>(sm)) axis = (int)v1->get_axis();
        else if (auto v8 = std::dynamic_pointer_cast<ov::op::v8::Softmax>(sm)) axis = (int)v8->get_axis();
        if (axis < 0) axis += rank;
        if (axis != rank - 1) continue;

        // Walk softmax input: allow QK MatMul, or Add(QK, bias), or Mul(QK, scale)
        auto sm_input = sm->get_input_node_shared_ptr(0);
        std::shared_ptr<ov::op::v0::MatMul> qk_mm;

        // Try direct MatMul
        qk_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(sm_input);
        if (!qk_mm) {
            // Try through Add (bias) or Multiply (scale)
            auto try_input = [&](const std::shared_ptr<ov::Node>& node) {
                qk_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node->get_input_node_shared_ptr(0));
                if (!qk_mm)
                    qk_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node->get_input_node_shared_ptr(1));
            };
            if (auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(sm_input)) try_input(add);
            else if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(sm_input)) try_input(mul);
            if (!qk_mm) {
                if (g_trace) fprintf(stderr, "[TritonFA]   skip %s: no QK MatMul\n", sm->get_friendly_name().c_str());
                continue;
            }
        }

        // Handle both transpose_b=true (Q@K^T) and transpose_b=false (K pre-transposed)
        bool k_pretransposed = !qk_mm->get_transpose_b();

        // AV MatMul
        auto sm_consumers = sm->output(0).get_target_inputs();
        if (sm_consumers.size() != 1) continue;
        auto av_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(sm_consumers.begin()->get_node()->shared_from_this());
        if (!av_mm || av_mm->get_input_node_shared_ptr(0).get() != sm.get()) continue;

        auto Q = qk_mm->input_value(0);
        auto K_raw = qk_mm->input_value(1);
        auto V = av_mm->input_value(1);

        if (Q.get_element_type() != ov::element::f16) continue;

        // When K is pre-transposed: shape is [..., D, Sk] instead of [..., Sk, D].
        // Swap last two dims to normalize to [..., Sk, D].
        auto K = k_pretransposed ? transpose_last2(K_raw) : K_raw;

        auto qs = get_shape(Q), ks = get_shape(K), vs = get_shape(V);
        auto scores_shape = get_shape(qk_mm->output(0));

        // Determine Sq, Sk from score matrix shape
        int64_t Sq, Sk;

        if (scores_shape.size() == 4) {
            Sq = scores_shape[2]; Sk = scores_shape[3];
        } else if (scores_shape.size() == 3) {
            Sq = scores_shape[1]; Sk = scores_shape[2];
        } else continue;

        // Dispatch to WMMA or Triton based on size (handled below)

        // === Determine layout and extract B, H, D ===
        int64_t B, H, D;
        bool need_transpose = false;  // true if Q/K/V are [B,H,S,D], need → [B,S,H,D]

        if (qs.size() == 4) {
            // 4D: either [B,H,Sq,D] or [B,Sq,H,D]
            if (qs[2] == Sq) {
                // [B, H, Sq, D] layout
                B = qs[0]; H = qs[1]; D = qs[3];
                need_transpose = true;
            } else if (qs[1] == Sq) {
                // [B, Sq, H, D] layout — already correct for Triton
                B = qs[0]; H = qs[2]; D = qs[3];
                need_transpose = false;
            } else continue;
        } else if (qs.size() == 3) {
            // [BH, Sq, D] — treat as B=BH, H=1
            B = qs[0]; H = 1; D = qs[2];
            need_transpose = false;
        } else continue;

        if (D > 128) continue;

        float scale = 1.0f / std::sqrt((float)D);

        if (g_trace)
            fprintf(stderr, "[TritonFA]   match %s: B=%lld H=%lld Sq=%lld Sk=%lld D=%lld transpose=%d\n",
                    sm->get_friendly_name().c_str(),
                    (long long)B, (long long)H, (long long)Sq, (long long)Sk, (long long)D, need_transpose);

        // === Insert transposes if needed ===
        ov::Output<ov::Node> Q_bshd = Q, K_bshd = K, V_bshd = V;
        if (need_transpose) {
            Q_bshd = insert_bhsd_to_bshd(Q);
            K_bshd = insert_bhsd_to_bshd(K);
            V_bshd = insert_bhsd_to_bshd(V);
        }

        // Dispatch: WMMA for small attention, Triton FA for large
        // WMMA: Sq,Sk <= 256, D <= 64, all % 16 == 0, no hipStreamSync needed
        // Triton: Sq,Sk > 512, any D <= 128, needs hipStreamSync
        bool use_wmma = (Sq <= 512 && Sk <= 512 && D <= 64 &&
                         Sq % 16 == 0 && Sk % 16 == 0 && D % 16 == 0 &&
                         Sk * D * 4 <= 65536 &&  // LDS: 2*(K+V) * SK * D * 2B <= 64KB
                         B * H >= 64);           // need enough blocks for GPU occupancy
        bool use_triton = (Sq > 512 && Sk > 512);

        std::shared_ptr<ov::Node> attn_node;
        if (use_wmma) {
            attn_node = std::make_shared<nodes::WMMAAttention>(
                Q_bshd, K_bshd, V_bshd, (int)B, (int)H, (int)Sq, (int)Sk, (int)D, scale);
            if (g_trace)
                fprintf(stderr, "[TritonFA]     → WMMA kernel (small attention)\n");
        } else if (use_triton) {
            attn_node = std::make_shared<nodes::FlashAttentionTriton>(
                Q_bshd, K_bshd, V_bshd, (int)B, (int)H, (int)Sq, (int)Sk, (int)D, scale);
            if (g_trace)
                fprintf(stderr, "[TritonFA]     → Triton FA kernel (large attention)\n");
        } else {
            if (g_trace)
                fprintf(stderr, "[TritonFA]     → skip (Sq=%lld Sk=%lld D=%lld, no kernel fits)\n",
                        (long long)Sq, (long long)Sk, (long long)D);
            continue;
        }

        if (need_transpose) {
            auto out_transposed = insert_bshd_to_bhsd(attn_node->output(0));
            ov::replace_node(av_mm, out_transposed.get_node_shared_ptr());
        } else {
            ov::replace_node(av_mm, attn_node);
        }

        ++fired; changed = true;
    }

    if (g_trace || fired > 0)
        fprintf(stderr, "[FlashAttentionTritonFusionPass] fused %d attention blocks\n", fired);
    return changed;
}

}}} // namespace
