// SwinFlashAttention fusion: matches MatMul(Q,K^T) → Softmax → MatMul(attn,V)
// and replaces with a single WMMA fused attention kernel.
//
// Runs AFTER QKVTransposeSplit and SwinFusedSoftmax passes.
// Matches both SwinFusedSoftmax (with bias) and plain Softmax nodes.
//
// Constraints: sq,sk <= 64, hd <= 64, fp16 only.

#include "swin_flash_attention_fusion.hpp"
#include "nodes/swin_flash_attention_node.hpp"
#include "nodes/swin_fused_softmax_node.hpp"

#include <openvino/op/matmul.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/core/graph_util.hpp>
#include <iostream>

namespace ov {
namespace rocm_gpu {
namespace pass {

bool SwinFlashAttentionFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int count = 0;

    for (const auto& node : model->get_ordered_ops()) {
        // Look for the PV MatMul: its input[0] is Softmax/SwinFusedSoftmax output
        auto av_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        if (!av_mm) continue;

        // PV MatMul: attn_weights × V
        // attn_weights comes from Softmax/SwinFusedSoftmax
        auto softmax_node = av_mm->get_input_node_shared_ptr(0);
        bool is_swin_softmax = (softmax_node->get_type_name() == std::string("SwinFusedSoftmax"));
        bool is_plain_softmax = (softmax_node->get_type_name() == std::string("Softmax"));
        if (!is_swin_softmax && !is_plain_softmax) continue;

        // Softmax must have exactly one consumer (the PV MatMul)
        if (softmax_node->output(0).get_target_inputs().size() != 1) continue;

        // Get V from PV MatMul input[1]
        auto v_out = av_mm->input_value(1);

        // Trace back through Softmax to find QK MatMul
        std::shared_ptr<ov::op::v0::MatMul> qk_mm;
        ov::Output<ov::Node> bias_out;
        bool has_bias = false;

        if (is_swin_softmax) {
            auto sfs = std::dynamic_pointer_cast<nodes::SwinFusedSoftmax>(softmax_node);
            if (!sfs) continue;
            has_bias = sfs->get_has_bias();
            // SwinFusedSoftmax input[0] is scores (from QK MatMul)
            qk_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
                softmax_node->get_input_node_shared_ptr(0));
            if (has_bias) bias_out = softmax_node->input_value(1);
        } else {
            // Plain Softmax input is directly QK MatMul or Add(QK, bias)
            qk_mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(
                softmax_node->get_input_node_shared_ptr(0));
        }

        if (!qk_mm) continue;
        if (qk_mm->output(0).get_target_inputs().size() != 1) continue;

        // Get Q and K from QK MatMul
        auto q_out = qk_mm->input_value(0);
        auto k_out = qk_mm->input_value(1);

        // Q shape should be [nW, H, sq, hd] or similar 4D
        auto q_ps = q_out.get_partial_shape();
        auto k_ps = k_out.get_partial_shape();
        auto v_ps = v_out.get_partial_shape();
        if (!q_ps.is_static() || !k_ps.is_static() || !v_ps.is_static()) continue;
        auto q_shape = q_ps.to_shape();
        auto k_shape = k_ps.to_shape();
        if (q_shape.size() != 4 || k_shape.size() != 4) continue;

        // fp16 only
        if (q_out.get_element_type() != ov::element::f16) continue;

        int nW = q_shape[0];
        int H  = q_shape[1];
        int sq = q_shape[2];
        int hd = q_shape[3];
        int sk = k_shape[2];

        // Constraints: small attention only (WMMA kernel limitation)
        if (sq > 64 || sk > 64 || hd > 64) continue;
        if (hd % 16 != 0) continue;

        // Determine scale: QK MatMul may have a Multiply(Q, scale) as input
        // For now, use scale=1.0 (attention scores are pre-scaled by the model)
        float scale = 1.0f;

        // Build SwinFlashAttention node
        ov::OutputVector args = {q_out, k_out, v_out};
        if (has_bias) args.push_back(bias_out);

        auto sfa = std::make_shared<nodes::SwinFlashAttention>(
            args, nW, H, sq, sk, hd, has_bias, scale);

        ov::replace_output_update_name(av_mm->output(0), sfa->output(0));
        count++;
        changed = true;
    }

    if (count > 0)
        std::cerr << "[SwinFlashAttn] Fused " << count << " attention patterns" << std::endl;
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
