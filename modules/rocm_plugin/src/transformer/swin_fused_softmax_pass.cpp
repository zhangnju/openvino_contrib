// SwinFusedSoftmax pass: fuses Add(scores, bias) + Softmax patterns.
// Pattern 1: Add([B,H,sq,sk], [H,sq,sk]) → Softmax  (swin with relative position bias)
// Pattern 2: Softmax([B,H,sq,sk]) without Add          (bevformer / plain attention)
//
// Both are replaced with SwinFusedSoftmax node using a native HIP kernel
// that is faster than MIOpen's SoftmaxForward for these shapes.

#include "swin_fused_softmax_pass.hpp"
#include "nodes/swin_fused_softmax_node.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/core/graph_util.hpp>
#include <iostream>

namespace ov {
namespace rocm_gpu {
namespace pass {

static int64_t get_softmax_axis(const std::shared_ptr<ov::Node>& sm) {
    if (auto sm1 = std::dynamic_pointer_cast<ov::op::v1::Softmax>(sm))
        return sm1->get_axis();
    if (auto sm8 = std::dynamic_pointer_cast<ov::op::v8::Softmax>(sm))
        return sm8->get_axis();
    return -1;
}

bool SwinFusedSoftmaxPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int fused_bias = 0, fused_plain = 0;

    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_type_name() != std::string("Softmax")) continue;

        // Must be last-axis softmax
        int64_t axis = get_softmax_axis(node);
        auto in_ps = node->get_input_partial_shape(0);
        if (!in_ps.is_static()) continue;
        auto in_shape = in_ps.to_shape();
        if (in_shape.size() < 3) continue;

        int ndim = in_shape.size();
        if (axis < 0) axis += ndim;
        if (axis != ndim - 1) continue;

        // Only fp16
        if (node->get_input_element_type(0) != ov::element::f16) continue;

        int sk = in_shape[ndim - 1];
        int sq = in_shape[ndim - 2];

        // Try Pattern 1: Add → Softmax (with bias)
        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(
            node->get_input_node_shared_ptr(0));

        if (add && add->output(0).get_target_inputs().size() == 1) {
            // Check Add inputs: scores [B,H,sq,sk] + bias [H,sq,sk]
            auto ps0 = add->get_input_partial_shape(0);
            auto ps1 = add->get_input_partial_shape(1);
            if (ps0.is_static() && ps1.is_static()) {
                auto s0 = ps0.to_shape(), s1 = ps1.to_shape();
                // Identify which is scores (4D) and which is bias (3D or broadcast)
                ov::Output<ov::Node> scores_out, bias_out;
                int B = 0, H = 0;

                auto try_match = [&](const ov::Shape& ss, const ov::Shape& sb,
                                     ov::Output<ov::Node> so, ov::Output<ov::Node> bo) -> bool {
                    if (ss.size() == 4 && ss[2] == (size_t)sq && ss[3] == (size_t)sk) {
                        // bias can be [H,sq,sk] or [1,H,sq,sk] or broadcastable
                        if ((sb.size() == 3 && sb[0] == ss[1] && sb[1] == (size_t)sq && sb[2] == (size_t)sk) ||
                            (sb.size() == 4 && sb[0] == 1 && sb[1] == ss[1] && sb[2] == (size_t)sq && sb[3] == (size_t)sk)) {
                            B = ss[0]; H = ss[1];
                            scores_out = so; bias_out = bo;
                            return true;
                        }
                    }
                    return false;
                };

                if (try_match(s0, s1, add->input_value(0), add->input_value(1)) ||
                    try_match(s1, s0, add->input_value(1), add->input_value(0))) {
                    auto sfs = std::make_shared<nodes::SwinFusedSoftmax>(
                        scores_out, bias_out, B, H, sq, sk);
                    ov::replace_output_update_name(node->output(0), sfs->output(0));
                    fused_bias++;
                    changed = true;
                    continue;
                }
            }
        }

        // Pattern 2: Reshape → Softmax (shifted window attention).
        // Reshape is purely logical (no data move), softmax along last dim is the same.
        // Fuse as plain softmax (no bias) — Reshape stays, we replace Softmax only.
        auto reshape_input = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
            node->get_input_node_shared_ptr(0));
        if (reshape_input && in_shape.size() == 4 && node->get_input_element_type(0) == ov::element::f16) {
            int B = in_shape[0], H = in_shape[1];
            auto sfs = std::make_shared<nodes::SwinFusedSoftmax>(
                node->input_value(0), B, H, sq, sk);
            ov::replace_output_update_name(node->output(0), sfs->output(0));
            fused_plain++;
            changed = true;
            continue;
        }
    }

    if (fused_bias + fused_plain > 0)
        std::cerr << "[SwinFusedSoftmax] Fused " << fused_bias << " bias+softmax, "
                  << fused_plain << " plain softmax patterns" << std::endl;
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
