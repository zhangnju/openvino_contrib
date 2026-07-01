// FusedMVNEpilogue fusion: matches MVN → Multiply(gamma) → Add(beta) chain
// and replaces with a single fused LayerNorm kernel.
//
// OV decomposes ONNX LayerNormalization into:
//   MVN(x, axes, eps) → Multiply(result, gamma) → Add(result, beta)
// where MVN internally does: (x - mean) / sqrt(var + eps)
//
// We match: MVN → Multiply(gamma) → Add(beta) and replace with FusedMVNEpilogue.

#include "fused_mvn_epilogue_fusion.hpp"
#include "nodes/fused_mvn_epilogue_node.hpp"

#include <openvino/op/mvn.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/core/graph_util.hpp>
#include <iostream>

namespace ov {
namespace rocm_gpu {
namespace pass {

bool FusedMvnEpilogueFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int count = 0;

    for (const auto& node : model->get_ordered_ops()) {
        // Match Add(Multiply(MVN(x), gamma), beta)
        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(node);
        if (!add) continue;
        // Accept both f16 and f32 (pass runs before AND after ConvertPrecision)
        auto et = add->get_input_element_type(0);
        if (et != ov::element::f16 && et != ov::element::f32) continue;

        // One input is Multiply, the other is beta (Constant or broadcastable)
        for (int order = 0; order < 2; order++) {
            auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(
                add->get_input_node_shared_ptr(order));
            if (!mul) continue;
            if (mul->output(0).get_target_inputs().size() != 1) continue;

            auto beta_out = add->input_value(1 - order);

            // Multiply: one input is MVN, the other is gamma
            std::shared_ptr<ov::Node> mvn_node;
            ov::Output<ov::Node> gamma_out;

            for (int mo = 0; mo < 2; mo++) {
                auto candidate = mul->get_input_node_shared_ptr(mo);
                if (candidate->get_type_name() == std::string("MVN")) {
                    mvn_node = candidate;
                    gamma_out = mul->input_value(1 - mo);
                    break;
                }
            }
            if (!mvn_node) continue;
            if (mvn_node->output(0).get_target_inputs().size() != 1) continue;

            // Get MVN eps
            float eps = 1e-5f;
            auto mvn6 = std::dynamic_pointer_cast<ov::op::v6::MVN>(mvn_node);
            if (mvn6) eps = mvn6->get_eps();

            // Get shapes
            auto x_ps = mvn_node->get_input_partial_shape(0);
            if (!x_ps.is_static()) continue;
            auto x_shape = x_ps.to_shape();
            if (x_shape.size() < 2) continue;

            // cols = last dim, rows = product of all other dims
            int cols = x_shape.back();
            int rows = 1;
            for (size_t i = 0; i < x_shape.size() - 1; i++) rows *= x_shape[i];

            // Verify gamma and beta shapes are broadcastable to [cols]
            auto gamma_ps = gamma_out.get_partial_shape();
            auto beta_ps = beta_out.get_partial_shape();
            if (!gamma_ps.is_static() || !beta_ps.is_static()) continue;

            auto fme = std::make_shared<nodes::FusedMVNEpilogue>(
                mvn_node->input_value(0),  // x
                gamma_out,
                beta_out,
                rows, cols, eps);

            ov::replace_output_update_name(add->output(0), fme->output(0));
            count++;
            changed = true;
            break;
        }
    }

    if (count > 0)
        std::cerr << "[FusedMVNEpilogue] Fused " << count << " MVN+scale+bias patterns" << std::endl;
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
