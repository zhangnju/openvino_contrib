// L2Normalize fusion: matches the OV-decomposed ReduceL2 pattern and replaces
// with a single L2Normalize node.
//
// OV decomposes ONNX ReduceL2 into: Power(x,2) → ReduceSum(axis,keepdims) → Sqrt
// Then the ONNX L2Norm pattern becomes:
//   x → [Power(2) → ReduceSum → Sqrt → Maximum(eps)] → Divide(x, norm)
//
// We match backwards from Divide: find Divide where one input is x and the other
// is Maximum(Sqrt(ReduceSum(Power(x,2)))), and x is the same in both branches.

#include "l2_normalize_fusion.hpp"
#include "nodes/l2_normalize_node.hpp"

#include <openvino/op/divide.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/core/graph_util.hpp>
#include <iostream>

namespace ov {
namespace rocm_gpu {
namespace pass {

bool L2NormalizeFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int count = 0;

    for (const auto& node : model->get_ordered_ops()) {
        auto divide = std::dynamic_pointer_cast<ov::op::v1::Divide>(node);
        if (!divide) continue;
        if (divide->get_input_element_type(0) != ov::element::f16) continue;

        // Divide has 2 inputs: x and norm
        // Try both orderings
        for (int order = 0; order < 2; order++) {
            auto x_out = divide->input_value(order);
            auto norm_input = divide->get_input_node_shared_ptr(1 - order);

            // norm_input should be Maximum or Clamp (clamps the norm to >= eps)
            float eps = 1e-12f;
            std::shared_ptr<ov::Node> sqrt_node;

            auto max_node = std::dynamic_pointer_cast<ov::op::v1::Maximum>(norm_input);
            auto clamp_node = std::dynamic_pointer_cast<ov::op::v0::Clamp>(norm_input);

            if (max_node) {
                // Maximum(sqrt_result, eps_const)
                auto c0 = std::dynamic_pointer_cast<ov::op::v0::Constant>(max_node->get_input_node_shared_ptr(0));
                auto c1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(max_node->get_input_node_shared_ptr(1));
                if (c0) { eps = c0->cast_vector<float>()[0]; sqrt_node = max_node->get_input_node_shared_ptr(1); }
                else if (c1) { eps = c1->cast_vector<float>()[0]; sqrt_node = max_node->get_input_node_shared_ptr(0); }
                else continue;
            } else if (clamp_node) {
                eps = clamp_node->get_min();
                sqrt_node = clamp_node->get_input_node_shared_ptr(0);
            } else {
                continue;
            }

            // sqrt_node should be Sqrt
            if (!sqrt_node || sqrt_node->get_type_name() != std::string("Sqrt")) continue;
            if (sqrt_node->output(0).get_target_inputs().size() > 2) continue;

            // Sqrt input should be ReduceSum
            auto reduce = sqrt_node->get_input_node_shared_ptr(0);
            if (reduce->get_type_name() != std::string("ReduceSum")) continue;
            if (reduce->output(0).get_target_inputs().size() != 1) continue;

            // ReduceSum input should be Power(x, 2) or Multiply(x, x)
            auto power_node = reduce->get_input_node_shared_ptr(0);
            ov::Output<ov::Node> original_x;

            if (power_node->get_type_name() == std::string("Power")) {
                auto exp_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                    power_node->get_input_node_shared_ptr(1));
                if (!exp_const) continue;
                float exp_val = exp_const->cast_vector<float>()[0];
                if (std::abs(exp_val - 2.0f) > 0.01f) continue;
                original_x = power_node->input_value(0);
            } else if (power_node->get_type_name() == std::string("Multiply")) {
                // Multiply(x, x) — same input
                if (power_node->input_value(0) != power_node->input_value(1)) continue;
                original_x = power_node->input_value(0);
            } else {
                continue;
            }

            // Verify x in Divide matches x in Power/Multiply
            if (original_x != x_out) continue;

            // Get reduce axis
            auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                reduce->get_input_node_shared_ptr(1));
            if (!axis_const) continue;
            auto axes = axis_const->cast_vector<int64_t>();
            if (axes.size() != 1) continue;
            int axis = axes[0];

            // Create L2Normalize node
            auto l2n = std::make_shared<nodes::L2Normalize>(original_x, axis, eps);
            ov::replace_output_update_name(divide->output(0), l2n->output(0));
            count++;
            changed = true;
            break;  // matched, skip other ordering
        }
    }

    if (count > 0)
        std::cerr << "[L2Normalize] Fused " << count << " L2Normalize patterns" << std::endl;
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
