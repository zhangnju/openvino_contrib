// FusedRTD fusion: matches output-side Reshape→Transpose→Reshape chains after
// MatMul or FullyConnected, replacing with FusedReshapeTransposeDot that compiles
// to a single rocMLIR mlir_reshape_transpose_reshape_dot kernel.
//
// Pattern A: MatMul[4D] → Transpose → Reshape (attention output head merge)
// Pattern B: MatMul/FC → Reshape → Transpose → Reshape (window reverse)

#include "fused_rtd_fusion.hpp"
#include "nodes/fused_rtd_node.hpp"

#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/core/graph_util.hpp>
#include "transformer/nodes/fully_connected.hpp"
#include <iostream>

namespace ov {
namespace rocm_gpu {
namespace pass {

// Try to match output-side Reshape → Transpose → Reshape chain.
// Returns the final Reshape node, fills out_r1/out_perm/out_r2.
// start_node is the MatMul/FC whose output we're walking from.
static std::shared_ptr<ov::op::v1::Reshape> match_output_rtr(
    const std::shared_ptr<ov::Node>& start_node,
    std::vector<int64_t>& out_r1,
    std::vector<int64_t>& out_perm,
    std::vector<int64_t>& out_r2,
    std::shared_ptr<ov::op::v1::Transpose>& out_transpose) {

    auto consumers = start_node->output(0).get_target_inputs();
    if (consumers.size() != 1) return nullptr;
    auto first_consumer = consumers.begin()->get_node()->shared_from_this();

    // Pattern A: → Transpose → Reshape
    auto tr = std::dynamic_pointer_cast<ov::op::v1::Transpose>(first_consumer);
    if (tr && tr->output(0).get_target_inputs().size() == 1) {
        auto perm_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            tr->get_input_node_shared_ptr(1));
        if (perm_const) {
            out_perm = perm_const->cast_vector<int64_t>();
            auto tp_consumer = tr->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            auto final_rsp = std::dynamic_pointer_cast<ov::op::v1::Reshape>(tp_consumer);
            if (final_rsp) {
                out_transpose = tr;
                auto mm_shape = start_node->get_output_shape(0);
                out_r1.assign(mm_shape.begin(), mm_shape.end());
                auto r2_shape = final_rsp->get_output_shape(0);
                out_r2.assign(r2_shape.begin(), r2_shape.end());
                return final_rsp;
            }
        }
    }

    // Pattern B: → Reshape → Transpose → Reshape
    auto reshape1 = std::dynamic_pointer_cast<ov::op::v1::Reshape>(first_consumer);
    if (!reshape1 || reshape1->output(0).get_target_inputs().size() != 1) return nullptr;

    tr = std::dynamic_pointer_cast<ov::op::v1::Transpose>(
        reshape1->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
    if (!tr || tr->output(0).get_target_inputs().size() != 1) return nullptr;

    auto perm_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        tr->get_input_node_shared_ptr(1));
    if (!perm_const) return nullptr;
    out_perm = perm_const->cast_vector<int64_t>();

    auto final_rsp = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
        tr->output(0).get_target_inputs().begin()->get_node()->shared_from_this());
    if (!final_rsp) return nullptr;

    out_transpose = tr;
    auto r1_shape = reshape1->get_output_shape(0);
    out_r1.assign(r1_shape.begin(), r1_shape.end());
    auto r2_shape = final_rsp->get_output_shape(0);
    out_r2.assign(r2_shape.begin(), r2_shape.end());
    return final_rsp;
}

bool FusedRTDFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int count = 0;

    for (const auto& node : model->get_ordered_ops()) {
        // Match MatMul or FullyConnected
        ov::Output<ov::Node> input_a, input_b;
        bool transpose_b = false;
        bool is_gemm = false;

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        auto fc = std::dynamic_pointer_cast<nodes::FullyConnected>(node);

        if (matmul) {
            if (matmul->get_input_element_type(0) != ov::element::f16) continue;
            input_a = matmul->input_value(0);
            input_b = matmul->input_value(1);
            transpose_b = matmul->get_transpose_b();
            is_gemm = true;
        } else if (fc) {
            if (fc->get_input_element_type(0) != ov::element::f16) continue;
            input_a = fc->input_value(0);
            input_b = fc->input_value(1);
            transpose_b = fc->get_transpose_b();
            is_gemm = true;
        }

        if (!is_gemm) continue;

        // ── Output-side: check for Reshape → Transpose → Reshape ──
        std::vector<int64_t> out_r1, out_perm, out_r2;
        std::shared_ptr<ov::op::v1::Transpose> out_transpose;
        auto final_reshape = match_output_rtr(node, out_r1, out_perm, out_r2, out_transpose);

        if (!final_reshape) continue;

        auto mm_out_ps = node->get_output_partial_shape(0);
        if (!mm_out_ps.is_static()) continue;
        auto mm_shape = mm_out_ps.to_shape();

        if (mm_shape.size() != 2 && mm_shape.size() != 4) continue;

        auto r2_shape = final_reshape->get_output_shape(0);

        auto rtd = std::make_shared<nodes::FusedReshapeTransposeDot>(
            input_a, input_b, transpose_b,
            out_r1, out_perm, out_r2, r2_shape);

        ov::replace_output_update_name(final_reshape->output(0), rtd->output(0));
        count++;
        changed = true;
    }

    if (count > 0)
        std::cerr << "[FusedRTD] Fused " << count
                  << " MatMul/FC→Reshape→Transpose→Reshape patterns" << std::endl;
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
