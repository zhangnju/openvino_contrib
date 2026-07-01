// QKVTransposeSplit fusion: finds Reshape → Transpose → 3×Gather(0,1,2) pattern
// in Swin attention blocks and replaces with a single fused kernel.
//
// Pattern: x [nW, sq, 3*H*hd]
//   → Reshape [nW, sq, 3, H, hd]
//   → Transpose perm=[2,0,3,1,4] → [3, nW, H, sq, hd]
//   → Gather(axis=0, idx=0) → Q [nW, H, sq, hd]
//   → Gather(axis=0, idx=1) → K [nW, H, sq, hd]
//   → Gather(axis=0, idx=2) → V [nW, H, sq, hd]

#include "qkv_transpose_split_fusion.hpp"
#include "nodes/qkv_transpose_split_node.hpp"

#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/core/graph_util.hpp>
#include <iostream>
#include <set>

namespace ov {
namespace rocm_gpu {
namespace pass {

bool QKVTransposeSplitFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int fused_count = 0;

    // Find all Transpose nodes with perm that moves dim 2 to front
    // (the "3" dim in [nW, sq, 3, H, hd] → [3, nW, H, sq, hd])
    auto ops = model->get_ordered_ops();
    std::set<ov::Node*> removed;

    // Debug: count total ops by type
    int n_transpose = 0, n_gather = 0, n_reshape = 0;
    for (const auto& node : ops) {
        if (node->get_type_name() == std::string("Transpose")) n_transpose++;
        if (node->get_type_name() == std::string("Gather")) n_gather++;
        if (node->get_type_name() == std::string("Reshape")) n_reshape++;
    }
    std::cerr << "[QKVSplit] Graph has " << ops.size() << " ops: "
              << n_transpose << " Transpose, " << n_gather << " Gather, "
              << n_reshape << " Reshape" << std::endl;

    for (const auto& node : ops) {
        if (removed.count(node.get())) continue;
        auto transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(node);
        if (!transpose) continue;

        // Check perm: must be [2, 0, 3, 1, 4] (5D) for standard Swin QKV
        auto perm_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            transpose->get_input_node_shared_ptr(1));
        if (!perm_const) continue;
        auto perm = perm_const->cast_vector<int64_t>();
        if (perm.size() != 5) continue;
        // Accept [2,0,3,1,4] — the standard Swin QKV transpose
        if (perm[0] != 2 || perm[1] != 0 || perm[2] != 3 || perm[3] != 1 || perm[4] != 4)
            continue;

        // Transpose must have exactly 3 consumers, all Gather with axis=0
        auto consumers = transpose->output(0).get_target_inputs();
        std::cerr << "[QKVSplit] Found Transpose perm=[2,0,3,1,4] with "
                  << consumers.size() << " consumers: ";
        for (const auto& inp : consumers)
            std::cerr << inp.get_node()->get_type_name() << " ";
        std::cerr << std::endl;

        // All consumers must be Gather with axis=0 and scalar index in {0,1,2}
        // Group by index: may have multiple Gathers with same index (shifted window)
        std::vector<std::shared_ptr<ov::Node>> gather_by_idx[3];
        bool all_gather = true;
        for (const auto& inp : consumers) {
            auto consumer = inp.get_node()->shared_from_this();
            if (consumer->get_type_name() != std::string("Gather")) {
                all_gather = false; break;
            }
            auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                consumer->get_input_node_shared_ptr(2));
            if (!axis_const || axis_const->cast_vector<int64_t>()[0] != 0) {
                all_gather = false; break;
            }
            auto idx_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                consumer->get_input_node_shared_ptr(1));
            if (!idx_const) { all_gather = false; break; }
            auto idx_val = idx_const->cast_vector<int64_t>();
            if (idx_val.size() != 1 || idx_val[0] < 0 || idx_val[0] > 2) {
                all_gather = false; break;
            }
            gather_by_idx[idx_val[0]].push_back(consumer);
        }
        std::cerr << "[QKVSplit]   Gather groups: idx0=" << gather_by_idx[0].size()
                  << " idx1=" << gather_by_idx[1].size()
                  << " idx2=" << gather_by_idx[2].size()
                  << " all_gather=" << all_gather << std::endl;
        if (!all_gather || gather_by_idx[0].empty() ||
            gather_by_idx[1].empty() || gather_by_idx[2].empty()) continue;

        // Check Transpose input: should be Reshape
        auto tp_input = transpose->get_input_node_shared_ptr(0);
        std::cerr << "[QKVSplit]   Transpose input type: " << tp_input->get_type_name()
                  << " shape=" << transpose->get_input_partial_shape(0) << std::endl;

        auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(tp_input);
        if (!reshape) continue;

        auto tp_in_ps = transpose->get_input_partial_shape(0);
        if (!tp_in_ps.is_static()) continue;
        auto rs_out_shape = tp_in_ps.to_shape();
        if (rs_out_shape.size() != 5 || rs_out_shape[2] != 3) {
            std::cerr << "[QKVSplit]   Reshape output not [nW,sq,3,H,hd], got "
                      << tp_in_ps << std::endl;
            continue;
        }

        int nW = rs_out_shape[0];
        int sq = rs_out_shape[1];
        int H  = rs_out_shape[3];
        int hd = rs_out_shape[4];

        bool is_fp16 = (reshape->get_input_element_type(0) == ov::element::f16);

        // Create QKVTransposeSplit node
        auto qkv_node = std::make_shared<nodes::QKVTransposeSplit>(
            reshape->input_value(0),  // feed Reshape's input directly
            nW, sq, H, hd, is_fp16);

        // Replace all Gather outputs: multiple Gathers with same index share one output
        for (int i = 0; i < 3; i++) {
            for (auto& g : gather_by_idx[i]) {
                ov::replace_output_update_name(g->output(0), qkv_node->output(i));
                removed.insert(g.get());
            }
        }
        removed.insert(transpose.get());

        fused_count++;
        changed = true;
    }

    if (fused_count > 0)
        std::cerr << "[QKVTransposeSplit] Fused " << fused_count
                  << " Reshape+Transpose+3×Gather patterns" << std::endl;
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
