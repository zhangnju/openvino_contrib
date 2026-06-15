// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dead_code_elimination.hpp"

#include <queue>
#include <unordered_set>
#include <openvino/op/constant.hpp>
#include <openvino/core/graph_util.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

bool RocmDCE::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // Step 1: BFS backward from Results to find all live nodes
    std::unordered_set<ov::Node*> live;
    std::queue<ov::Node*> bfs;

    for (auto& r : model->get_results())
        bfs.push(r.get());
    while (!bfs.empty()) {
        auto* n = bfs.front(); bfs.pop();
        if (!live.insert(n).second) continue;
        for (size_t i = 0; i < n->get_input_size(); ++i)
            bfs.push(n->get_input_node_ptr(i));
    }
    // Parameters and Constants feeding live nodes are marked live by the BFS above.
    // Explicitly mark Parameters too.
    for (auto& p : model->get_parameters())
        live.insert(p.get());

    // Step 2: Collect dead nodes (topological order, reversed)
    auto ops = model->get_ordered_ops();
    std::vector<std::shared_ptr<ov::Node>> dead;
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        auto& node = *it;
        if (!live.count(node.get()))
            dead.push_back(node);
    }
    if (dead.empty()) return false;

    // Step 3: Disconnect dead nodes from their live inputs.
    // For each dead node's input that comes from a LIVE node, we replace
    // that input connection with a zero Constant. This removes the dead
    // node from the live node's consumer list.
    // We process dead nodes in reverse topological order so that by the
    // time we process a dead node, all its dead consumers are already
    // disconnected and we can safely reassign its connections.
    size_t removed = 0;
    for (auto& node : dead) {
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            auto src = node->input_value(i);
            auto* src_node = src.get_node();
            if (!live.count(src_node)) continue;  // source is also dead, skip

            // Source is live but consumer (this node) is dead.
            // Disconnect by replacing the input with a zero Constant.
            auto et    = src.get_element_type();
            auto shape = src.get_partial_shape();
            if (!shape.rank().is_static()) continue;

            // Use shape {0} for scalars or the exact shape for small tensors.
            // For large tensors (weights), use shape {1} to avoid memory.
            ov::Shape cst_shape;
            bool is_small = ov::shape_size(shape.to_shape()) <= 4096;
            cst_shape = is_small ? shape.to_shape() : ov::Shape{1};

            auto dummy = ov::op::v0::Constant::create(et, cst_shape, {0});
            try {
                node->input(i).replace_source_output(dummy->output(0));
                ++removed;
            } catch (...) {}
        }
    }

    fprintf(stderr, "[RocmDCE] Removed %zu inputs from %zu dead nodes\n",
            removed, dead.size());
    return removed > 0;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
