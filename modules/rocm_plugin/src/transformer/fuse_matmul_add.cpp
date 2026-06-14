// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "fuse_matmul_add.hpp"
#include <unordered_set>

#include "openvino/core/rt_info.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <ops/matmul.hpp>

using namespace ov::pass::pattern;

namespace {

// Get the underlying Constant, unwrapping a single Convert if present.
// After ConvertPrecision, bias constants appear as Convert(f32→f16, Constant).
static std::shared_ptr<ov::op::v0::Constant> get_bias_constant(const std::shared_ptr<ov::Node>& n) {
    if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(n)) return c;
    if (auto cv = std::dynamic_pointer_cast<ov::op::v0::Convert>(n)) {
        if (cv->output(0).get_target_inputs().size() == 1)
            return std::dynamic_pointer_cast<ov::op::v0::Constant>(cv->get_input_node_shared_ptr(0));
    }
    return nullptr;
}

std::pair<std::shared_ptr<ov::op::v0::MatMul>, std::shared_ptr<ov::op::v0::Constant>> get_matmul_constant_nodes(const std::shared_ptr<ov::Node>& add_node) {
    auto bias1 = get_bias_constant(add_node->get_input_node_shared_ptr(1));
    auto bias0 = get_bias_constant(add_node->get_input_node_shared_ptr(0));
    if (bias1) {
        return {std::dynamic_pointer_cast<ov::op::v0::MatMul>(add_node->get_input_node_shared_ptr(0)), bias1};
    } else if (bias0) {
        return {std::dynamic_pointer_cast<ov::op::v0::MatMul>(add_node->get_input_node_shared_ptr(1)), bias0};
    }
    return {nullptr, nullptr};
}

bool is_add_to_be_fused(const ov::Output<ov::Node>& output) {
    auto add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(output.get_node_shared_ptr());
    if (!add_node) {
        return false;
    }
    // Note: is_dynamic() check removed — BERT uses dynamic batch, which is OK.
    std::shared_ptr<ov::op::v0::MatMul> matmul_node;
    std::shared_ptr<ov::op::v0::Constant> constant_node;
    std::tie(matmul_node, constant_node) = get_matmul_constant_nodes(add_node);
    if (!matmul_node || !constant_node) {
        return false;
    }

    // Bias [C] or [1,...,1,C] valid for matmul output [...,R,C].
    // Check last dim matches and all other bias dims are 1.
    const auto& ps_out = matmul_node->get_output_partial_shape(0);
    if (!ps_out.rank().is_static()) return false;
    const auto& cs = constant_node->get_output_shape(0);
    if (cs.empty()) return false;
    // The last dim of the bias must equal the last dim of the MatMul output
    if (!ps_out[ps_out.rank().get_length()-1].is_static()) return false;
    size_t n_cols = ps_out[ps_out.rank().get_length()-1].get_length();
    if (cs.back() != n_cols) return false;
    for (size_t i = 0; i + 1 < cs.size(); ++i)
        if (cs[i] != 1) return false;

    // Safety: FullyConnected output dtype must match MatMul output dtype.
    // If Add promotes type (e.g., f16 MatMul + f32 bias → f32 Add), fusing would
    // change downstream dtype and break consumers expecting f32.
    if (matmul_node->get_output_element_type(0) != add_node->get_output_element_type(0))
        return false;

    return true;
}
} // namespace

namespace ov::rocm_gpu::pass {
bool fuse_matmul_and_add(Matcher &m) {
    auto add_node = std::dynamic_pointer_cast<ov::op::v1::Add>(m.get_match_root());
    if (!add_node) return false;
    // Validate everything in the callback
    std::shared_ptr<ov::op::v0::MatMul> matmul_node;
    std::shared_ptr<ov::op::v0::Constant> constant_node;
    std::tie(matmul_node, constant_node) = get_matmul_constant_nodes(add_node);
    if (!matmul_node || !constant_node) return false;
    if (!is_add_to_be_fused(add_node->output(0))) return false;

    // Determine which add input is the bias (the non-MatMul one)
    ov::Output<ov::Node> bias_output;
    auto in0 = add_node->get_input_node_shared_ptr(0);
    if (in0.get() == matmul_node.get()) {
        bias_output = add_node->input_value(1);
    } else {
        bias_output = add_node->input_value(0);
    }
    const auto fully_connected_node =
        std::make_shared<ov::rocm_gpu::nodes::FullyConnected>(matmul_node->get_input_source_output(0),
                                                                matmul_node->get_input_source_output(1),
                                                                bias_output,
                                                                matmul_node->get_transpose_a(),
                                                                matmul_node->get_transpose_b());
    fully_connected_node->set_friendly_name(add_node->get_friendly_name());
    ov::copy_runtime_info({matmul_node, add_node}, fully_connected_node);

    ov::replace_node(add_node, fully_connected_node);
    // Note: if matmul_node now has no consumers, DeadMatMulElimination will clean it up.
    return true;
}

DeadMatMulElimination::DeadMatMulElimination() {}

// Check if a node is "transitively dead" (all its consumers are also dead)
static bool is_transitively_dead(const std::shared_ptr<ov::Node>& node,
                                  std::unordered_set<ov::Node*>& dead_set) {
    if (dead_set.count(node.get())) return true;
    // A node is dead if it is not a Result/Parameter and all its output consumers are dead
    if (ov::is_type<ov::op::v0::Result>(node) || ov::is_type<ov::op::v0::Parameter>(node))
        return false;
    for (size_t o = 0; o < node->get_output_size(); ++o) {
        for (const auto& t : node->output(o).get_target_inputs()) {
            auto consumer = t.get_node()->shared_from_this();
            if (!is_transitively_dead(consumer, dead_set)) return false;
        }
    }
    dead_set.insert(node.get());
    return true;
}

bool DeadMatMulElimination::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    // Build set of transitively-dead nodes (nodes whose outputs only reach dead nodes)
    std::unordered_set<ov::Node*> dead_set;
    std::vector<std::shared_ptr<ov::op::v0::MatMul>> dead_matmuls;

    for (const auto& node : model->get_ordered_ops()) {
        auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        if (!mm) continue;
        if (is_transitively_dead(mm, dead_set)) {
            dead_matmuls.push_back(mm);
        }
    }
    // fprintf(stderr, "[DCE] Dead MatMul: %zu\n", dead_matmuls.size());
    for (auto& mm : dead_matmuls) {
        // Replace the dead MatMul with a zero constant so it can be cleaned up
        // This breaks the edge from MatMul to its dead consumer (Add)
        auto shape = mm->get_output_shape(0);
        auto dtype = mm->get_output_element_type(0);
        auto zero = std::make_shared<ov::op::v0::Constant>(dtype, shape, 0);
        mm->output(0).replace(zero->output(0));
        changed = true;
    }
    return changed;
}

FullyConnectedTransformation::FullyConnectedTransformation() {
    MATCHER_SCOPE(FullyConnectedTransformation);
    // Match any Add node — validate everything in the callback.
    auto result = wrap_type<ov::op::v1::Add>();

    matcher_pass_callback callback = [](Matcher &m) { return fuse_matmul_and_add(m); };

    auto m = std::make_shared<Matcher>(result, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::rocm_gpu::pass
