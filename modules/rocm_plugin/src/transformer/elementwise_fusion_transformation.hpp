// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Elementwise fusion transformation: finds chains of elementwise ops and
// replaces them with a single FusedElementwise node.
//
// Design inspired by LingLong's ElementwiseFuser (union-find based grouping)
// and AITemplate's fuse_ops.py:fuse_elementwise.
//
// For yolo26x this eliminates ~148+ separate Swish/Mul/Add kernel launches.

#pragma once

#include <openvino/pass/pass.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "openvino/core/graph_util.hpp"

#include "transformer/nodes/fused_elementwise_node.hpp"

#include <openvino/op/relu.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/tanh.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/hard_sigmoid.hpp>
#include <openvino/op/hswish.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/log.hpp>
#include <openvino/op/erf.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: is this node a fuseable elementwise op?
// ─────────────────────────────────────────────────────────────────────────────

static inline bool is_elementwise_fuseable(const std::shared_ptr<ov::Node>& n) {
    if (!n) return false;
    // Skip nodes that produce outputs consumed by multiple consumers —
    // they can't be absorbed into a linear chain without duplication.
    // (The pass handles multi-consumer detection during chain extraction.)
    const auto* type = &n->get_type_info();
    return (
        ov::is_type<ov::op::v0::Relu>(n)       ||
        ov::is_type<ov::op::v0::Sigmoid>(n)     ||
        ov::is_type<ov::op::v4::Swish>(n)       ||
        ov::is_type<ov::op::v0::Tanh>(n)        ||
        ov::is_type<ov::op::v7::Gelu>(n)        ||
        ov::is_type<ov::op::v4::Swish>(n)       ||
        ov::is_type<ov::op::v4::HSwish>(n)      ||
        ov::is_type<ov::op::v1::Multiply>(n)    ||
        ov::is_type<ov::op::v1::Add>(n)         ||
        ov::is_type<ov::op::v1::Subtract>(n)    ||
        ov::is_type<ov::op::v1::Divide>(n)      ||
        ov::is_type<ov::op::v0::Abs>(n)         ||
        ov::is_type<ov::op::v0::Negative>(n)    ||
        ov::is_type<ov::op::v0::Sqrt>(n)        ||
        ov::is_type<ov::op::v0::Exp>(n)         ||
        ov::is_type<ov::op::v0::Log>(n)         ||
        ov::is_type<ov::op::v0::Erf>(n)
    );
    (void)type;  // suppress unused warning
    return false;
}

// Convert OV node to FusedEwStep description
static inline bool to_fused_step(const std::shared_ptr<ov::Node>& n,
                                  const std::shared_ptr<ov::Node>& prev_node,
                                  nodes::FusedEwStep& step)
{
    step.param   = 0.f;
    step.has_aux = false;

    if      (ov::is_type<ov::op::v0::Relu>(n))     { step.op_type = "Relu"; }
    else if (ov::is_type<ov::op::v0::Sigmoid>(n))  { step.op_type = "Sigmoid"; }
    else if (ov::is_type<ov::op::v4::Swish>(n))    { step.op_type = "Swish"; }
    else if (ov::is_type<ov::op::v0::Tanh>(n))     { step.op_type = "Tanh"; }
    else if (ov::is_type<ov::op::v7::Gelu>(n))     { step.op_type = "Gelu"; }
    else if (ov::is_type<ov::op::v4::HSwish>(n))   { step.op_type = "HardSwish"; }
    else if (ov::is_type<ov::op::v0::Abs>(n))      { step.op_type = "Abs"; }
    else if (ov::is_type<ov::op::v0::Negative>(n)) { step.op_type = "Neg"; }
    else if (ov::is_type<ov::op::v0::Sqrt>(n))     { step.op_type = "Sqrt"; }
    else if (ov::is_type<ov::op::v0::Exp>(n))      { step.op_type = "Exp"; }
    else if (ov::is_type<ov::op::v0::Log>(n))      { step.op_type = "Log"; }
    else if (ov::is_type<ov::op::v0::Erf>(n))      { step.op_type = "Erf"; }
    else if (ov::is_type<ov::op::v1::Multiply>(n) ||
             ov::is_type<ov::op::v1::Add>(n)      ||
             ov::is_type<ov::op::v1::Subtract>(n) ||
             ov::is_type<ov::op::v1::Divide>(n))
    {
        // Binary op: one input comes from the chain (prev_node), the other is auxiliary
        if (ov::is_type<ov::op::v1::Multiply>(n))  step.op_type = "Multiply";
        else if (ov::is_type<ov::op::v1::Add>(n))  step.op_type = "Add";
        else if (ov::is_type<ov::op::v1::Subtract>(n)) step.op_type = "Subtract";
        else step.op_type = "Divide";
        step.has_aux = true;
    }
    else return false;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// ElementwiseFusionPass: ModelPass that scans graph for fuseable chains
// ─────────────────────────────────────────────────────────────────────────────

class ElementwiseFusionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ElementwiseFusionPass", "0");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        bool changed = false;
        bool local_changed = true;

        // Iterate until no more fusions are possible (chains can be extended)
        while (local_changed) {
            local_changed = false;
            const auto& ops = model->get_ordered_ops();

            for (auto& root : ops) {
                if (!is_elementwise_fuseable(root)) continue;

                // Only start a chain from nodes whose primary input is NOT a fuseable ew op
                // (i.e., is the "head" of a chain)
                const auto& in0_node = root->get_input_node_shared_ptr(0);
                if (is_elementwise_fuseable(in0_node)) continue;


                // Check the root is not a graph parameter/constant
                if (ov::is_type<ov::op::v0::Parameter>(root) ||
                    ov::is_type<ov::op::v0::Constant>(root)) continue;

                // Extend the chain greedily
                std::vector<std::shared_ptr<ov::Node>> chain = {root};
                std::shared_ptr<ov::Node> cur = root;

                while (chain.size() < static_cast<size_t>(nodes::FusedElementwise::kMaxChain)) {
                    // Find the single consumer of cur's output[0]
                    if (cur->get_output_size() == 0) break;
                    auto output = cur->output(0);
                    auto consumers = output.get_target_inputs();
                    if (consumers.size() != 1) break;

                    auto next = consumers.begin()->get_node()->shared_from_this();
                    if (!is_elementwise_fuseable(next)) break;

                    // For binary ops, the non-chain input must come from outside the chain
                    // (to avoid cycles and to correctly set up the aux pointer)
                    if (next->get_input_size() == 2) {
                        auto in0 = next->get_input_node_shared_ptr(0);
                        auto in1 = next->get_input_node_shared_ptr(1);
                        // At least one input must be cur (chain link)
                        bool in0_is_cur = (in0.get() == cur.get());
                        bool in1_is_cur = (in1.get() == cur.get());
                        if (!in0_is_cur && !in1_is_cur) break;
                    }

                    chain.push_back(next);
                    cur = next;
                }

                if (chain.size() < 2) continue;  // no fusion benefit for single op

                // Build FusedElementwise node
                std::vector<nodes::FusedEwStep> steps;
                ov::OutputVector fused_inputs = {root->input_value(0)};

                auto* prev = root.get();
                for (size_t ci = 0; ci < chain.size(); ++ci) {
                    auto& node = chain[ci];
                    nodes::FusedEwStep step;
                    if (!to_fused_step(node, chain[ci == 0 ? 0 : ci - 1], step)) break;

                    if (step.has_aux && node->get_input_size() == 2) {
                        // Add the aux (non-chain) input
                        auto in0 = node->input_value(0).get_node_shared_ptr();
                        auto in1 = node->input_value(1).get_node_shared_ptr();
                        // For ci=0 (root node), in0 is always the primary chain link
                        // (the chain starts from root->input_value(0) = fused_inputs[0]).
                        // For ci>0, check if in0 comes from the previous chain node.
                        bool in0_is_prev = (ci == 0) ?
                            true :  // in0 of root = fused_inputs[0], always the chain link
                            (in0.get() == chain[ci-1].get());
                        if (in0_is_prev) {
                            fused_inputs.push_back(node->input_value(1));
                        } else {
                            fused_inputs.push_back(node->input_value(0));
                        }
                    }
                    steps.push_back(step);
                    (void)prev;
                    prev = node.get();
                }

                if (steps.size() < 2) continue;  // didn't build a valid multi-step chain

                // Trace: print FusedElementwise chain (enabled by ROCMLIR_TRACE_EWFUSION=1)
                if (std::getenv("ROCMLIR_TRACE_EWFUSION")) {
                    const auto& src = root->get_input_node_shared_ptr(0);
                    std::string ops_str;
                    for (auto& n : chain) {
                        if (!ops_str.empty()) ops_str += "→";
                        ops_str += n->get_type_info().name;
                        if (n->get_input_size() == 2) {
                            for (size_t i = 0; i < 2; ++i) {
                                auto aux = n->get_input_node_shared_ptr(i);
                                if (aux.get() != root.get())
                                    ops_str += "[aux=" + std::string(aux->get_type_info().name) + "]";
                            }
                        }
                    }
                    std::cerr << "[EWFusion] src=" << src->get_type_info().name
                              << " chain=" << ops_str
                              << " #aux=" << (fused_inputs.size() - 1) << "\n";
                }

                // Create the fused node
                auto fused = std::make_shared<nodes::FusedElementwise>(fused_inputs, steps);
                fused->set_friendly_name(chain.back()->get_friendly_name() + "_fused");
                ov::copy_runtime_info(chain.back(), fused);

                // Replace the last node in the chain with the fused node
                ov::replace_node(chain.back(), fused);
                local_changed = true;
                changed = true;
                break;  // Restart scan after modification
            }
        }

        return changed;
    }
};

// Maximum chain length for the FusedElementwise node (same as kernel)
inline constexpr int kFusedEwMaxChain_node = 16;

// Add this to nodes::FusedElementwise for convenient access
}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov

// Patch: make kMaxChain accessible from the node class
namespace ov {
namespace rocm_gpu {
namespace nodes {
struct FusedElementwise;
}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
