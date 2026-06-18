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
#include <openvino/op/erf.hpp>
#include <openvino/op/gelu.hpp>  // includes both v0::Gelu and v7::Gelu
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
#include <openvino/op/round.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/one_hot.hpp>
#include <openvino/op/convert.hpp>
#include <cstdlib>

namespace ov {
namespace rocm_gpu {
namespace pass {

// Clamp is only fuseable when its bounds are exactly the u8 quant range (0,255),
// because the kernel hardcodes that range for the Clamp op code.
static inline bool is_u8_clamp(const std::shared_ptr<ov::Node>& n) {
    auto c = ov::as_type_ptr<ov::op::v0::Clamp>(n);
    return c && c->get_min() == 0.0 && c->get_max() == 255.0;
}

// A Convert(i32 -> f32) with a single consumer can be absorbed as the HEAD of a
// fused chain: the kernel reads the i32 primary and converts on load, making the
// Convert free. This collapses the post-MatMulInteger dequant epilogue
// (Convert -> Multiply(scale) -> Multiply(out_scale) -> BiasAdd) into one kernel.
static inline std::shared_ptr<ov::op::v0::Convert> as_i32_to_f32_convert_head(
        const std::shared_ptr<ov::Node>& n) {
    auto c = ov::as_type_ptr<ov::op::v0::Convert>(n);
    if (!c) return nullptr;
    if (c->get_input_element_type(0) != ov::element::i32) return nullptr;
    if (c->get_destination_type() != ov::element::f32) return nullptr;
    if (c->output(0).get_target_inputs().size() != 1) return nullptr;
    return c;
}

// Last-dim (per-channel) broadcast: aux total element count equals the primary's
// LAST dimension and all of aux's dims except the last are 1. The kernel can then
// read aux[i % C] (C = last dim). This is the common dequant pattern: a per-output-
// channel scale/bias [C] multiplied/added against an activation [..., C].
static inline bool is_last_dim_bcast(const ov::PartialShape& aux_ps,
                                     const ov::PartialShape& prim_ps) {
    if (!aux_ps.is_static() || !prim_ps.is_static()) return false;
    const auto a = aux_ps.to_shape();
    const auto p = prim_ps.to_shape();
    if (p.empty() || a.empty()) return false;
    const size_t C = p.back();
    if (ov::shape_size(a) != C || C == 1) return false;
    // aux must be all-ones except the last dim (which is C).
    for (size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] != 1) return false;
    return a.back() == C;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: is this node a fuseable elementwise op?
// ─────────────────────────────────────────────────────────────────────────────

static inline bool is_elementwise_fuseable(const std::shared_ptr<ov::Node>& n) {
    if (!n) return false;
    // FusedElementwise JIT kernel only supports f16/f32. Skip integer-typed nodes.
    const auto out_type = n->get_output_element_type(0);
    if (out_type != ov::element::f16 && out_type != ov::element::f32) return false;
    return (
        ov::is_type<ov::op::v0::Relu>(n)       ||
        ov::is_type<ov::op::v0::Sigmoid>(n)     ||
        ov::is_type<ov::op::v4::Swish>(n)       ||
        ov::is_type<ov::op::v0::Tanh>(n)        ||
        ov::is_type<ov::op::v7::Gelu>(n)        ||
        ov::is_type<ov::op::v0::Gelu>(n)        ||
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
        ov::is_type<ov::op::v0::Erf>(n)         ||
        ov::is_type<ov::op::v5::Round>(n)       ||
        is_u8_clamp(n)
        // Round/Clamp now fuseable: the kernel broadcasts a scalar aux (aux[0]) so the
        // INT8 dequant Mul/Div-by-scalar chains (x*scale → Round → Clamp(0,255)) fuse
        // into a single launch. Clamp only when bounds are exactly the u8 range (0,255),
        // which the Clamp op code hardcodes.
    );
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
    else if (ov::is_type<ov::op::v7::Gelu>(n) ||
             ov::is_type<ov::op::v0::Gelu>(n))    { step.op_type = "Gelu"; }
    else if (ov::is_type<ov::op::v4::HSwish>(n))   { step.op_type = "HardSwish"; }
    else if (ov::is_type<ov::op::v0::Abs>(n))      { step.op_type = "Abs"; }
    else if (ov::is_type<ov::op::v0::Negative>(n)) { step.op_type = "Neg"; }
    else if (ov::is_type<ov::op::v0::Sqrt>(n))     { step.op_type = "Sqrt"; }
    else if (ov::is_type<ov::op::v0::Exp>(n))      { step.op_type = "Exp"; }
    else if (ov::is_type<ov::op::v0::Log>(n))      { step.op_type = "Log"; }
    else if (ov::is_type<ov::op::v0::Erf>(n))      { step.op_type = "Erf"; }
    else if (ov::is_type<ov::op::v5::Round>(n))    { step.op_type = "Round"; }
    else if (ov::is_type<ov::op::v0::Clamp>(n))    { step.op_type = "Clamp"; }
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
                        // The FusedElementwise kernel reads aux either per-element (aux[i])
                        // or, when the aux is a single-element scalar, broadcast (aux[0]).
                        // It does NOT support general broadcasts. So fuse only if the aux
                        // element count equals the chain output OR is exactly 1 (scalar,
                        // e.g. the INT8 dequant x/span). Anything else would read OOB.
                        auto aux_out = in0_is_cur ? next->input_value(1) : next->input_value(0);
                        const auto& aux_ps   = aux_out.get_partial_shape();
                        const auto& chain_ps = cur->output(0).get_partial_shape();
                        if (!aux_ps.is_static() || !chain_ps.is_static()) break;
                        const size_t aux_n = ov::shape_size(aux_ps.to_shape());
                        const size_t chain_n = ov::shape_size(chain_ps.to_shape());
                        // Kernel supports aux that is: full (==chain), scalar (==1, read
                        // aux[0]), or last-dim/per-channel broadcast (read aux[i % C]).
                        if (aux_n != chain_n && aux_n != 1 && !is_last_dim_bcast(aux_ps, chain_ps)) break;
                    }

                    chain.push_back(next);
                    cur = next;
                }

                // Fuse single-op chains for activation functions (Gelu/Erf/etc.) that
                // cannot chain with their consumer but need to run in f16 (not standalone f32 kernel)
                const bool is_activation_root =
                    ov::is_type<ov::op::v7::Gelu>(root) ||
                    ov::is_type<ov::op::v0::Gelu>(root) ||
                    ov::is_type<ov::op::v0::Erf>(root)  ||
                    ov::is_type<ov::op::v0::Tanh>(root);
                if (chain.size() < 2 && !is_activation_root) continue;

                // Skip pure-scalar chains (output is a single element). These are the
                // INT8 zero-point computations (round(-xmin/scale)). They give no fusion
                // benefit (1 element) and their primary input is often a const-folded
                // scalar with no materialised device buffer → the kernel's primary_in[i]
                // read faults. Leave them as standalone ops.
                {
                    const auto& root_out_ps = chain.back()->output(0).get_partial_shape();
                    if (root_out_ps.is_static() && ov::shape_size(root_out_ps.to_shape()) <= 1)
                        continue;
                }

                // Skip chains whose PRIMARY input is a Constant. These are static
                // weight-quantization chains (small, e.g. [512]/[768]); the fused op's
                // kernel reads primary_in[i] from a device buffer, but a const-folded
                // Constant primary may not be materialised on device → fault. Low value
                // anyway (constant, could be folded at compile time). Activation-quant
                // chains (the real target) have a compute-node primary, not a Constant.
                if (ov::is_type<ov::op::v0::Constant>(root->get_input_node_shared_ptr(0)))
                    continue;

                // Guard the ROOT binary op's aux too (the loop above only checked added
                // links): root primary = input(0), aux = input(1). Allow aux that matches
                // the primary element count OR is a 1-element scalar (kernel broadcasts it).
                if (root->get_input_size() == 2) {
                    const auto& p_ps = root->input_value(0).get_partial_shape();
                    const auto& a_ps = root->input_value(1).get_partial_shape();
                    if (!p_ps.is_static() || !a_ps.is_static()) continue;
                    const size_t p_n = ov::shape_size(p_ps.to_shape());
                    const size_t a_n = ov::shape_size(a_ps.to_shape());
                    if (a_n != p_n && a_n != 1 && !is_last_dim_bcast(a_ps, p_ps)) continue;
                }

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

                // Absorb an i32->f32 Convert at the chain head (the post-MatMulInteger
                // dequant epilogue: Convert -> Mul(scale) -> Mul(out_scale) -> BiasAdd).
                // Rewire fused_inputs[0] to the Convert's i32 input and prepend a Cast
                // step; the kernel reads the i32 primary and converts on load, so the
                // Convert becomes free and the whole epilogue is one kernel.
                std::shared_ptr<ov::op::v0::Convert> head_convert;
                if (!std::getenv("FE_NO_CONVERT_HEAD")) {
                    head_convert = as_i32_to_f32_convert_head(root->get_input_node_shared_ptr(0));
                    if (head_convert) {
                        fused_inputs[0] = head_convert->input_value(0);
                        nodes::FusedEwStep cast_step;
                        cast_step.op_type = "Convert";
                        cast_step.param   = 0.f;
                        cast_step.has_aux = false;
                        steps.insert(steps.begin(), cast_step);
                    }
                }

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
