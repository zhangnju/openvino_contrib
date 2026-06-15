// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "force_f16_matmul_pass.hpp"

#include <openvino/op/matmul.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/constant.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include "transformer/nodes/fully_connected.hpp"

namespace ov {
namespace rocm_gpu {
namespace pass {

bool ForceF16MatMulOutput::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    // Collect nodes first to avoid modifying during iteration
    std::vector<std::shared_ptr<ov::Node>> candidates;
    for (const auto& node : model->get_ordered_ops()) {
        const std::string& name = node->get_type_info().name;
        if (name != "MatMul" && name != "FullyConnected") continue;

        // Both inputs must be f16
        if (node->get_input_element_type(0) != ov::element::f16) continue;
        if (node->get_input_size() > 1 &&
            node->get_input_element_type(1) != ov::element::f16) continue;

        // Output must be f32 (set by ConvertPrecision for accumulation accuracy)
        if (node->get_output_element_type(0) != ov::element::f32) continue;

        candidates.push_back(node);
    }

    for (auto& node : candidates) {
        const std::string& type_name = node->get_type_info().name;

        std::shared_ptr<ov::Node> new_node;

        if (type_name == "MatMul") {
            auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
            if (!mm) continue;
            // TypeRelaxed<MatMul>: declare f16 inputs + f16 output,
            // rocBLAS will use its internal compute type (f16 with f32 accumulator)
            // but store the result as f16, eliminating the downstream Convert.
            new_node = std::make_shared<ov::op::TypeRelaxed<ov::op::v0::MatMul>>(
                *mm,
                ov::element::TypeVector{ov::element::f16, ov::element::f16},
                ov::element::TypeVector{ov::element::f16});
        } else {
            // FullyConnected (custom node): use TypeRelaxed wrapper
            auto fc = std::dynamic_pointer_cast<ov::rocm_gpu::nodes::FullyConnected>(node);
            if (!fc) continue;
            auto tr = std::make_shared<ov::op::TypeRelaxed<ov::rocm_gpu::nodes::FullyConnected>>(
                *fc,
                ov::element::TypeVector(node->get_input_size(), ov::element::f16),
                ov::element::TypeVector{ov::element::f16});
            new_node = tr;
        }

        if (!new_node) continue;
        new_node->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, new_node);
        ov::replace_node(node, new_node);

        // Downstream Convert(f32→f16) nodes now receive f16 input and output f16
        // → they become Convert(f16→f16) = no-op, eliminated by RemoveRedundantConvert
        fprintf(stderr, "[ForceF16MatMul] %s '%s': output f32→f16\n",
                type_name.c_str(), new_node->get_friendly_name().c_str());
        changed = true;
    }

    if (changed)
        fprintf(stderr, "[ForceF16MatMul] Converted %zu MatMul/FC outputs to f16\n",
                candidates.size());
    return changed;
}

// ─── EliminateF16ToF32Convert ─────────────────────────────────────────────────
// Remove Convert(f16→f32) nodes that widen f16 data unnecessarily.
// In f16 inference mode, these "decompression" converts cause downstream
// FusedElementwise chains to run the f32 kernel instead of the f16 kernel,
// doubling memory bandwidth and halving throughput.
//
// Pattern: Producer(f16 output) → Convert(f16→f32) → Consumer(sees f32 input)
// After:   Producer(f16 output) → Consumer(sees f16 input directly)
//
// This is safe because:
// 1. We're in f16 inference mode — all computation should use f16
// 2. The consumers (Add, Mul, etc.) support f16 natively via FusedElementwise
// 3. MIGraphX does the same: --fp16 mode eliminates all f16↔f32 converts

bool EliminateF16ToF32Convert::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    size_t converts_removed = 0;
    size_t constants_converted = 0;

    // Collect Convert(f16→f32) nodes: these widen f16 activations to f32,
    // causing downstream FusedElementwise to run f32 kernels instead of f16.
    // Source: OV's ConvertPrecision inserts these as "decompression" converts
    // after ops whose output it kept as f16 (e.g. after MatMul in BERT).
    std::vector<std::shared_ptr<ov::op::v0::Convert>> converts;
    for (const auto& node : model->get_ordered_ops()) {
        auto cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(node);
        if (!cvt) continue;
        if (cvt->get_input_element_type(0) == ov::element::f16 &&
            cvt->get_output_element_type(0) == ov::element::f32)
            converts.push_back(cvt);
    }

    // Step 1: For each Convert(f16→f32) we're about to remove, check its consumers.
    // If a consumer has OTHER inputs that are f32, convert them to f16 to avoid
    // type mismatches after bypass.
    // - Small constants (<=4096 elements): materialize a new f16 constant
    // - Large constants / non-constants: insert a Convert(f32→f16) node
    size_t converts_inserted = 0;
    for (auto& cvt : converts) {
        for (const auto& tgt : cvt->output(0).get_target_inputs()) {
            auto consumer = tgt.get_node()->shared_from_this();
            for (size_t i = 0; i < consumer->get_input_size(); ++i) {
                auto inp = consumer->input_value(i);
                if (inp.get_element_type() != ov::element::f32) continue;
                auto cst = std::dynamic_pointer_cast<ov::op::v0::Constant>(inp.get_node_shared_ptr());
                if (cst && ov::shape_size(cst->get_shape()) <= 4096) {
                    // Small constant: materialize f16 directly
                    auto f16_const = ov::op::v0::Constant::create(
                        ov::element::f16, cst->get_shape(),
                        std::vector<float>(cst->get_vector<float>()));
                    f16_const->set_friendly_name(cst->get_friendly_name());
                    ov::copy_runtime_info(cst, f16_const);
                    ov::replace_node(cst, f16_const);
                    constants_converted++;
                } else {
                    // Large constant or non-constant: insert Convert(f32→f16)
                    auto cvt_node = std::make_shared<ov::op::v0::Convert>(
                        inp, ov::element::f16);
                    consumer->input(i).replace_source_output(cvt_node->output(0));
                    converts_inserted++;
                }
                changed = true;
            }
        }
    }

    // Step 2: Bypass Convert(f16→f32) nodes — consumers now receive f16 directly.
    for (auto& cvt : converts) {
        auto f16_source = cvt->input_value(0);
        ov::replace_output_update_name(cvt->output(0), f16_source);
        converts_removed++;
        changed = true;
    }

    // Step 3: Fix remaining type mismatches iteratively.
    // After bypassing converts, modified nodes' outputs may change from f32→f16,
    // which propagates type changes downstream. Repeatedly scan and fix until stable.
    // We re-infer types each iteration so output types reflect current inputs.
    size_t mismatches_fixed = 0;
    if (changed) {
        bool found_mismatch = true;
        while (found_mismatch) {
            found_mismatch = false;
            // Re-infer types in topological order; skip nodes with mismatched inputs
            for (const auto& node : model->get_ordered_ops()) {
                try { node->revalidate_and_infer_types(); } catch (...) {}
            }
            for (const auto& node : model->get_ordered_ops()) {
                if (node->get_input_size() < 2) continue;
                bool has_f16 = false, has_f32 = false;
                for (size_t i = 0; i < node->get_input_size(); ++i) {
                    auto et = node->get_input_element_type(i);
                    if (et == ov::element::f16) has_f16 = true;
                    if (et == ov::element::f32) has_f32 = true;
                }
                if (!has_f16 || !has_f32) continue;
                for (size_t i = 0; i < node->get_input_size(); ++i) {
                    auto inp = node->input_value(i);
                    if (inp.get_element_type() != ov::element::f32) continue;
                    auto cst = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                        inp.get_node_shared_ptr());
                    if (cst && ov::shape_size(cst->get_shape()) <= 4096) {
                        auto f16_const = ov::op::v0::Constant::create(
                            ov::element::f16, cst->get_shape(),
                            std::vector<float>(cst->get_vector<float>()));
                        f16_const->set_friendly_name(cst->get_friendly_name());
                        ov::copy_runtime_info(cst, f16_const);
                        ov::replace_node(cst, f16_const);
                        constants_converted++;
                    } else {
                        auto cvt_fix = std::make_shared<ov::op::v0::Convert>(
                            inp, ov::element::f16);
                        node->input(i).replace_source_output(cvt_fix->output(0));
                        converts_inserted++;
                    }
                    mismatches_fixed++;
                    found_mismatch = true;
                }
            }
        }
    }

    if (changed)
        fprintf(stderr, "[EliminateF16ToF32] targeted: %zu small Constants f32→f16, "
                "%zu Convert(f32→f16) inserted, %zu Convert(f16→f32) bypassed, "
                "%zu type mismatches fixed\n",
                constants_converted, converts_inserted, converts_removed,
                mismatches_fixed);
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
