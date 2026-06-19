// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "matmul_dequant_convert_pass.hpp"

#include <openvino/op/matmul.hpp>
#include <openvino/op/convert.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/core/graph_util.hpp>

#include <cstdio>
#include <cstdlib>

namespace ov {
namespace rocm_gpu {
namespace pass {

bool MatMulDequantConvertPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (std::getenv("ROCM_DISABLE_DEQUANT_CONVERT")) return false;

    // Collect candidates first (we mutate the graph while replacing).
    std::vector<std::pair<std::shared_ptr<ov::op::v0::MatMul>,
                          std::shared_ptr<ov::op::v0::Convert>>> candidates;
    for (const auto& node : model->get_ordered_ops()) {
        auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
        if (!mm) continue;
        // INT8 GEMM: operands are i32 (zero-point already subtracted upstream),
        // output is i32. The executor casts to f16/accumulates f32 regardless.
        if (mm->get_input_element_type(0) != ov::element::i32) continue;
        if (mm->get_output_element_type(0) != ov::element::i32) continue;

        // Sole consumer must be a Convert(i32->f32) — the dequant epilogue head.
        auto consumers = mm->output(0).get_target_inputs();
        if (consumers.size() != 1) continue;
        auto cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(
            consumers.begin()->get_node()->shared_from_this());
        if (!cvt) continue;
        if (cvt->get_output_element_type(0) != ov::element::f32) continue;

        candidates.emplace_back(mm, cvt);
    }

    int rewritten = 0;
    for (auto& [mm, cvt] : candidates) {
        // Retype the MatMul to f32 output (operands stay i32). The executor's
        // i32_out_is_f32_ path then writes the f32 GEMM result straight out and
        // skips cast_f32_to_i32; the Convert(i32->f32) is eliminated below.
        auto relaxed = std::make_shared<ov::op::TypeRelaxed<ov::op::v0::MatMul>>(
            *mm,
            ov::element::TypeVector{ov::element::i32, ov::element::i32},
            ov::element::TypeVector{ov::element::f32});
        relaxed->set_friendly_name(mm->get_friendly_name());
        ov::copy_runtime_info({mm, cvt}, relaxed);
        ov::replace_node(mm, relaxed);
        // The Convert's consumers now read the MatMul's f32 output directly.
        cvt->output(0).replace(relaxed->output(0));
        ++rewritten;
    }

    if (rewritten && std::getenv("ROCM_TRACE_DEQUANT_CONVERT"))
        fprintf(stderr, "[MatMulDequantConvert] retyped %d i32 MatMul outputs to f32 "
                "(eliminated f32->i32->f32 round-trip)\n", rewritten);
    return rewritten > 0;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
