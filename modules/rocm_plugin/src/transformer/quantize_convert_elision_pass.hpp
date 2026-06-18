// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Elides the redundant u8 round-trip in the INT8 activation-quantize epilogue.
//
// After CommonOptimizations the quantize cone ends as:
//   ... -> Round -> Clamp(0,255) -> Convert(f32->u8) -> { Convert(u8->f32),
//                                                          Convert(u8->i32) }
// The Round+Clamp(0,255) already produce integer-valued f32 in [0,255], so the
// f32->u8 cast does not change the numeric value — it only narrows the storage
// type. The downstream Convert(u8->f32) is therefore a no-op (value unchanged),
// and Convert(u8->i32) is equivalent to Convert(f32->i32) straight from the
// Clamp output. The plugin's MatMul anyway casts its i32 operands to f16, so the
// u8 intermediate is never used as u8 — it is pure overhead (2-3 full-tensor
// kernels per quantize point).
//
// This pass, for each Convert(f32->u8) `q`:
//   * Convert(u8->f32) consumer  -> replaced by q's f32 input  (both Converts gone)
//   * Convert(u8->i32) consumer  -> rewired to read q's f32 input (u8 hop gone;
//                                    the i32 Convert stays, now f32->i32)
// If `q` ends up with no consumers it is removed by later DCE.
//
// Gated by ROCM_DISABLE_U8_ELISION for bisection.

#pragma once

#include <openvino/pass/pass.hpp>
#include <openvino/op/convert.hpp>

#include <cstdio>
#include <cstdlib>

namespace ov { namespace rocm_gpu { namespace pass {

class QuantizeConvertElisionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("QuantizeConvertElisionPass", "0");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override {
        if (std::getenv("ROCM_DISABLE_U8_ELISION")) return false;
        int elided_f32 = 0, rewired_i32 = 0;

        for (const auto& node : model->get_ordered_ops()) {
            auto q = std::dynamic_pointer_cast<ov::op::v0::Convert>(node);
            if (!q) continue;
            if (q->get_output_element_type(0) != ov::element::u8) continue;
            // The producer value (f32, already integer-valued in [0,255]).
            auto src = q->input_value(0);
            if (src.get_element_type() != ov::element::f32) continue;

            // Iterate a snapshot of consumers (we mutate the graph).
            auto consumers = q->output(0).get_target_inputs();
            std::vector<ov::Input<ov::Node>> cons_vec(consumers.begin(), consumers.end());
            for (auto& ti : cons_vec) {
                auto cvt = std::dynamic_pointer_cast<ov::op::v0::Convert>(
                    ti.get_node()->shared_from_this());
                if (!cvt) continue;
                const auto dst = cvt->get_output_element_type(0);
                if (dst == ov::element::f32) {
                    // u8->f32 is a no-op given the integer-valued f32 source.
                    cvt->output(0).replace(src);
                    ++elided_f32;
                } else if (dst == ov::element::i32) {
                    // Read the f32 source directly; this Convert becomes f32->i32.
                    cvt->input(0).replace_source_output(src);
                    ++rewired_i32;
                }
            }
        }

        if ((elided_f32 || rewired_i32) && std::getenv("ROCM_TRACE_U8_ELISION"))
            fprintf(stderr, "[QuantizeConvertElision] elided %d u8->f32, rewired %d u8->i32\n",
                    elided_f32, rewired_i32);
        return elided_f32 > 0 || rewired_i32 > 0;
    }
};

}}} // namespace ov::rocm_gpu::pass
