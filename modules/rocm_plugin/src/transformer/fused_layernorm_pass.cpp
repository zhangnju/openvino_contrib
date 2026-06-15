// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Converts nodes::LayerNorm → nodes::FusedLayerNorm (native f16 HIP kernel).
// Plain pattern only: LayerNorm(x, gamma, beta) → FusedLayerNorm(x, gamma, beta)
//
// Residual fusion (absorbing Add(fc, skip) into FusedLayerNorm with has_residual=true)
// was extensively investigated but always regresses performance:
// - Without disconnect_from_x: x has 4 consumers (rm1,sub,xW_mul,LN), safe_to_absorb fails
// - With disconnect_from_x: x has 1 consumer (LN), BUT disconnect breaks ElementwiseFusionPass
//   chain formation → 26 FusedElementwise → 24 broadcasting Add (slower)
// - Smem kernel cache: doesn't help because L2 cache already serves LN pass 2 reads
//
// Root cause: MIGraphX fuse_pointwise_reduce is a JIT compilation technique that
// keeps x+skip in REGISTERS across the entire add+reduce pipeline. Pre-compiled
// OV kernels cannot do this - there will always be an intermediate memory boundary.
// The 15% gap to MIGraphX (588 vs 700 FPS) is a fundamental pre-compiled vs JIT
// architecture difference, not a graph transformation problem.

#include "fused_layernorm_pass.hpp"
#include "nodes/fused_layernorm_node.hpp"
#include "nodes/layer_norm_node.hpp"
#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>

namespace ov { namespace rocm_gpu { namespace pass {
namespace {

static bool try_fuse_ln(const std::shared_ptr<nodes::LayerNorm>& ln) {
    auto out_shape = ln->get_output_partial_shape(0);
    if (!out_shape.rank().is_static()) return false;
    int64_t rank = out_shape.rank().get_length();
    int64_t axis = ln->get_axis();
    if (axis < 0) axis += rank;
    if (!out_shape[axis].is_static()) return false;
    int64_t hidden = out_shape[axis].get_length();
    int64_t rows = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (d == axis) continue;
        if (!out_shape[d].is_static()) return false;
        rows *= out_shape[d].get_length();
    }
    if (hidden == 0 || rows == 0 || ln->get_input_size() < 3) return false;
    auto x_val = ln->input_value(0);
    auto gamma = ln->input_value(1);
    auto beta  = ln->input_value(2);
    auto fused = std::make_shared<nodes::FusedLayerNorm>(x_val, gamma, beta, rows, hidden);
    fused->set_friendly_name(ln->get_friendly_name() + "/FusedLN");
    ov::replace_node(ln, fused);
    fprintf(stderr, "[FusedLNPass] Plain LN: rows=%lld hidden=%lld axis=%lld\n",
            (long long)rows, (long long)hidden, (long long)axis);
    return true;
}
} // namespace

bool FusedLayerNormPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    std::vector<std::shared_ptr<nodes::LayerNorm>> lnodes;
    for (const auto& node : model->get_ordered_ops())
        if (auto ln = std::dynamic_pointer_cast<nodes::LayerNorm>(node))
            lnodes.push_back(ln);
    for (auto& ln : lnodes) {
        if (ln->output(0).get_target_inputs().empty()) continue;
        try { if (try_fuse_ln(ln)) changed = true; } catch (...) {}
    }
    return changed;
}
}}} // namespaces
