// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Fuses: FullyConnected(x, W, bias) → Gelu → FusedFCGELU(x, W, bias)

#include "fused_fc_gelu_pass.hpp"
#include "nodes/fused_fc_gelu_node.hpp"
#include "nodes/fully_connected.hpp"

#include <openvino/op/gelu.hpp>
#include <openvino/core/graph_util.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

namespace {

static std::shared_ptr<ov::Node> single_consumer(const std::shared_ptr<ov::Node>& n) {
    const auto ts = n->output(0).get_target_inputs();
    if (ts.size() != 1) return nullptr;
    return ts.begin()->get_node()->shared_from_this();
}

static bool try_fuse_fc_gelu(const std::shared_ptr<ov::op::v7::Gelu>& gelu) {
    // Gelu input must be FullyConnected (single consumer)
    auto fc = std::dynamic_pointer_cast<nodes::FullyConnected>(
        gelu->get_input_node_shared_ptr(0));
    if (!fc) return false;
    // FC must have only this Gelu as consumer
    if (fc->output(0).get_target_inputs().size() != 1) return false;

    // Get shapes
    auto x_shape  = fc->get_input_partial_shape(0);
    auto W_shape  = fc->get_input_partial_shape(1);
    auto out_shape = gelu->get_output_partial_shape(0);

    if (!x_shape.rank().is_static() || !W_shape.rank().is_static()) return false;
    // Require 2D x [seq, in_dim]
    if (x_shape.rank().get_length() > 2) return false;

    // Get dimensions from output shape (more reliable than parsing W with transpose flags)
    auto out_ps = gelu->get_output_partial_shape(0);
    if (!out_ps.rank().is_static()) return false;
    int64_t rank_o = out_ps.rank().get_length();
    if (!out_ps[rank_o-1].is_static() || !out_ps[rank_o-2].is_static()) return false;
    // out_shape = [seq, out_dim] for 2D or [1, seq, out_dim] for 3D
    int64_t out_dim = out_ps[rank_o-1].get_length();
    // in_dim from W (account for possible transpose)
    auto W_ps = W_shape;
    int64_t rank_w  = W_ps.rank().get_length();
    if (!W_ps[rank_w-1].is_static() || !W_ps[rank_w-2].is_static()) return false;
    // FullyConnected(x, W, bias): y = x × W (no transpose by default in OV plugin)
    // W shape is [in_dim, out_dim] → out matches fc output out_dim
    // Determine in_dim from W shape: the dim that is NOT out_dim
    int64_t w0 = W_ps[rank_w-2].get_length();
    int64_t w1 = W_ps[rank_w-1].get_length();
    int64_t in_dim = (w1 == out_dim) ? w0 : w1;

    // x shape: may be [seq, in_dim] or [1, seq, in_dim]
    int64_t rank_x = x_shape.rank().get_length();
    if (!x_shape[rank_x - 2].is_static()) return false;
    int64_t seq = x_shape[rank_x - 2].get_length();

    // FC inputs: [0]=x, [1]=W, [2]=bias
    if (fc->get_input_size() < 3) return false;
    auto x    = fc->input_value(0);
    auto W    = fc->input_value(1);
    auto bias = fc->input_value(2);

    // Create FusedFCGELU node
    auto fused = std::make_shared<nodes::FusedFCGELU>(x, W, bias, seq, in_dim, out_dim);
    fused->set_friendly_name(gelu->get_friendly_name() + "/FusedFCGELU");

    ov::replace_node(gelu, fused);

    fprintf(stderr, "[FusedFCGELUPass] Fused FC+GELU: seq=%lld in=%lld out=%lld\n",
            (long long)seq, (long long)in_dim, (long long)out_dim);
    return true;
}

static bool try_fuse_v0_gelu(const std::shared_ptr<ov::op::v0::Gelu>& gelu) {
    auto fc = std::dynamic_pointer_cast<nodes::FullyConnected>(
        gelu->get_input_node_shared_ptr(0));
    if (!fc || fc->output(0).get_target_inputs().size() != 1) return false;

    auto x_shape  = fc->get_input_partial_shape(0);
    auto W_shape  = fc->get_input_partial_shape(1);
    auto out_ps   = gelu->get_output_partial_shape(0);
    if (!x_shape.rank().is_static() || !W_shape.rank().is_static()) return false;
    if (!out_ps.rank().is_static()) return false;
    if (x_shape.rank().get_length() > 2) return false;

    int64_t rank_o  = out_ps.rank().get_length();
    int64_t out_dim = out_ps[rank_o-1].is_static() ? out_ps[rank_o-1].get_length() : 0;
    if (out_dim == 0) return false;

    int64_t rank_w = W_shape.rank().get_length();
    int64_t w0 = W_shape[rank_w-2].is_static() ? W_shape[rank_w-2].get_length() : 0;
    int64_t w1 = W_shape[rank_w-1].is_static() ? W_shape[rank_w-1].get_length() : 0;
    int64_t in_dim = (w1 == out_dim) ? w0 : w1;
    if (in_dim == 0) return false;

    int64_t rank_x = x_shape.rank().get_length();
    if (!x_shape[rank_x - 2].is_static()) return false;
    int64_t seq = x_shape[rank_x - 2].get_length();

    if (fc->get_input_size() < 3) return false;
    auto fused = std::make_shared<nodes::FusedFCGELU>(
        fc->input_value(0), fc->input_value(1), fc->input_value(2),
        seq, in_dim, out_dim);
    fused->set_friendly_name(gelu->get_friendly_name() + "/FusedFCGELU");
    ov::replace_node(gelu, fused);

    fprintf(stderr, "[FusedFCGELUPass] Fused FC+GELU(v0): seq=%lld in=%lld out=%lld\n",
            (long long)seq, (long long)in_dim, (long long)out_dim);
    return true;
}

}  // namespace

bool FusedFCGELUPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    // Collect Gelu nodes
    std::vector<std::shared_ptr<ov::Node>> gelus;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v7::Gelu>(node) ||
            std::dynamic_pointer_cast<ov::op::v0::Gelu>(node))
            gelus.push_back(node);
    }

    for (auto& g : gelus) {
        if (g->output(0).get_target_inputs().empty()) continue;
        try {
            if (auto g7 = std::dynamic_pointer_cast<ov::op::v7::Gelu>(g)) {
                if (try_fuse_fc_gelu(g7)) changed = true;
            } else if (auto g0 = std::dynamic_pointer_cast<ov::op::v0::Gelu>(g)) {
                if (try_fuse_v0_gelu(g0)) changed = true;
            }
        } catch (...) {}
    }
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
