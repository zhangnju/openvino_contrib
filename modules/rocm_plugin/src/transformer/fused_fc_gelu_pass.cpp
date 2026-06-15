// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Fuses FC + GELU into FusedFCGELU:
//   (A) Explicit ov::op::v7/v0::Gelu after FullyConnected
//   (B) TF-style tanh GELU decomposition:
//       fc_out -> Pow(3) -> Mul(0.04472) -> Add(fc_out) -> Mul(0.79788) -> Tanh
//              -> Add(1.0) -> Mul(0.5) -> Mul(fc_out) -> output

#include "fused_fc_gelu_pass.hpp"
#include "nodes/fused_fc_gelu_node.hpp"
#include "nodes/fully_connected.hpp"

#include <openvino/op/gelu.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/tanh.hpp>
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

// Pierce through Convert/Identity wrappers
static std::shared_ptr<ov::Node> skip_cvt(std::shared_ptr<ov::Node> n) {
    for (int d = 0; d < 5 && n; ++d) {
        const std::string& t = n->get_type_info().name;
        if (t == "Convert" || t == "Identity") n = n->get_input_node_shared_ptr(0);
        else break;
    }
    return n;
}

// Check if node is a scalar Constant with approximately the given value
static bool approx_scalar(const std::shared_ptr<ov::Node>& n, float expected, float tol = 0.01f) {
    auto raw = skip_cvt(n);
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(raw);
    if (!c || ov::shape_size(c->get_shape()) != 1) return false;
    auto v = c->cast_vector<float>();
    return !v.empty() && std::abs(v[0] - expected) < tol;
}

// ── Helpers for explicit Gelu paths ──────────────────────────────────────────

static bool get_fc_dims(const std::shared_ptr<nodes::FullyConnected>& fc,
                        const std::shared_ptr<ov::Node>& output_node,
                        int64_t& seq_out, int64_t& in_dim_out, int64_t& out_dim_out) {
    auto x_shape  = fc->get_input_partial_shape(0);
    auto W_shape  = fc->get_input_partial_shape(1);
    auto out_ps   = output_node->get_output_partial_shape(0);

    if (!x_shape.rank().is_static() || !W_shape.rank().is_static()) return false;
    if (!out_ps.rank().is_static()) return false;

    int64_t rank_x = x_shape.rank().get_length();
    // Support rank 2 [seq, hidden] and rank 3 [batch, seq, hidden]
    if (rank_x < 2 || rank_x > 3) return false;

    int64_t rank_o  = out_ps.rank().get_length();
    int64_t out_dim = out_ps[rank_o-1].is_static() ? out_ps[rank_o-1].get_length() : 0;
    if (out_dim == 0) return false;

    int64_t rank_w = W_shape.rank().get_length();
    int64_t w0 = W_shape[rank_w-2].is_static() ? W_shape[rank_w-2].get_length() : 0;
    int64_t w1 = W_shape[rank_w-1].is_static() ? W_shape[rank_w-1].get_length() : 0;
    int64_t in_dim = (w1 == out_dim) ? w0 : w1;
    if (in_dim == 0) return false;

    // Compute M dimension: for rank 3 [batch, seq, hidden], M = batch * seq
    int64_t seq = 1;
    for (int64_t d = 0; d < rank_x - 1; ++d) {
        if (!x_shape[d].is_static()) return false;
        seq *= x_shape[d].get_length();
    }

    seq_out = seq; in_dim_out = in_dim; out_dim_out = out_dim;
    return true;
}

static bool try_fuse_explicit_gelu(const std::shared_ptr<ov::Node>& gelu_node,
                                    const std::string& label) {
    auto fc = std::dynamic_pointer_cast<nodes::FullyConnected>(
        gelu_node->get_input_node_shared_ptr(0));
    if (!fc || fc->get_input_size() < 3) return false;
    if (fc->output(0).get_target_inputs().size() != 1) return false;

    int64_t seq, in_dim, out_dim;
    if (!get_fc_dims(fc, gelu_node, seq, in_dim, out_dim)) return false;

    auto fused = std::make_shared<nodes::FusedFCGELU>(
        fc->input_value(0), fc->input_value(1), fc->input_value(2),
        seq, in_dim, out_dim);
    fused->set_friendly_name(gelu_node->get_friendly_name() + "/FusedFCGELU");
    ov::replace_node(gelu_node, fused);
    fprintf(stderr, "[FusedFCGELUPass] %s seq=%lld in=%lld out=%lld\n",
            label.c_str(), (long long)seq, (long long)in_dim, (long long)out_dim);
    return true;
}

// ── TF-style tanh GELU detection ─────────────────────────────────────────────
//
// Pattern (backward from Tanh):
//   Tanh <- Mul(~0.7979, Add(fc_out, Mul(~0.04472, Pow(fc_out, 3))))
//
// Pattern (forward from Tanh):
//   Tanh -> Add(~1.0) -> Mul(~0.5) -> Mul(fc_out) -> final_output
//
// fc_out is consumed in 3 places; all must trace to the same FullyConnected node.

static bool try_fuse_tf_tanh_gelu(const std::shared_ptr<ov::Node>& tanh_node) {
    if (tanh_node->get_input_size() != 1) return false;

    // ── Backward: Tanh <- Mul(sqrt2pi) <- Add <- Mul(cubic_coeff) <- Pow(fc_out, 3) ──

    auto mul_sqrt2pi = tanh_node->get_input_node_shared_ptr(0);
    if (mul_sqrt2pi->get_type_info().name != std::string("Multiply")) return false;
    if (mul_sqrt2pi->get_input_size() != 2) return false;

    // One Multiply input ~ 0.7979 (sqrt(2/pi)), the other is Add
    std::shared_ptr<ov::Node> add_cubic;
    bool has_sqrt2pi = false;
    for (int i = 0; i < 2; ++i) {
        auto inp = mul_sqrt2pi->get_input_node_shared_ptr(i);
        if (approx_scalar(inp, 0.7978846f)) {
            has_sqrt2pi = true;
        } else if (inp->get_type_info().name == std::string("Add")) {
            add_cubic = inp;
        }
    }
    if (!has_sqrt2pi || !add_cubic || add_cubic->get_input_size() != 2) return false;

    // Add inputs: fc_out and Mul(~0.04472, Pow(fc_out, 3))
    std::shared_ptr<ov::Node> fc_out_cand;
    std::shared_ptr<ov::Node> mul_cubic;
    for (int i = 0; i < 2; ++i) {
        auto inp = add_cubic->get_input_node_shared_ptr(i);
        const std::string& t = inp->get_type_info().name;
        if (t == "Multiply" && inp->get_input_size() == 2) {
            for (int j = 0; j < 2; ++j) {
                if (approx_scalar(inp->get_input_node_shared_ptr(j), 0.044715f, 0.001f)) {
                    mul_cubic = inp;
                    fc_out_cand = add_cubic->get_input_node_shared_ptr(1 - i);
                    break;
                }
            }
        }
        if (mul_cubic) break;
    }
    if (!mul_cubic || !fc_out_cand) return false;

    // Pow(fc_out, 3): find the Pow node inside mul_cubic
    std::shared_ptr<ov::Node> pow_node;
    for (int i = 0; i < 2; ++i) {
        auto inp = mul_cubic->get_input_node_shared_ptr(i);
        const std::string& t = inp->get_type_info().name;
        if (t == "Power" || t == "Pow") { pow_node = inp; break; }
    }
    if (!pow_node) return false;
    if (!approx_scalar(pow_node->get_input_node_shared_ptr(1), 3.0f)) return false;
    // Verify Pow's base == fc_out_cand
    if (skip_cvt(pow_node->get_input_node_shared_ptr(0)).get() != fc_out_cand.get()) return false;

    // ── Forward: Tanh -> Add(1.0) -> Mul(0.5) -> Mul(fc_out) ──────────────────

    auto add_one = single_consumer(tanh_node);
    if (!add_one || add_one->get_type_info().name != std::string("Add")) return false;
    bool has_one = false;
    for (int i = 0; i < 2; ++i) {
        if (approx_scalar(add_one->get_input_node_shared_ptr(i), 1.0f)) { has_one = true; break; }
    }
    if (!has_one) return false;

    auto mul_half = single_consumer(add_one);
    if (!mul_half || mul_half->get_type_info().name != std::string("Multiply")) return false;
    bool has_half = false;
    for (int i = 0; i < 2; ++i) {
        if (approx_scalar(mul_half->get_input_node_shared_ptr(i), 0.5f)) { has_half = true; break; }
    }
    if (!has_half) return false;

    auto final_mul = single_consumer(mul_half);
    if (!final_mul || final_mul->get_type_info().name != std::string("Multiply")) return false;

    // final_mul must have fc_out_cand as one input
    bool fc_in_final = false;
    for (int i = 0; i < 2; ++i) {
        if (skip_cvt(final_mul->get_input_node_shared_ptr(i)).get() == fc_out_cand.get()) {
            fc_in_final = true; break;
        }
    }
    if (!fc_in_final) return false;

    // ── fc_out_cand must be FullyConnected ────────────────────────────────────
    auto fc = std::dynamic_pointer_cast<nodes::FullyConnected>(fc_out_cand);
    if (!fc || fc->get_input_size() < 3) return false;

    int64_t seq, in_dim, out_dim;
    if (!get_fc_dims(fc, final_mul, seq, in_dim, out_dim)) return false;

    // ── Replace final_mul with FusedFCGELU ───────────────────────────────────
    // Intermediate nodes (Pow, mul_cubic, add_cubic, mul_sqrt2pi, Tanh, add_one, mul_half)
    // become dead and are cleaned up by downstream NopElimination.
    auto fused = std::make_shared<nodes::FusedFCGELU>(
        fc->input_value(0), fc->input_value(1), fc->input_value(2),
        seq, in_dim, out_dim);
    fused->set_friendly_name(fc->get_friendly_name() + "/FusedFCGELU_TF");
    ov::replace_node(final_mul, fused);

    fprintf(stderr, "[FusedFCGELUPass] TF-tanh GELU fused: seq=%lld in=%lld out=%lld\n",
            (long long)seq, (long long)in_dim, (long long)out_dim);
    return true;
}

}  // namespace

bool FusedFCGELUPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;

    std::vector<std::shared_ptr<ov::Node>> gelus;
    std::vector<std::shared_ptr<ov::Node>> tanhs;

    for (const auto& node : model->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ov::op::v7::Gelu>(node) ||
            std::dynamic_pointer_cast<ov::op::v0::Gelu>(node))
            gelus.push_back(node);
        else if (std::dynamic_pointer_cast<ov::op::v0::Tanh>(node))
            tanhs.push_back(node);
    }

    // Path A: explicit Gelu ops
    for (auto& g : gelus) {
        if (g->output(0).get_target_inputs().empty()) continue;
        try {
            if (std::dynamic_pointer_cast<ov::op::v7::Gelu>(g)) {
                if (try_fuse_explicit_gelu(g, "v7::Gelu")) changed = true;
            } else if (std::dynamic_pointer_cast<ov::op::v0::Gelu>(g)) {
                if (try_fuse_explicit_gelu(g, "v0::Gelu")) changed = true;
            }
        } catch (...) {}
    }

    // Path B: TF-style tanh GELU decomposition
    for (auto& t : tanhs) {
        if (t->output(0).get_target_inputs().empty()) continue;
        try { if (try_fuse_tf_tanh_gelu(t)) changed = true; } catch (...) {}
    }

    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
