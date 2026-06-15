// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// TF-style LayerNorm fusion for ROCm GPU plugin.
//
// TensorFlow-exported BERT uses a pre-folded algebraic form of LayerNorm:
//   y = x * W + B   where W = gamma/std,  B = beta - mean * gamma/std
//
// This pass recognizes the full pattern and replaces it with a single
// miopenLayerNormForward kernel call via the custom LayerNorm OV node.
//
// Matched pattern (after ConvertPrecision, before ElementwiseFusionPass):
//
//  x ──── ReduceMean(axis) ──── mean ──── [Convert] ──── Sub(x, mean_cvt) ────── sq_mul
//          │                                                                      │
//          │                     sq_mul = Mul(diff, diff)                         │
//          │                                    │                                 ▼
//          │                              ReduceMean(axis) ──── Add(eps) ──── Sqrt
//          │                                                                    │
//          │                                                               Divide(1,sqrt) = recip
//          │                                                                    │
//          │                                               gamma_c ──── Mul(recip, gamma_c) = W
//          │                                                                    │
//          │────────────────────────────── Mul(x, W) = xW                      │
//          │                                                                    │
//          └───────────────────────────────────── Mul(mean, W) = mW             │
//                                                                 │             │
//                                                    beta_c ──── Sub(beta_c, mW) = B
//                                                                 │
//                                           xW ──── Add(xW, B) ──── output
//
// NOTE: OV's CommonOptimizations (ReshapeSinkingMatMul, etc.) may eliminate
// the Reshape between embeddings and encoder layers.  When a Reshape from
// [batch, seq, hidden] → [batch*seq, hidden] is removed, the ReduceMean
// axes are NOT updated.  For instance, the ONNX may have axes=[1] on a 2D
// [seq, hidden] tensor (normalizing over hidden), but after reshape removal
// the input becomes 3D [batch, seq, hidden] and axes=[1] now points at
// the seq dimension instead of hidden.  We detect this by comparing the
// gamma constant size to x.shape[axis] and, when they differ, search for
// the correct axis that matches gamma.

#include "layer_norm_fusion.hpp"
#include "nodes/layer_norm_node.hpp"
#include "nodes/fused_layernorm_node.hpp"

#include <optional>
#include <openvino/op/constant.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/core/graph_util.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

namespace {

static std::vector<int64_t> get_axes(const std::shared_ptr<ov::op::v1::ReduceMean>& rm) {
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(rm->get_input_node_shared_ptr(1));
    if (!c) return {};
    const auto& et = c->get_element_type();
    if (et == ov::element::i64) return c->cast_vector<int64_t>();
    std::vector<int64_t> r;
    if (et == ov::element::i32) { for (auto v : c->cast_vector<int32_t>()) r.push_back(v); }
    else { for (auto v : c->cast_vector<float>()) r.push_back((int64_t)v); }
    return r;
}

static std::optional<float> get_scalar_const(const std::shared_ptr<ov::Node>& n) {
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(n);
    if (!c || ov::shape_size(c->get_shape()) != 1) return std::nullopt;
    const auto& et = c->get_element_type();
    if (et == ov::element::f32) return c->cast_vector<float>()[0];
    if (et == ov::element::f64) return (float)c->cast_vector<double>()[0];
    if (et == ov::element::f16) {
        auto v = c->cast_vector<float>(); // OV auto-converts f16→f32
        return v.empty() ? std::nullopt : std::optional<float>(v[0]);
    }
    return std::nullopt;
}

// Skip through passthrough nodes (Identity, Convert)
static std::shared_ptr<ov::Node> skip_pass(std::shared_ptr<ov::Node> n) {
    while (n && n->inputs().size() == 1) {
        const std::string& t = n->get_type_info().name;
        if (t == "Identity" || t == "Convert") n = n->get_input_node_shared_ptr(0);
        else break;
    }
    return n;
}

static std::shared_ptr<ov::Node> single_consumer(const std::shared_ptr<ov::Node>& n) {
    const auto ts = n->output(0).get_target_inputs();
    if (ts.size() != 1) return nullptr;
    return ts.begin()->get_node()->shared_from_this();
}

// Check if node (possibly wrapped in Convert) is a Constant of size inner_size
static bool is_vector_const(const std::shared_ptr<ov::Node>& n, size_t inner_size) {
    auto raw = skip_pass(n);  // pierce Convert/Identity
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(raw);
    if (!c) return false;
    return ov::shape_size(c->get_shape()) == inner_size;
}

// Return the total number of elements of a Constant node (piercing Convert/Identity).
// Returns 0 if not a constant.
static size_t const_element_count(const std::shared_ptr<ov::Node>& n) {
    auto raw = skip_pass(n);
    auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(raw);
    if (!c) return 0;
    return ov::shape_size(c->get_shape());
}

// Get the underlying Constant node, piercing Convert/Identity wrappers
static std::shared_ptr<ov::Node> unwrap_const(const std::shared_ptr<ov::Node>& n) {
    return skip_pass(n);
}

// Try to match TF-style pre-folded LayerNorm rooted at rm1 (the mean ReduceMean).
// Returns true if matched and replaces the output node with a LayerNorm node.
static bool try_fuse_tf_layernorm(const std::shared_ptr<ov::op::v1::ReduceMean>& rm1) {
    if (!rm1->get_keep_dims()) return false;
    auto axes1 = get_axes(rm1);
    if (axes1.size() != 1) return false;
    int64_t axis = axes1[0];

    // x = rm1's primary input
    auto x = rm1->input_value(0);
    auto rank = x.get_partial_shape().rank();
    if (!rank.is_static()) return false;
    int64_t ndim = rank.get_length();
    if (axis < 0) axis += ndim;

    // inner dimension (the normalized one)
    if (!x.get_partial_shape()[axis].is_static()) return false;
    size_t inner_size = x.get_partial_shape()[axis].get_length();

    // ── Find Sub(x, mean_cvt) ──────────────────────────────────────────────
    // rm1 → [Convert]* → rhs of Sub; x → [Convert]* → lhs of Sub
    std::shared_ptr<ov::op::v1::Subtract> sub_node;
    {
        std::vector<std::shared_ptr<ov::Node>> frontier = {rm1};
        for (int depth = 0; depth < 4 && !sub_node; ++depth) {
            std::vector<std::shared_ptr<ov::Node>> next;
            for (const auto& fn : frontier) {
                for (const auto& t : fn->output(0).get_target_inputs()) {
                    auto u = t.get_node()->shared_from_this();
                    if (auto s = std::dynamic_pointer_cast<ov::op::v1::Subtract>(u)) {
                        sub_node = s; break;
                    }
                    const std::string& un = u->get_type_info().name;
                    if (un == "Identity" || un == "Convert") next.push_back(u);
                }
                if (sub_node) break;
            }
            frontier = std::move(next);
        }
    }
    if (!sub_node) return false;

    // Verify Sub(x_cvt, mean_cvt): lhs traces back to x, rhs traces back to rm1.
    // Use skip_pass to cross Convert nodes inserted by ConvertPrecision.
    {
        auto s0 = skip_pass(sub_node->get_input_node_shared_ptr(0));
        auto s1 = skip_pass(sub_node->get_input_node_shared_ptr(1));
        // x.get_node() might itself be wrapped: compare after skip_pass
        auto x_raw = skip_pass(x.get_node()->shared_from_this());
        if (s0.get() != x_raw.get()) std::swap(s0, s1);
        if (s0.get() != x_raw.get()) return false;
        // rhs must trace to rm1 (or rm1 wrapped in Convert)
        auto s1_raw = skip_pass(s1);
        if (s1_raw.get() != rm1.get()) return false;
    }

    // ── sq_mul → rm2 → Add(eps) → Sqrt → recip ────────────────────────────
    std::shared_ptr<ov::op::v1::Multiply> sq_mul;
    for (const auto& t : sub_node->output(0).get_target_inputs()) {
        auto m = std::dynamic_pointer_cast<ov::op::v1::Multiply>(t.get_node()->shared_from_this());
        if (m && m->get_input_node_ptr(0) == sub_node.get() &&
                 m->get_input_node_ptr(1) == sub_node.get()) { sq_mul = m; break; }
    }
    if (!sq_mul) return false;

    auto rm2 = std::dynamic_pointer_cast<ov::op::v1::ReduceMean>(single_consumer(sq_mul));
    if (!rm2 || !rm2->get_keep_dims() || get_axes(rm2) != axes1) return false;

    auto eps_add = std::dynamic_pointer_cast<ov::op::v1::Add>(single_consumer(rm2));
    if (!eps_add) return false;
    float eps_val = 1e-5f;
    {
        auto a0 = eps_add->get_input_node_shared_ptr(0);
        auto a1 = eps_add->get_input_node_shared_ptr(1);
        auto eps = (a0.get() == rm2.get()) ? get_scalar_const(a1) : get_scalar_const(a0);
        if (!eps) return false;
        eps_val = *eps;
    }

    auto sqrt_node = std::dynamic_pointer_cast<ov::op::v0::Sqrt>(single_consumer(eps_add));
    if (!sqrt_node) return false;

    // recip = Divide(1, sqrt) or Reciprocal(sqrt) or Power(sqrt, -1)
    auto recip = single_consumer(sqrt_node);
    if (!recip) return false;
    const std::string& rname = recip->get_type_info().name;
    if (rname != "Divide" && rname != "Reciprocal" && rname != "Power") return false;
    // For Divide: verify numerator is constant 1
    if (rname == "Divide") {
        auto num = recip->get_input_node_shared_ptr(0);  // numerator
        auto one = get_scalar_const(num);
        if (!one || std::abs(*one - 1.0f) > 1e-4f) return false;
        // denominator must be sqrt_node
        if (recip->get_input_node_ptr(1) != sqrt_node.get()) return false;
    }

    // ── W = Multiply(recip, gamma_c) ───────────────────────────────────────
    // recip feeds into Mul(recip, [Convert(]gamma_const[)]) = W
    std::shared_ptr<ov::op::v1::Multiply> W_mul;
    std::shared_ptr<ov::Node> gamma_node;  // the gamma input (may be Convert-wrapped)
    for (const auto& t : recip->output(0).get_target_inputs()) {
        auto m = std::dynamic_pointer_cast<ov::op::v1::Multiply>(t.get_node()->shared_from_this());
        if (!m) continue;
        auto m0 = m->get_input_node_shared_ptr(0);
        auto m1 = m->get_input_node_shared_ptr(1);
        std::shared_ptr<ov::Node> gc_raw;
        if (m0.get() == recip.get()) gc_raw = m1;
        else if (m1.get() == recip.get()) gc_raw = m0;
        if (gc_raw && is_vector_const(gc_raw, inner_size)) {
            W_mul = m; gamma_node = gc_raw; break;
        }
    }

    // ── Axis fixup for reshape-eliminated graphs ───────────────────────────
    // OV's CommonOptimizations (ReshapeSinkingMatMul, etc.) may eliminate
    // Reshape ops that flattened [batch,seq,hidden] → [batch*seq,hidden].
    // When this happens, the ReduceMean axes are NOT updated, so axes=[1]
    // on the original 2D tensor now points at the wrong dimension on the
    // 3D tensor.  Detect this by checking gamma: if gamma doesn't match
    // inner_size but DOES match the last dim of x, fix the axis.
    if (!W_mul) {
        // Probe the gamma candidate from recip's consumer Multiply
        size_t gamma_size = 0;
        std::shared_ptr<ov::op::v1::Multiply> candidate_W_mul;
        std::shared_ptr<ov::Node> candidate_gamma;
        for (const auto& t : recip->output(0).get_target_inputs()) {
            auto m = std::dynamic_pointer_cast<ov::op::v1::Multiply>(t.get_node()->shared_from_this());
            if (!m) continue;
            auto m0 = m->get_input_node_shared_ptr(0);
            auto m1 = m->get_input_node_shared_ptr(1);
            std::shared_ptr<ov::Node> gc_raw;
            if (m0.get() == recip.get()) gc_raw = m1;
            else if (m1.get() == recip.get()) gc_raw = m0;
            if (gc_raw) {
                gamma_size = const_element_count(gc_raw);
                if (gamma_size > 0) {
                    candidate_W_mul = m;
                    candidate_gamma = gc_raw;
                    break;
                }
            }
        }
        // Check if gamma matches the last dimension of x (the hidden dim)
        if (gamma_size > 0 && ndim >= 2) {
            int64_t last_axis = ndim - 1;
            if (x.get_partial_shape()[last_axis].is_static() &&
                (size_t)x.get_partial_shape()[last_axis].get_length() == gamma_size &&
                gamma_size != inner_size) {
                // Axis was wrong due to reshape elimination.  Fix it.
                axis = last_axis;
                inner_size = gamma_size;
                // Also verify rm2 axes match (they should, since both
                // ReduceMean nodes had the same axes in the ONNX).
                W_mul = candidate_W_mul;
                gamma_node = candidate_gamma;
            }
        }
    }

    if (!W_mul || !gamma_node) return false;
    // gamma_c is the unwrapped Constant (used only for name in debug print)
    auto gamma_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(unwrap_const(gamma_node));

    // ── x_times_W = Multiply(x, W) ─────────────────────────────────────────
    std::shared_ptr<ov::op::v1::Multiply> xW_mul;
    for (const auto& t : W_mul->output(0).get_target_inputs()) {
        auto m = std::dynamic_pointer_cast<ov::op::v1::Multiply>(t.get_node()->shared_from_this());
        if (!m) continue;
        auto m0 = m->get_input_node_ptr(0);
        auto m1 = m->get_input_node_ptr(1);
        if ((m0 == x.get_node() && m1 == W_mul.get()) ||
            (m1 == x.get_node() && m0 == W_mul.get())) {
            xW_mul = m; break;
        }
    }
    if (!xW_mul) return false;

    // ── mean_times_W = Multiply(mean, W) ───────────────────────────────────
    std::shared_ptr<ov::op::v1::Multiply> mW_mul;
    for (const auto& t : W_mul->output(0).get_target_inputs()) {
        auto m = std::dynamic_pointer_cast<ov::op::v1::Multiply>(t.get_node()->shared_from_this());
        if (!m || m.get() == xW_mul.get()) continue;
        auto m0 = m->get_input_node_ptr(0);
        auto m1 = m->get_input_node_ptr(1);
        if ((skip_pass(m->get_input_node_shared_ptr(0)).get() == rm1.get() && m1 == W_mul.get()) ||
            (skip_pass(m->get_input_node_shared_ptr(1)).get() == rm1.get() && m0 == W_mul.get())) {
            mW_mul = m; break;
        }
    }
    if (!mW_mul) return false;

    // ── B = Sub(beta_c, mW) ────────────────────────────────────────────────
    auto B_sub = std::dynamic_pointer_cast<ov::op::v1::Subtract>(single_consumer(mW_mul));
    if (!B_sub) return false;
    auto B_in0 = B_sub->get_input_node_shared_ptr(0);
    auto B_in1 = B_sub->get_input_node_shared_ptr(1);
    std::shared_ptr<ov::Node> beta_node;
    if (B_in1.get() == mW_mul.get() && is_vector_const(B_in0, inner_size)) beta_node = B_in0;
    else if (B_in0.get() == mW_mul.get() && is_vector_const(B_in1, inner_size)) beta_node = B_in1;
    if (!beta_node) return false;
    auto beta_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(unwrap_const(beta_node));

    // ── output = Add(xW, B) ────────────────────────────────────────────────
    auto out_add = std::dynamic_pointer_cast<ov::op::v1::Add>(single_consumer(B_sub));
    if (!out_add) {
        // May be: Add(B, xW) – B side has single consumer but need to check both Add inputs
        // Try finding Add among xW_mul consumers too
        for (const auto& t : xW_mul->output(0).get_target_inputs()) {
            auto a = std::dynamic_pointer_cast<ov::op::v1::Add>(t.get_node()->shared_from_this());
            if (!a) continue;
            auto a0 = a->get_input_node_ptr(0);
            auto a1 = a->get_input_node_ptr(1);
            if ((a0 == xW_mul.get() && a1 == B_sub.get()) ||
                (a1 == xW_mul.get() && a0 == B_sub.get())) {
                out_add = a; break;
            }
        }
    }
    if (!out_add) return false;

    // ── All matched! Build LayerNorm (or FusedLayerNorm) replacement ──────────
    auto x_actual = sub_node->input_value(0);

    auto gamma_out = W_mul->get_input_node_ptr(0) == recip.get()
                     ? W_mul->input_value(1) : W_mul->input_value(0);
    auto beta_out  = B_sub->get_input_node_ptr(0) == mW_mul.get()
                     ? B_sub->input_value(1) : B_sub->input_value(0);

    // ── P1 optimisation: fuse 2-op Add chain + LayerNorm into single kernel ───
    // Detect: x_actual = Add(Op1_result, residual)
    //     where Op1 = Add(src, bias_const)  [2-op FusedElementwise chain root]
    // Create: FusedLayerNorm(src, bias_const, residual, gamma, beta)
    // This absorbs BOTH Adds, preventing the 1st Add from becoming a
    // standalone broadcasting_add kernel when the 2nd is absorbed alone.
    auto x_raw_node = skip_pass(x_actual.get_node_shared_ptr());
    bool did_fused_ln = false;

    if (x_raw_node &&
        x_raw_node->get_type_info().name == std::string("Add") &&
        x_raw_node->get_input_size() == 2) {

        // x_raw_node = Add(Op1_result, residual) -- second Add in chain
        // Try both orderings: Op1 could be input 0 or 1
        for (int swap = 0; swap < 2 && !did_fused_ln; ++swap) {
            auto op1_result_val = x_raw_node->input_value(swap);
            auto residual_val   = x_raw_node->input_value(1 - swap);
            auto op1_node = skip_pass(op1_result_val.get_node_shared_ptr());

            // Check if Op1 is also a plain Add (first Add in the chain)
            if (!op1_node ||
                op1_node->get_type_info().name != std::string("Add") ||
                op1_node->get_input_size() != 2)
                continue;

            auto src_val  = op1_node->input_value(0);
            auto bias_val = op1_node->input_value(1);

            // bias must be a vector constant of size == hidden
            size_t bias_sz = const_element_count(bias_val.get_node_shared_ptr());
            if (bias_sz == 0) {
                // Try swapping src and bias within Op1
                std::swap(src_val, bias_val);
                bias_sz = const_element_count(bias_val.get_node_shared_ptr());
                if (bias_sz == 0)
                    continue;
            }

            // Safety: x_raw_node (outer Add) consumers must all be TF-LN internal nodes
            bool safe_outer = true;
            for (const auto& inp : x_raw_node->output(0).get_target_inputs()) {
                const std::string& ct = inp.get_node()->get_type_info().name;
                if (ct != "ReduceMean" && ct != "Subtract" && ct != "Multiply" &&
                    ct != "Add" && ct != "Convert" && ct != "Identity")
                    safe_outer = false;
            }
            // Op1 (inner Add) must have exactly one live consumer (the outer Add)
            bool safe_inner = (op1_node->output(0).get_target_inputs().size() == 1);

            if (!safe_outer || !safe_inner)
                continue;

            if (!src_val.get_partial_shape().rank().is_static() ||
                !bias_val.get_partial_shape().rank().is_static() ||
                !residual_val.get_partial_shape().rank().is_static())
                continue;

            auto out_ps = out_add->get_output_partial_shape(0);
            if (!out_ps.rank().is_static()) continue;

            int64_t rank_o = out_ps.rank().get_length();
            int64_t last   = out_ps[rank_o-1].is_static() ? out_ps[rank_o-1].get_length() : 0;
            int64_t rows_f = 1;
            bool ok = (last > 0);
            for (int64_t d = 0; d < rank_o-1 && ok; ++d) {
                if (!out_ps[d].is_static()) { ok = false; break; }
                rows_f *= out_ps[d].get_length();
            }
            if (!ok) continue;

            // Verify bias size matches hidden dim
            if ((int64_t)bias_sz != last) continue;

            // 3-input FusedLayerNorm: computes LN(src + bias + residual)
            auto fused_node = std::make_shared<nodes::FusedLayerNorm>(
                src_val, bias_val, residual_val, gamma_out, beta_out,
                rows_f, last);
            fused_node->set_friendly_name(rm1->get_friendly_name() + "/FusedLN_3input");
            ov::replace_node(out_add, fused_node);

            // Disconnect dead TF-LN chain nodes from x_raw_node to prevent
            // ElementwiseFusionPass from forming chains through dead internals
            auto* x_ptr = x_raw_node.get();
            auto disc = [&](const std::shared_ptr<ov::Node>& n) {
                for (size_t i = 0; i < n->get_input_size(); ++i) {
                    if (skip_pass(n->get_input_node_shared_ptr(i)).get() == x_ptr) {
                        auto et = n->get_input_element_type(i);
                        auto dummy = ov::op::v0::Constant::create(et, ov::Shape{1}, {0.f});
                        try { n->input(i).replace_source_output(dummy->output(0)); } catch (...) {}
                        break;
                    }
                }
            };
            disc(rm1); disc(sub_node); disc(xW_mul);

            fprintf(stderr,
                "[LayerNormFusion] 3-input FusedLN (src+bias+residual): rows=%lld hidden=%lld\n",
                (long long)rows_f, (long long)last);
            did_fused_ln = true;
        }
    }

    if (!did_fused_ln) {
        // Fallback: plain intermediate LayerNorm (upgraded by FusedLayerNormPass)
        auto ln_node = std::make_shared<nodes::LayerNorm>(
            x_actual, gamma_out, beta_out, (double)eps_val, axis);
        ln_node->set_friendly_name(rm1->get_friendly_name() + "/TFLayerNorm");
        ov::replace_node(out_add, ln_node);
        fprintf(stderr, "[LayerNormFusion] TF-style fused at axis=%lld eps=%g inner=%zu\n",
                (long long)axis, (double)eps_val, inner_size);
    }
    return true;
}

}  // namespace

bool LayerNormFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    // Collect ReduceMean nodes (avoid modifying during iteration)
    std::vector<std::shared_ptr<ov::op::v1::ReduceMean>> rms;
    for (const auto& node : model->get_ordered_ops())
        if (auto rm = std::dynamic_pointer_cast<ov::op::v1::ReduceMean>(node))
            rms.push_back(rm);

    for (auto& rm : rms) {
        if (rm->output(0).get_target_inputs().empty()) continue;
        try { if (try_fuse_tf_layernorm(rm)) changed = true; } catch (...) {}
    }
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
