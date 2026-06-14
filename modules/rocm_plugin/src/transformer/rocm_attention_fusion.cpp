// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rocm_attention_fusion.hpp"
#include "ops/rocm_attention_matmul.hpp"
#include "ops/rocm_attention_softmax.hpp"
#include "transformer/nodes/fused_convolution.hpp"
#include <openvino/op/matmul.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/add.hpp>
#include <cstdlib>
#include <cmath>

namespace ov::rocm_gpu::pass {

bool RocmAttentionFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (!ov::rocm_gpu::RocmAttentionMatMulOp::isEnabled()) return false;

    bool changed = false;
    int found = 0;

    // After OV's TransposeMatMulTransformation, the attention graph looks like:
    //
    // QKV Conv [1,768,20,20]
    //   → Reshape [1,6,128,400]
    //   → VariadicSplit(axis=2) → Q[1,6,32,400] + K[1,6,32,400] + V[1,6,64,400]
    //
    // QKT MatMul:  input[0]=Q(from VariadicSplit), input[1]=Multiply(K, scale)
    //              transpose_a=true (OV fused the Transpose into MatMul attr)
    //              output [1,6,400,400]
    //
    // AV  MatMul:  input[0]=V(from VariadicSplit), input[1]=Softmax(...)
    //              output [1,6,64,400]

    for (const auto& op : model->get_ordered_ops()) {
        auto mm = std::dynamic_pointer_cast<ov::op::v0::MatMul>(op);
        if (!mm) continue;
        if (mm->get_rt_info().count("rocm_attn_kind")) continue;

        // Output must be [B, nh, N, N] (Q×K^T attention score, square)
        auto out_shape = mm->get_output_partial_shape(0);
        if (out_shape.is_dynamic() || out_shape.rank().get_length() != 4) continue;
        if (!out_shape[2].is_static() || !out_shape[3].is_static()) continue;
        if (out_shape[2] != out_shape[3]) continue;
        const int nh = static_cast<int>(out_shape[1].is_static() ? out_shape[1].get_length() : 0);
        const int N  = static_cast<int>(out_shape[2].get_length());
        if (nh <= 0) continue;

        // input[0] must come from VariadicSplit (Q after OV fused Transpose into MatMul)
        auto inp0_node = mm->input(0).get_source_output().get_node_shared_ptr();
        auto vsplit = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(inp0_node);
        if (!vsplit) continue;

        // VariadicSplit on axis=2 with 3 outputs [dq, dq, dv]
        auto axis_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            vsplit->input(1).get_source_output().get_node_shared_ptr());
        if (!axis_c) continue;
        std::vector<int64_t> axis_v;
        { const auto& _et=axis_c->get_element_type();
          if(_et==ov::element::i64) axis_v=axis_c->cast_vector<int64_t>();
          else if(_et==ov::element::i32) for(auto _v:axis_c->cast_vector<int32_t>()) axis_v.push_back(_v);
          else for(auto _v:axis_c->cast_vector<float>()) axis_v.push_back(static_cast<int64_t>(_v)); }
        if (axis_v.empty() || axis_v[0] != 2) continue;

        auto lens_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            vsplit->input(2).get_source_output().get_node_shared_ptr());
        if (!lens_c) continue;
        std::vector<int64_t> lens;
        { const auto& _et=lens_c->get_element_type();
          if(_et==ov::element::i64) lens=lens_c->cast_vector<int64_t>();
          else if(_et==ov::element::i32) for(auto _v:lens_c->cast_vector<int32_t>()) lens.push_back(_v);
          else for(auto _v:lens_c->cast_vector<float>()) lens.push_back(static_cast<int64_t>(_v)); }
        if (lens.size() != 3 || lens[0] != lens[1]) continue;
        const int dim_q = static_cast<int>(lens[0]);
        const int dim_v = static_cast<int>(lens[2]);

        // input[1] must come from Multiply (K × scale factor)
        auto inp1_node = mm->input(1).get_source_output().get_node_shared_ptr();
        if (!std::dynamic_pointer_cast<ov::op::v1::Multiply>(inp1_node)) continue;

        // Get pre-Reshape QKV tensor
        auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
            vsplit->input(0).get_source_output().get_node_shared_ptr());
        if (!reshape) continue;

        auto qkv_out = reshape->input(0).get_source_output();
        const auto qkv_shape = qkv_out.get_partial_shape();
        if (qkv_shape.is_dynamic() || qkv_shape.rank().get_length() != 4) continue;
        const int H = static_cast<int>(qkv_shape[2].is_static() ? qkv_shape[2].get_length() : 0);
        const int W = static_cast<int>(qkv_shape[3].is_static() ? qkv_shape[3].get_length() : 0);
        if (H <= 0 || W <= 0 || H * W != N) continue;

        // Tag the QKT MatMul for fusion
        mm->get_rt_info()["rocm_attn_kind"]   = std::string("qkt");
        mm->get_rt_info()["rocm_attn_nheads"] = std::to_string(nh);
        mm->get_rt_info()["rocm_attn_dim_q"]  = std::to_string(dim_q);
        mm->get_rt_info()["rocm_attn_dim_v"]  = std::to_string(dim_v);
        mm->get_rt_info()["rocm_attn_H"]      = std::to_string(H);
        mm->get_rt_info()["rocm_attn_W"]      = std::to_string(W);
        mm->get_rt_info()["rocm_attn_qkv_name"] = qkv_out.get_node_shared_ptr()->get_name();
        found++;
        changed = true;

        // Find downstream Softmax and AV MatMul via BFS: QKT → Mul(scale) → Softmax → AV MatMul
        // Also tag Softmax nodes for RocmAttentionSoftmaxOp fusion (fused scale+softmax kernel).
        std::vector<std::shared_ptr<ov::Node>> frontier = {mm->shared_from_this()};
        for (int step = 0; step < 10 && !frontier.empty(); ++step) {
            std::vector<std::shared_ptr<ov::Node>> next_frontier;
            for (auto& cur : frontier) {
                for (size_t oi = 0; oi < cur->get_output_size(); ++oi) {
                    for (const auto& tgt : cur->output(oi).get_target_inputs()) {
                        auto next = tgt.get_node()->shared_from_this();

                        // Tag Softmax nodes for fused scale+softmax kernel
                        auto sf = std::dynamic_pointer_cast<ov::op::v8::Softmax>(next);
                        if (!sf) sf = std::dynamic_pointer_cast<ov::op::v8::Softmax>(next);  // same type
                        // Also check v1::Softmax
                        bool is_softmax = (std::dynamic_pointer_cast<ov::op::v8::Softmax>(next) != nullptr ||
                                           std::dynamic_pointer_cast<ov::op::v1::Softmax>(next) != nullptr);
                        if (is_softmax && !next->get_rt_info().count("rocm_attn_softmax")
                            && RocmAttentionSoftmaxOp::isEnabled()) {
                            // Find scale from upstream Mul: QKT_out → Mul(scale_const) → this Softmax
                            float scale = 1.0f / std::sqrt(static_cast<float>(dim_q));
                            auto mul_inp = next->input(0).get_source_output().get_node_shared_ptr();
                            if (std::dynamic_pointer_cast<ov::op::v1::Multiply>(mul_inp)) {
                                // Try to extract scale from Multiply's constant input
                                for (size_t mi = 0; mi < mul_inp->get_input_size(); ++mi) {
                                    auto sc_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                                        mul_inp->input(mi).get_source_output().get_node_shared_ptr());
                                    if (sc_c) {
                                        auto sv = sc_c->cast_vector<float>();
                                        if (!sv.empty()) { scale = sv[0]; break; }
                                    }
                                }
                            }
                            next->get_rt_info()["rocm_attn_softmax"] = std::string("1");
                            next->get_rt_info()["rocm_attn_nheads"]  = std::to_string(nh);
                            next->get_rt_info()["rocm_attn_seq"]     = std::to_string(N);
                            next->get_rt_info()["rocm_attn_scale"]   = std::to_string(scale);
                            found++;
                        }

                        // Tag AV MatMul
                        auto av = std::dynamic_pointer_cast<ov::op::v0::MatMul>(next);
                        if (av && !av->get_rt_info().count("rocm_attn_kind")) {
                            auto av_inp0 = av->input(0).get_source_output().get_node_shared_ptr();
                            // Walk up through Transpose/Reshape to find VariadicSplit
                            auto vs_check = av_inp0;
                            for (int wu = 0; wu < 3; ++wu) {
                                if (std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(vs_check)) break;
                                if (vs_check->get_input_size() > 0)
                                    vs_check = vs_check->input(0).get_source_output().get_node_shared_ptr();
                                else break;
                            }
                            if (std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(vs_check)) {
                                av->get_rt_info()["rocm_attn_kind"]     = std::string("av");
                                av->get_rt_info()["rocm_attn_nheads"]   = std::to_string(nh);
                                av->get_rt_info()["rocm_attn_dim_q"]    = std::to_string(dim_q);
                                av->get_rt_info()["rocm_attn_dim_v"]    = std::to_string(dim_v);
                                av->get_rt_info()["rocm_attn_H"]        = std::to_string(H);
                                av->get_rt_info()["rocm_attn_W"]        = std::to_string(W);
                                av->get_rt_info()["rocm_attn_qkv_name"] = qkv_out.get_node_shared_ptr()->get_name();
                                found++;

                                // Look for pe(V) pattern downstream of AV:
                                // AV → ... → Add(pe_out, av_out)
                                // where pe = FusedGroupConvolution(V_from_split, filter, bias)
                                // Pattern: AV_output → [Transpose/Reshape*] → Add node
                                //          V_from_split → FusedGroupConv → [Reshape*] → Add
                                // We tag the Add node with pe info so the factory can fuse it.
                                {
                                    // BFS from AV to find pe pattern in two forms:
                                    // Form A (gfx950): FusedGroupConvolution → [Reshape] → Add(av_output)
                                    // Form B (gfx1201): 4-input FusedGroupConvolution(V, filter, bias, skip=av_output)
                                    //                   where FuseGroupConvolutionWithBiasAddAdd absorbed the Add.
                                    const char* fpe = std::getenv("ROCM_FUSE_PE");
                                    // pe fusion: enabled only if pe_fusion_ flag is set (gfx1201)
                                    // AND env var ROCM_FUSE_PE != "0"
                                    const bool pe_enabled_bfs = pe_fusion_ && (!fpe || std::string(fpe) != "0");

                                    const char* dbg_pe = std::getenv("ROCM_DEBUG_PE");
                                    std::vector<std::shared_ptr<ov::Node>> av_frontier = {av->shared_from_this()};
                                    for (int s2 = 0; s2 < 12 && !av_frontier.empty(); ++s2) {
                                        std::vector<std::shared_ptr<ov::Node>> av_next;
                                        for (auto& cn : av_frontier) {
                                            for (size_t oi2 = 0; oi2 < cn->get_output_size(); ++oi2) {
                                                for (const auto& t2 : cn->output(oi2).get_target_inputs()) {
                                                    auto n2 = t2.get_node()->shared_from_this();
                                                    if (dbg_pe) std::cerr << "[AttnFusion][BFS s=" << s2 << "] " << n2->get_type_name() << " nin=" << n2->get_input_size() << " name=" << n2->get_friendly_name() << "\n";

                                                    // ── Form B: 4-input FusedGroupConvolution with skip=attn ──────────
                                                    // FuseGroupConvolutionWithBiasAddAdd already fused the pe+add into
                                                    // a 4-input FGC: (V, filter, bias, skip=attn_output) → output
                                                    // Detected when: FGC has 4 inputs AND input[3] (skip) comes from AV path
                                                    {
                                                        auto fgc4 = std::dynamic_pointer_cast<nodes::FusedGroupConvolution>(n2);
                                                        if (fgc4 && fgc4->get_input_size() == 4 &&
                                                            !fgc4->get_rt_info().count("rocm_attn_pe_conv")) {
                                                            // input[0] = V (from VariadicSplit, possibly through Reshape)
                                                            // Verify it comes from the SAME VariadicSplit as AV's V input
                                                            auto v_src = fgc4->input(0).get_source_output().get_node_shared_ptr();
                                                            // Walk up through Reshape/Transpose to find VariadicSplit
                                                            for (int wu = 0; wu < 4; ++wu) {
                                                                if (std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(v_src)) break;
                                                                if (v_src->get_input_size() > 0)
                                                                    v_src = v_src->input(0).get_source_output().get_node_shared_ptr();
                                                                else break;
                                                            }
                                                            // Check that this VariadicSplit is the same as AV's QKV split
                                                            bool is_same_vs = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(v_src) &&
                                                                              (v_src.get() == vs_check.get());
                                                            if (is_same_vs) {
                                                                if (pe_enabled_bfs) {
                                                                    av->get_rt_info()["rocm_attn_pe_add"]       = std::string("1");
                                                                    av->get_rt_info()["rocm_attn_pe_conv_name"] = fgc4->get_name();
                                                                    // Form B: FGC IS the pe+add node — make it a noop
                                                                    // The pe fused kernel computes pe(V)+bias+attn → output
                                                                    // FGC output = pe fused kernel output
                                                                    fgc4->get_rt_info()["rocm_attn_pe_conv"]   = std::string("1");
                                                                    fgc4->get_rt_info()["rocm_pe_4input"]      = std::string("1");
                                                                    // Alias FGC output to its skip input (= AV output = pe+attn result)
                                                                    // input[3] is the skip (attn_output from AV), already contains pe+attn
                                                                    // after pe fused kernel runs. NoOpConvOp does nothing, so
                                                                    // downstream consumers read from the pe+attn buffer via aliasing.
                                                                    // Use rocm_pe_4input_skip to mark: output ← input[3]
                                                                    found++;
                                                                }
                                                                std::cerr << "[AttnFusion] Found pe(V) 4-input FGC: "
                                                                          << fgc4->get_friendly_name() << "\n";
                                                                goto av_found;
                                                            }
                                                        }
                                                    }

                                                    // ── Form A: FusedGroupConvolution → [Reshape] → Add ──────────────
                                                    // ── Form A: disabled ──────────────────────────────────────────────
                                                    // Form A (FGC → Add) is disabled: on gfx950, ElementwiseFusionPass
                                                    // absorbs pe_Add into FusedElementwise, making pe_FGC(NoOpConvOp)
                                                    // provide an uninitialized buffer to FusedElementwise → GPU fault.
                                                    // pe(V) fusion is only supported via Form B (4-input FGC) on gfx1201.
                                                    av_next.push_back(n2);
                                                }
                                            }
                                        }
                                        av_frontier = std::move(av_next);
                                    }
                                }
                                goto av_found;
                            }
                        }
                        next_frontier.push_back(next);
                    }
                }
            }
            frontier = std::move(next_frontier);
        }
        av_found:;
    }

    std::cerr << "[AttnFusion] Tagged " << found << " MatMul nodes for RocmAttentionMatMulOp\n";
    return changed;
}

}  // namespace ov::rocm_gpu::pass
