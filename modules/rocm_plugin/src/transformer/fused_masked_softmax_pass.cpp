// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Fuses: scores → [Add(mask)] → Softmax → FusedMaskedSoftmax

#include "fused_masked_softmax_pass.hpp"
#include "nodes/fused_masked_softmax_node.hpp"

#include <openvino/op/add.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/core/graph_util.hpp>

// Helper to get Softmax's reduction axis regardless of op version
static int64_t get_softmax_axis(const std::shared_ptr<ov::Node>& sm) {
    if (auto sm1 = std::dynamic_pointer_cast<ov::op::v1::Softmax>(sm))
        return sm1->get_axis();
    if (auto sm8 = std::dynamic_pointer_cast<ov::op::v8::Softmax>(sm))
        return sm8->get_axis();
    return -1;
}

namespace ov {
namespace rocm_gpu {
namespace pass {

namespace {

static bool try_fuse(const std::shared_ptr<ov::op::v1::Softmax>& softmax) {
    // Softmax input must be Add (masked attention scores)
    auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(
        softmax->get_input_node_shared_ptr(0));
    if (!add) return false;
    if (add->output(0).get_target_inputs().size() != 1) return false;

    // Add inputs: one is scores [1,heads,sq,sk], other is mask [1,1,sq,sk]
    auto in0 = add->get_input_node_shared_ptr(0);
    auto in1 = add->get_input_node_shared_ptr(1);

    // Determine which is scores (4D with full head dim) and which is mask (1 in head dim)
    auto ps0 = add->input(0).get_partial_shape();
    auto ps1 = add->input(1).get_partial_shape();

    if (!ps0.rank().is_static() || !ps1.rank().is_static()) return false;
    if (ps0.rank().get_length() != 4 || ps1.rank().get_length() != 4) return false;

    // scores has shape [1, heads, sq, sk], mask has [1, 1, sq, sk]
    ov::Output<ov::Node> scores_out, mask_out;
    int64_t heads = 0, sq = 0, sk = 0;

    auto try_pair = [&](const ov::PartialShape& ps_s, const ov::PartialShape& ps_m,
                        ov::Output<ov::Node> s, ov::Output<ov::Node> m) -> bool {
        if (!ps_s[1].is_static() || !ps_m[1].is_static()) return false;
        if (ps_m[1].get_length() != 1) return false;  // mask must have 1 head
        if (!ps_s[2].is_static() || !ps_s[3].is_static()) return false;
        heads = ps_s[1].get_length();
        sq    = ps_s[2].get_length();
        sk    = ps_s[3].get_length();
        scores_out = s;
        mask_out   = m;
        return true;
    };

    if (!try_pair(ps0, ps1, add->input_value(0), add->input_value(1)) &&
        !try_pair(ps1, ps0, add->input_value(1), add->input_value(0)))
        return false;

    auto fused = std::make_shared<nodes::FusedMaskedSoftmax>(
        scores_out, mask_out, heads, sq, sk);
    fused->set_friendly_name(softmax->get_friendly_name() + "/FusedMaskedSoftmax");
    ov::replace_node(softmax, fused);

    fprintf(stderr, "[FusedMaskedSoftmaxPass] Fused Add+Softmax: heads=%lld sq=%lld sk=%lld\n",
            (long long)heads, (long long)sq, (long long)sk);
    return true;
}

}  // namespace

bool FusedMaskedSoftmaxPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    // Collect both v1::Softmax and v8::Softmax nodes
    std::vector<std::shared_ptr<ov::Node>> softmaxes;
    int total_all = 0;
    for (const auto& node : model->get_ordered_ops()) {
        ++total_all;
        if (std::dynamic_pointer_cast<ov::op::v1::Softmax>(node) ||
            std::dynamic_pointer_cast<ov::op::v8::Softmax>(node))
            softmaxes.push_back(node);
    }
    fprintf(stderr, "[FusedMaskedSoftmaxPass] total=%d softmax=%zu\n",
            total_all, softmaxes.size());

    for (auto& sm_node : softmaxes) {
        if (sm_node->output(0).get_target_inputs().empty()) continue;
        // Wrap in try_fuse by converting to v1::Softmax or v8::Softmax
        try {
            // Check if input is an Add (mask add)
            auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(
                sm_node->get_input_node_shared_ptr(0));
            if (!add) {
                auto src = sm_node->get_input_node_shared_ptr(0);
                fprintf(stderr, "[FusedMaskedSoftmaxPass] Softmax input is %s (not Add)\n",
                        src->get_type_name());
                continue;
            }
            fprintf(stderr, "[FusedMaskedSoftmaxPass] Found Add before Softmax, shapes: %s, %s\n",
                    add->input(0).get_partial_shape().to_string().c_str(),
                    add->input(1).get_partial_shape().to_string().c_str());
            if (add->output(0).get_target_inputs().size() != 1) continue;

            auto ps0 = add->input(0).get_partial_shape();
            auto ps1 = add->input(1).get_partial_shape();
            if (!ps0.rank().is_static() || !ps1.rank().is_static()) continue;

            ov::Output<ov::Node> scores_out, mask_out;
            int64_t heads = 0, sq = 0, sk = 0;

            // Scores must be 4D [batch, heads, sq, sk]
            // Mask may be 4D [1,1,sq,sk] or 2D [sq,sk] (after squeeze)
            auto try_pair = [&](const ov::PartialShape& ps_s, const ov::PartialShape& ps_m,
                                ov::Output<ov::Node> s, ov::Output<ov::Node> m) -> bool {
                if (ps_s.rank().get_length() != 4) return false;
                if (!ps_s[1].is_static() || !ps_s[2].is_static() || !ps_s[3].is_static()) return false;
                heads = ps_s[1].get_length();
                sq    = ps_s[2].get_length();
                sk    = ps_s[3].get_length();
                // Mask: 2D [sq,sk] or 4D [1,1,sq,sk]
                if (ps_m.rank().get_length() == 2) {
                    if (!ps_m[0].is_static() || !ps_m[1].is_static()) return false;
                    if (ps_m[0].get_length() != sq || ps_m[1].get_length() != sk) return false;
                } else if (ps_m.rank().get_length() == 4) {
                    if (!ps_m[1].is_static() || ps_m[1].get_length() != 1) return false;
                } else return false;
                scores_out = s;
                mask_out   = m;
                return true;
            };

            if (!try_pair(ps0, ps1, add->input_value(0), add->input_value(1)) &&
                !try_pair(ps1, ps0, add->input_value(1), add->input_value(0)))
                continue;

            auto fused = std::make_shared<nodes::FusedMaskedSoftmax>(
                scores_out, mask_out, heads, sq, sk);
            fused->set_friendly_name(sm_node->get_friendly_name() + "/FusedMaskedSoftmax");
            ov::replace_node(sm_node, fused);

            fprintf(stderr, "[FusedMaskedSoftmaxPass] Fused Add+Softmax: heads=%lld sq=%lld sk=%lld\n",
                    (long long)heads, (long long)sq, (long long)sk);
            changed = true;
        } catch (...) {}
    }
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
