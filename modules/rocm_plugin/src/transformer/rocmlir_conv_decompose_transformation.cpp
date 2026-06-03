// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rocmlir_conv_decompose_transformation.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/core/rt_info.hpp"

namespace ov::rocm_gpu::pass {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: check if this conv needs decomposition (non-1×1 kernel)
// ─────────────────────────────────────────────────────────────────────────────
static bool needs_decompose(const std::shared_ptr<ov::op::v1::Convolution>& conv) {
    // Only decompose when kernel > 1×1 (these are the unstable shapes on gfx950)
    const auto& filter_shape = conv->get_input_partial_shape(1);
    if (filter_shape.rank().is_dynamic()) return false;
    const auto rank = filter_shape.rank().get_length();
    if (rank < 4) return false;
    for (int i = 2; i < rank; ++i) {
        if (filter_shape[i].is_dynamic()) return false;
        if (filter_shape[i].get_length() > 1) return true;
    }
    return false;
}

static bool needs_decompose_group(const std::shared_ptr<ov::op::v1::GroupConvolution>& conv) {
    const auto& filter_shape = conv->get_input_partial_shape(1);
    if (filter_shape.rank().is_dynamic()) return false;
    const auto rank = filter_shape.rank().get_length();
    if (rank < 5) return false;  // GroupConv filter: [G, K/G, C/G, R, S]
    for (int i = 3; i < rank; ++i) {
        if (filter_shape[i].is_dynamic()) return false;
        if (filter_shape[i].get_length() > 1) return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build the decomposed subgraph for Convolution
// ─────────────────────────────────────────────────────────────────────────────
//
//  input [N, C, H, W]      filter [K, C, R, S]
//      │                       │
//  Tile(K copies)         Reshape([K*C, 1, R, S])
//  → [N, K*C, H, W]           │
//      │                       │
//  GroupConvolution(groups=K*C) → [N, K*C, OH, OW]
//      │
//  Reshape([N, K, C, OH, OW])
//      │
//  ReduceSum(axis=2, keepdims=False) → [N, K, OH, OW]

static std::shared_ptr<ov::Node> build_decomposed_conv(
        const std::shared_ptr<ov::op::v1::Convolution>& conv) {

    const auto input  = conv->input_value(0);
    const auto filter = conv->input_value(1);

    const auto& in_shape     = conv->get_input_shape(0);   // [N, C, H, W]
    const auto& filter_shape = conv->get_input_shape(1);   // [K, C, R, S]
    const int64_t N = in_shape[0];
    const int64_t C = in_shape[1];
    const int64_t H = in_shape[2];
    const int64_t W = in_shape[3];
    const int64_t K = filter_shape[0];
    const int64_t R = filter_shape[2];
    const int64_t S = filter_shape[3];
    const int64_t G = K * C;

    // ── Step 1: Reshape filter [K, C, R, S] → [K*C, 1, R, S] ────────────────
    auto filter_shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {4}, std::vector<int64_t>{G, 1, R, S});
    auto filter_reshaped = std::make_shared<ov::op::v1::Reshape>(
        filter, filter_shape_const, false);

    // ── Step 2: Tile input [N, C, H, W] → [N, K*C, H, W] ───────────────────
    // Tile along channel axis: repeat K times so that each group sees one (k, c) pair
    // Tile repeats = [1, K, 1, 1]
    auto tile_repeats = ov::op::v0::Constant::create(
        ov::element::i64, {4}, std::vector<int64_t>{1, K, 1, 1});
    auto input_tiled = std::make_shared<ov::op::v0::Tile>(input, tile_repeats);
    // input_tiled: [N, K*C, H, W]  (channel order: c0..cC-1, c0..cC-1, ..., K times)

    // ── Step 3: GroupConvolution with groups=K*C ────────────────────────────
    // filter_reshaped: [K*C, 1, R, S]  → need to reshape to GroupConv format [G, K/G, C/G, R, S]
    // GroupConv expects filter: [G, out_per_group, in_per_group, R, S]
    // Here G=K*C, out_per_group=1, in_per_group=1 → [K*C, 1, 1, R, S]
    auto gc_filter_shape = ov::op::v0::Constant::create(
        ov::element::i64, {5}, std::vector<int64_t>{G, 1, 1, R, S});
    auto gc_filter = std::make_shared<ov::op::v1::Reshape>(
        filter, gc_filter_shape, false);

    auto group_conv = std::make_shared<ov::op::v1::GroupConvolution>(
        input_tiled,
        gc_filter,
        conv->get_strides(),
        conv->get_pads_begin(),
        conv->get_pads_end(),
        conv->get_dilations(),
        conv->get_auto_pad());
    // Output: [N, K*C, OH, OW]

    // ── Step 4: Reshape [N, K*C, OH, OW] → [N, K, C, OH, OW] ──────────────
    const auto& out_shape = group_conv->get_output_shape(0); // [N, K*C, OH, OW]
    const int64_t OH = out_shape[2];
    const int64_t OW = out_shape[3];

    auto reshape_shape = ov::op::v0::Constant::create(
        ov::element::i64, {5}, std::vector<int64_t>{N, K, C, OH, OW});
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(
        group_conv, reshape_shape, false);

    // ── Step 5: ReduceSum along C axis (axis=2) → [N, K, OH, OW] ───────────
    auto reduce_axis = ov::op::v0::Constant::create(
        ov::element::i64, {1}, std::vector<int64_t>{2});
    auto result = std::make_shared<ov::op::v1::ReduceSum>(
        reshaped, reduce_axis, false /*keep_dims*/);

    result->set_friendly_name(conv->get_friendly_name());
    ov::copy_runtime_info(conv, result);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// RocMLIRConvDecompose — for ov::op::v1::Convolution
// ─────────────────────────────────────────────────────────────────────────────

RocMLIRConvDecompose::RocMLIRConvDecompose() {
    MATCHER_SCOPE(RocMLIRConvDecompose);

    auto conv_pattern = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>(
        ov::pass::pattern::has_static_shape());

    auto callback = [](ov::pass::pattern::Matcher& m) -> bool {
        auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(
            m.get_match_root());
        if (!conv || !needs_decompose(conv)) return false;

        if (!conv->get_input_partial_shape(0).is_static() ||
            !conv->get_input_partial_shape(1).is_static()) return false;

        // Skip asymmetric padding (handled by ConvolutionAsymPaddingTransformation)
        const auto& pb = conv->get_pads_begin();
        const auto& pe = conv->get_pads_end();
        for (size_t i = 0; i < pb.size(); ++i) {
            if (pb[i] != pe[i]) return false;
        }

        auto decomposed = build_decomposed_conv(conv);
        ov::replace_node(conv, decomposed);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        conv_pattern, matcher_name);
    register_matcher(m, callback);
}

// ─────────────────────────────────────────────────────────────────────────────
// RocMLIRGroupConvDecompose — for ov::op::v1::GroupConvolution
// ─────────────────────────────────────────────────────────────────────────────

RocMLIRGroupConvDecompose::RocMLIRGroupConvDecompose() {
    MATCHER_SCOPE(RocMLIRGroupConvDecompose);

    auto conv_pattern = ov::pass::pattern::wrap_type<ov::op::v1::GroupConvolution>(
        ov::pass::pattern::has_static_shape());

    auto callback = [](ov::pass::pattern::Matcher& m) -> bool {
        auto conv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(
            m.get_match_root());
        if (!conv || !needs_decompose_group(conv)) return false;
        if (!conv->get_input_partial_shape(0).is_static() ||
            !conv->get_input_partial_shape(1).is_static()) return false;

        const auto& filter_shape = conv->get_input_shape(1);  // [G, K/G, C/G, R, S]
        if (filter_shape[1] == 1 && filter_shape[2] == 1) return false;

        return false;  // Future: decompose non-trivial group conv
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        conv_pattern, matcher_name);
    register_matcher(m, callback);
}

} // namespace ov::rocm_gpu::pass
