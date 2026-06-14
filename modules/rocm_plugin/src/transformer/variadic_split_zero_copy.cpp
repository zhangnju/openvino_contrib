// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "variadic_split_zero_copy.hpp"

#include <cstdlib>
#include <openvino/op/variadic_split.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/core/rt_info.hpp>

#include "ops/variadic_split_alias.hpp"
#include <openvino/core/graph_util.hpp>

namespace ov::rocm_gpu::pass {

bool VariadicSplitZeroCopyPass::isEnabled() {
    const char* env = std::getenv("ROCM_ZEROCOPY_SPLIT");
    // Enabled by default; set ROCM_ZEROCOPY_SPLIT=0 to disable.
    if (env && std::string(env) == "0") return false;
    return true;
}

bool VariadicSplitZeroCopyPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (!isEnabled()) return false;

    bool changed = false;
    std::vector<std::shared_ptr<ov::op::v1::VariadicSplit>> candidates;

    for (const auto& op : model->get_ordered_ops()) {
        auto vsplit = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(op);
        if (!vsplit) continue;

        // Already converted
        if (std::dynamic_pointer_cast<nodes::VariadicSplitAlias>(op)) continue;

        // Check: axis == 1 (channel) on a 4D NCHW input
        const auto& in_pshape = vsplit->get_input_partial_shape(0);
        if (in_pshape.is_dynamic() || in_pshape.rank().get_length() != 4) continue;

        auto axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            vsplit->input(1).get_source_output().get_node_shared_ptr());
        if (!axis_const) continue;
        std::vector<int64_t> axis_data;
        { const auto& at=axis_const->get_element_type();
          if(at==ov::element::i64) axis_data=axis_const->cast_vector<int64_t>();
          else if(at==ov::element::i32) for(auto v:axis_const->cast_vector<int32_t>()) axis_data.push_back(v);
          else for(auto v:axis_const->cast_vector<float>()) axis_data.push_back(static_cast<int64_t>(v)); }
        if (axis_data.empty()) continue;
        // Support both positive and negative axis
        const int64_t rank = 4;
        int64_t axis = axis_data[0];
        if (axis < 0) axis += rank;
        if (axis != 1) continue;  // only channel-axis splits

        // Check: split lengths are all static and > 0
        auto lens_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            vsplit->input(2).get_source_output().get_node_shared_ptr());
        if (!lens_const) continue;
        std::vector<int64_t> lens;
        { const auto& et=lens_const->get_element_type();
          if(et==ov::element::i64) lens=lens_const->cast_vector<int64_t>();
          else if(et==ov::element::i32) for(auto v:lens_const->cast_vector<int32_t>()) lens.push_back(v);
          else for(auto v:lens_const->cast_vector<float>()) lens.push_back(static_cast<int64_t>(v)); }
        bool all_positive = true;
        for (auto l : lens) if (l <= 0) { all_positive = false; break; }
        if (!all_positive) continue;

        // Check: all outputs are static
        bool all_static = true;
        for (size_t i = 0; i < vsplit->get_output_size(); ++i) {
            if (vsplit->get_output_partial_shape(i).is_dynamic()) { all_static = false; break; }
        }
        if (!all_static) continue;

        candidates.push_back(vsplit);
    }

    std::cerr << "[ZeroCopySplit] Converting " << candidates.size()
              << " VariadicSplit(axis=1) nodes to alias (zero-copy)\n";

    for (auto& vsplit : candidates) {
        // Create replacement: VariadicSplitAlias with same inputs
        auto alias = std::make_shared<nodes::VariadicSplitAlias>(
            vsplit->input(0).get_source_output(),
            vsplit->input(1).get_source_output(),
            vsplit->input(2).get_source_output());

        alias->set_friendly_name(vsplit->get_friendly_name());
        ov::copy_runtime_info(vsplit, alias);

        ov::replace_node(vsplit, alias);
        changed = true;
    }

    return changed;
}

}  // namespace ov::rocm_gpu::pass
