// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// See header. After OV CommonOptimizations folds the QDQ chain, the surviving
// dynamic-quant cost is, per activation-quant point, TWO full-tensor reductions:
//   xmax = Maximum(0, ReduceMax(x));  xmin = Minimum(0, ReduceMin(x))
//   span = Subtract(xmax, xmin)
// We anchor on that Subtract and replace the {ReduceMax,Maximum,ReduceMin,Minimum,
// Subtract} sub-graph with a single nodes::DynamicQuantizeStats(x) that makes one
// pass over x. Downstream Multiply scaling that consumes `span` is left untouched.

#include "dynamic_quantize_fusion_pass.hpp"
#include "nodes/dynamic_quantize_node.hpp"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/constant.hpp>

#include <cstdio>
#include <vector>

namespace ov { namespace rocm_gpu { namespace pass {
namespace {

template <typename T>
std::shared_ptr<T> as(const ov::Output<ov::Node>& o) {
    return std::dynamic_pointer_cast<T>(o.get_node_shared_ptr());
}
template <typename T>
std::shared_ptr<T> as(const std::shared_ptr<ov::Node>& n) {
    return std::dynamic_pointer_cast<T>(n);
}

// Is `c` a scalar/single-element Constant equal to 0?
bool is_zero_const(const ov::Output<ov::Node>& o) {
    auto c = as<ov::op::v0::Constant>(o);
    if (!c) return false;
    if (ov::shape_size(c->get_shape()) != 1) return false;
    auto v = c->cast_vector<float>();
    return !v.empty() && v[0] == 0.0f;
}

// From a Maximum/Minimum(0, Reduce(x)) node, return the reduced source x if it
// matches Maximum(0, ReduceMax(x)) (want_max=true) or Minimum(0, ReduceMin(x)).
ov::Output<ov::Node> reduce_source(const std::shared_ptr<ov::Node>& mm, bool want_max, bool& ok) {
    ok = false;
    ov::Output<ov::Node> result;
    if (mm->get_input_size() != 2) return result;
    std::shared_ptr<ov::Node> reduce;
    bool has_zero = false;
    for (int i = 0; i < 2; ++i) {
        auto in = mm->input_value(i);
        if (is_zero_const(in)) { has_zero = true; continue; }
        reduce = in.get_node_shared_ptr();
    }
    if (!has_zero || !reduce) return result;
    bool rmatch = want_max ? (bool)as<ov::op::v1::ReduceMax>(reduce)
                           : (bool)as<ov::op::v1::ReduceMin>(reduce);
    if (!rmatch) return result;
    result = reduce->input_value(0);
    ok = true;
    return result;
}

} // namespace

bool DynamicQuantizeFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int fused = 0;

    std::vector<std::shared_ptr<ov::op::v1::Subtract>> subs;
    for (const auto& node : model->get_ordered_ops())
        if (auto s = as<ov::op::v1::Subtract>(node))
            subs.push_back(s);

    for (auto& sub : subs) {
        if (sub->output(0).get_target_inputs().empty()) continue;
        auto maxn = as<ov::op::v1::Maximum>(sub->input_value(0));
        auto minn = as<ov::op::v1::Minimum>(sub->input_value(1));
        if (!maxn || !minn) continue;

        bool ok_max = false, ok_min = false;
        auto xmax_src = reduce_source(maxn, true, ok_max);
        auto xmin_src = reduce_source(minn, false, ok_min);
        if (!ok_max || !ok_min) continue;
        // Both reductions must read the SAME activation tensor.
        if (xmax_src != xmin_src) continue;

        auto stats = std::make_shared<nodes::DynamicQuantizeStats>(xmax_src);
        stats->set_friendly_name(sub->get_friendly_name() + "/FusedDQLStats");
        ov::copy_runtime_info({sub, maxn, minn}, stats);
        // span (Subtract) → out[0]; xmin (Minimum) → out[1]. Replacing the Minimum
        // too lets ReduceMin/ReduceMax become dead and be removed by NopElimination.
        sub->output(0).replace(stats->output(0));
        minn->output(0).replace(stats->output(1));
        ++fused;
        changed = true;
    }

    if (fused)
        fprintf(stderr, "[DynamicQuantizeFusion] Fused %d min/max stat sub-graphs\n", fused);
    return changed;
}

}}} // namespaces
