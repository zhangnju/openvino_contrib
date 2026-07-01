// Roll fusion: matches Slice+Slice→Concat pattern that implements circular shift.
// Pattern: Concat(axis=A, [Slice(x, start=S, end=END), Slice(x, start=0, end=S)])
// where both Slices read from the same source x. This is equivalent to roll(x, shift=-S, axis=A).
//
// Additionally detects two consecutive single-axis rolls (axis=1 then axis=2)
// and merges them into a single Roll node with shifts on both axes.

#include "roll_fusion.hpp"
#include "nodes/roll_node.hpp"

#include <openvino/op/concat.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/core/graph_util.hpp>
#include <iostream>
#include <set>

namespace ov {
namespace rocm_gpu {
namespace pass {

// Extract the begin value along the concat axis from a StridedSlice node.
// StridedSlice inputs: [data, begin, end, strides] where begin/end are Constant vectors.
// Returns the begin value at the given axis, or -999999 on failure.
static int64_t get_strided_slice_begin(const std::shared_ptr<ov::Node>& ss, int64_t axis) {
    auto begin_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        ss->get_input_node_shared_ptr(1));
    if (!begin_const) return -999999;
    auto begins = begin_const->cast_vector<int64_t>();
    if (axis < 0 || axis >= (int64_t)begins.size()) return -999999;
    return begins[axis];
}

static int64_t get_strided_slice_end(const std::shared_ptr<ov::Node>& ss, int64_t axis) {
    auto end_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        ss->get_input_node_shared_ptr(2));
    if (!end_const) return -999999;
    auto ends = end_const->cast_vector<int64_t>();
    if (axis < 0 || axis >= (int64_t)ends.size()) return -999999;
    return ends[axis];
}

static bool try_match_roll(const std::shared_ptr<ov::op::v0::Concat>& concat,
                           ov::Output<ov::Node>& source,
                           int64_t& shift, int64_t& axis) {
    if (concat->get_input_size() != 2) return false;
    axis = concat->get_axis();

    auto in0 = concat->get_input_node_shared_ptr(0);
    auto in1 = concat->get_input_node_shared_ptr(1);

    // Pattern: VariadicSplit(x, axis, [S, dim-S]) → [part0, part1]
    // Concat(part1, part0) = roll(x, -S)  (second part first = shift left by S)
    // Concat(part0, part1) = identity (no shift) — skip this

    // Both inputs from the same VariadicSplit, using its two outputs
    auto vs0 = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(in0);
    auto vs1 = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(in1);

    // Case: both come from same VariadicSplit (different output ports)
    if (vs0 && vs0.get() == vs1.get()) {
        // Doesn't happen — each Concat input is a separate VariadicSplit output
    }

    // Case: in0 and in1 are outputs of the SAME VariadicSplit
    auto parent0 = in0->get_input_node_shared_ptr(0);  // this doesn't work for multi-output
    // Actually, for VariadicSplit outputs, the node is the same but output index differs.
    // Let's check by looking at the source output connections:

    // in0 is output port X of some node, in1 is output port Y of same node
    auto src_out0 = concat->input_value(0);
    auto src_out1 = concat->input_value(1);

    if (src_out0.get_node() == src_out1.get_node()) {
        // Same node, different output ports — this IS a VariadicSplit
        auto vs = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(
            src_out0.get_node_shared_ptr());
        if (!vs) return false;

        source = vs->input_value(0);
        auto src_ps = source.get_partial_shape();
        if (!src_ps.is_static()) return false;
        int64_t dim_size = src_ps.to_shape()[axis];

        // Get split axis from VariadicSplit
        auto vs_axis_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            vs->get_input_node_shared_ptr(1));
        if (!vs_axis_const) return false;
        int64_t vs_axis = vs_axis_const->cast_vector<int64_t>()[0];
        if (vs_axis < 0) vs_axis += src_ps.rank().get_length();
        if (vs_axis != axis) return false;

        // Get split lengths
        auto lengths_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            vs->get_input_node_shared_ptr(2));
        if (!lengths_const) return false;
        auto lengths = lengths_const->cast_vector<int64_t>();
        if (lengths.size() != 2) return false;

        int port0 = src_out0.get_index();
        int port1 = src_out1.get_index();

        // Concat(vs[1], vs[0]) = roll(x, -lengths[0])
        // Concat(vs[0], vs[1]) = identity (no roll)
        if (port0 == 1 && port1 == 0) {
            shift = -(int64_t)lengths[0];
            return true;
        }
        if (port0 == 0 && port1 == 1) {
            shift = (int64_t)lengths[1];
            return true;
        }
        return false;
    }

    // Case: two separate VariadicSplit nodes from same source (shouldn't happen normally)
    if (in0->get_type_name() == std::string("VariadicSplit") &&
        in1->get_type_name() == std::string("VariadicSplit") &&
        in0->input_value(0) == in1->input_value(0)) {
        source = in0->input_value(0);
        // This is the original two-Slice pattern but with VariadicSplit
        // More complex — skip for now
    }

    return false;
}

bool RollFusionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    int count = 0;
    std::set<ov::Node*> removed;

    auto ops = model->get_ordered_ops();

    // Debug: count op types relevant to Roll
    int n_concat = 0, n_slice = 0, n_strided_slice = 0;
    for (const auto& op : ops) {
        std::string tn = op->get_type_name();
        if (tn == "Concat") n_concat++;
        if (tn == "Slice") n_slice++;
        if (tn == "StridedSlice") n_strided_slice++;
    }
    std::cerr << "[RollFusion] Graph: " << ops.size() << " ops, "
              << n_concat << " Concat, " << n_slice << " Slice, "
              << n_strided_slice << " StridedSlice" << std::endl;

    for (const auto& node : ops) {
        if (removed.count(node.get())) continue;
        auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(node);
        if (!concat) continue;

        // Debug: log Concat input types
        if (concat->get_input_size() == 2) {
            auto i0 = concat->get_input_node_shared_ptr(0);
            auto i1 = concat->get_input_node_shared_ptr(1);
            bool same_src = (i0->get_input_size() > 0 && i1->get_input_size() > 0 &&
                             i0->input_value(0) == i1->input_value(0));
            std::cerr << "[RollFusion]   Concat axis=" << concat->get_axis()
                      << " in0=" << i0->get_type_name()
                      << " in1=" << i1->get_type_name()
                      << " same_src=" << same_src << std::endl;
        }

        ov::Output<ov::Node> source;
        int64_t shift_val, axis_val;
        if (!try_match_roll(concat, source, shift_val, axis_val)) continue;

        // Check if this Roll's output feeds into another Roll (consecutive rolls on different axes)
        auto consumers = concat->output(0).get_target_inputs();
        std::shared_ptr<ov::op::v0::Concat> next_concat;
        if (consumers.size() == 1) {
            next_concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(
                consumers.begin()->get_node()->shared_from_this());
        }

        ov::Output<ov::Node> next_source;
        int64_t next_shift, next_axis;
        if (next_concat && !removed.count(next_concat.get()) &&
            try_match_roll(next_concat, next_source, next_shift, next_axis) &&
            next_axis != axis_val) {
            // Merge two consecutive rolls into one
            auto roll = std::make_shared<nodes::Roll>(
                source,
                std::vector<int64_t>{shift_val, next_shift},
                std::vector<int64_t>{axis_val, next_axis});
            ov::replace_output_update_name(next_concat->output(0), roll->output(0));
            removed.insert(concat.get());
            removed.insert(next_concat.get());
            count++;
        } else {
            // Single axis roll
            auto roll = std::make_shared<nodes::Roll>(
                source,
                std::vector<int64_t>{shift_val},
                std::vector<int64_t>{axis_val});
            ov::replace_output_update_name(concat->output(0), roll->output(0));
            removed.insert(concat.get());
            count++;
        }
        changed = true;
    }

    if (count > 0)
        std::cerr << "[RollFusion] Fused " << count << " Roll patterns" << std::endl;
    return changed;
}

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
