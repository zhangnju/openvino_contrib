// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// OneHot op: produces one-hot encoding along the last axis (axis=-1).
// Inputs:
//   [0] indices  — integer tensor (i32 or i64)
//   [1] depth    — scalar i64 constant (number of classes)
//   [2] values   — 2-element constant [off_value, on_value]  (OV v1 OneHot)
//                  (Note: OV stores [off, on] for the v1 op)
// Output:
//   indices.shape + [depth]  one-hot matrix

#include "one_hot.hpp"
#include <fmt/format.h>
#include <rocm_operation_registry.hpp>
#include <openvino/op/one_hot.hpp>
#include <openvino/op/constant.hpp>
#include "converters.hpp"

namespace ov {
namespace rocm_gpu {

OneHotOp::OneHotOp(const CreationContext& context,
                   const std::shared_ptr<ov::Node>& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    // indices shape
    auto indices_shape = node->get_input_shape(0);
    num_indices_ = 1;
    for (auto d : indices_shape) num_indices_ *= d;

    // depth: try to read from constant input[1]; if dynamic, read at Execute time
    auto depth_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
        node->get_input_node_shared_ptr(1));
    if (depth_const) {
        const auto& et = depth_const->get_element_type();
        if (et == ov::element::i64)
            depth_ = depth_const->cast_vector<int64_t>()[0];
        else if (et == ov::element::i32)
            depth_ = depth_const->cast_vector<int32_t>()[0];
        else
            depth_ = depth_const->cast_vector<float>().empty() ? 0 :
                     static_cast<int64_t>(depth_const->cast_vector<float>()[0]);
        depth_from_input_ = false;
    } else {
        depth_from_input_ = true;  // read depth at Execute time from inputs[1]
    }

    // Helper: safely read scalar value from Constant node as float
    auto read_scalar = [](const std::shared_ptr<ov::op::v0::Constant>& c) -> float {
        if (!c) return 0.f;
        auto et = c->get_element_type();
        if (et == ov::element::f32) {
            auto v = c->cast_vector<float>();
            return v.empty() ? 0.f : v[0];
        } else if (et == ov::element::f64) {
            auto v = c->cast_vector<double>();
            return v.empty() ? 0.f : static_cast<float>(v[0]);
        } else if (et == ov::element::i64) {
            auto v = c->cast_vector<int64_t>();
            return v.empty() ? 0.f : static_cast<float>(v[0]);
        } else if (et == ov::element::i32) {
            auto v = c->cast_vector<int32_t>();
            return v.empty() ? 0.f : static_cast<float>(v[0]);
        }
        return 0.f;
    };

    // OV v1::OneHot: input[2]=on_value (scalar), input[3]=off_value (scalar)
    // OV v0::OneHot style: input[2]=[off_value, on_value] 1D pair
    if (node->get_input_size() >= 4) {
        // v1 style: on_value at [2], off_value at [3]
        auto on_c  = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(2));
        auto off_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(3));
        on_val_  = read_scalar(on_c);
        off_val_ = read_scalar(off_c);
    } else if (node->get_input_size() >= 3) {
        // Old style: values pair [off, on] at input[2]
        auto vals_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(2));
        if (vals_c) {
            auto et = vals_c->get_element_type();
            if (et == ov::element::f32) {
                auto v = vals_c->cast_vector<float>();
                if (v.size() >= 2) { off_val_ = v[0]; on_val_ = v[1]; }
            } else if (et == ov::element::f64) {
                auto v = vals_c->cast_vector<double>();
                if (v.size() >= 2) { off_val_ = float(v[0]); on_val_ = float(v[1]); }
            } else if (et == ov::element::i64) {
                auto v = vals_c->cast_vector<int64_t>();
                if (v.size() >= 2) { off_val_ = float(v[0]); on_val_ = float(v[1]); }
            } else if (et == ov::element::i32) {
                auto v = vals_c->cast_vector<int32_t>();
                if (v.size() >= 2) { off_val_ = float(v[0]); on_val_ = float(v[1]); }
            } else {
                // fallback: read as int32
                auto v = vals_c->cast_vector<int32_t>();
                if (v.size() >= 2) { off_val_ = float(v[0]); on_val_ = float(v[1]); }
            }
        }
    }

    // Check if values need to be read at Execute time
    auto vals_const_2 = (node->get_input_size() >= 3) ?
        std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(2)) : nullptr;
    auto vals_const_3 = (node->get_input_size() >= 4) ?
        std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(3)) : nullptr;
    values_from_input_ = (vals_const_2 == nullptr && vals_const_3 == nullptr);

    // indices element type
    indices_i32_ = (node->get_input_element_type(0) == ov::element::i32);

    // output element type
    out_type_ = convertDataType<kernel::Type_t>(node->get_output_element_type(0));
}

void OneHotOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() >= 1 && outputs.size() == 1, "Node name: ", GetName());

    int64_t depth = depth_;
    float   on    = on_val_;
    float   off   = off_val_;

    // Read depth from device tensor at runtime if not a compile-time constant
    if (depth_from_input_ && inputs.size() >= 2) {
        int64_t tmp = 0;
        hipMemcpy(&tmp, inputs[1].get(), sizeof(int64_t), hipMemcpyDeviceToHost);
        depth = tmp;
    }

    // Read on/off values from device tensor at runtime if needed
    if (values_from_input_ && inputs.size() >= 3) {
        // values = [off, on] packed as float or int64
        // Use element type of output to determine format
        if (out_type_ == kernel::Type_t::f32 || out_type_ == kernel::Type_t::f16) {
            float pair[2] = {0.f, 1.f};
            hipMemcpy(pair, inputs[2].get(), 2 * sizeof(float), hipMemcpyDeviceToHost);
            off = pair[0]; on = pair[1];
        } else {
            int64_t pair[2] = {0, 1};
            hipMemcpy(pair, inputs[2].get(), 2 * sizeof(int64_t), hipMemcpyDeviceToHost);
            off = static_cast<float>(pair[0]);
            on  = static_cast<float>(pair[1]);
        }
    }

    kernel::launchOneHot(
        context.getThreadContext().stream().get(),
        inputs[0].get(),
        indices_i32_,
        on, off, depth,
        num_indices_,
        outputs[0].get(),
        out_type_);
}

OPERATION_REGISTER(OneHotOp, OneHot);

}  // namespace rocm_gpu
}  // namespace ov
