// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <fmt/format.h>

#include "interpolate_components.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"

#include "error.hpp"

namespace ov::rocm_gpu::Interpolate::Details {

void getAxesAndScales(const ov::op::v4::Interpolate& node, std::vector<size_t>& axes, std::vector<float>& scales) {
    const auto num_inputs = node.get_input_size();
    const auto& input_shape = node.get_input_shape(0);
    const auto& output_shape = node.get_output_shape(0);
    const auto rank = input_shape.size();

    // Extract axes: from input[3] if present, else default to all dimensions
    if (num_inputs >= 4) {
        auto axes_c = ov::as_type_ptr<op::v0::Constant>(node.input_value(3).get_node_shared_ptr());
        if (axes_c) {
            const auto& at = axes_c->get_element_type();
            if (at == ov::element::i64) {
                for (auto v : axes_c->cast_vector<int64_t>()) axes.push_back(static_cast<size_t>(v));
            } else if (at == ov::element::i32) {
                for (auto v : axes_c->cast_vector<int32_t>()) axes.push_back(static_cast<size_t>(v));
            } else {
                axes = axes_c->cast_vector<size_t>();
            }
        }
    }
    if (axes.empty()) {
        // Default: all dimensions
        for (size_t i = 0; i < rank; ++i) axes.push_back(i);
    }

    // Extract scales
    switch (node.get_attrs().shape_calculation_mode) {
        case ov::op::v4::Interpolate::ShapeCalcMode::SIZES: {
            scales.resize(axes.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                const auto axe = axes[i];
                scales[i] = static_cast<float>(output_shape[axe]) / static_cast<float>(input_shape[axe]);
            }
        } break;
        case ov::op::v4::Interpolate::ShapeCalcMode::SCALES:
            if (num_inputs >= 3) {
                auto scales_c = ov::as_type_ptr<op::v0::Constant>(node.input_value(2).get_node_shared_ptr());
                if (scales_c) {
                    scales = scales_c->cast_vector<float>();
                    OPENVINO_ASSERT(axes.size() == scales.size());
                    break;
                }
            }
            // Fallback: compute from shapes
            scales.resize(axes.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                const auto axe = axes[i];
                scales[i] = static_cast<float>(output_shape[axe]) / static_cast<float>(input_shape[axe]);
            }
            break;
        default:
            throw_ov_exception(fmt::format("Interpolate operation: unsupported shape calculation mode {}.",
                                         node.get_attrs().shape_calculation_mode));
    }
}

}  // namespace ov::rocm_gpu::Interpolate::Details
