// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Factory: FusedConvolutionSlice → FusedConvolutionRocMLIR.
// Creates a temporary FusedConvolution (with sliced input shape) to build
// ConvolutionParams, then uses FusedConvolutionRocMLIR with slice metadata.

#include <fmt/format.h>
#include <error.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>

#include "rocm_operation_registry.hpp"
#include "fused_convolution_rocmlir.hpp"
#include "fused_convolution_miopen.hpp"
#include "transformer/nodes/fused_convolution_slice.hpp"
#include "transformer/nodes/activation_type.hpp"
#include "ops/convolution_components/convolution_components.hpp"
#include "ops/convolution_components/convolution_miopen_components.hpp"
#include "openvino/op/swish.hpp"

namespace ov {
namespace rocm_gpu {

static OperationBase::Ptr fusedConvolutionSliceFactory(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        OperationBase::IndexCollection&& inputIds,
        OperationBase::IndexCollection&& outputIds)
{
    using nodes::FusedConvolutionSlice;
    using nodes::FusedConvolution;
    using nodes::ActivationMode;

    auto slice_node = std::dynamic_pointer_cast<FusedConvolutionSlice>(node);
    OPENVINO_ASSERT(slice_node, "fusedConvolutionSliceFactory: wrong node type");

    // Build FusedConvolutionParams via a temporary FusedConvolution node.
    // The temporary node uses the SLICED input shape (C_slice), which matches
    // the filter dimensions and passes OV's shape validation.
    const auto& in_shape_full = node->get_input_shape(0);  // [N, C_full, H, W]
    const auto& flt_shape     = node->get_input_shape(1);  // [K, C_slice, R, S]
    const auto& bias_shape    = node->get_input_shape(2);  // [K]
    const ov::element::Type etype = node->get_input_element_type(0);
    const int C_slice = static_cast<int>(flt_shape[1]);

    // Create synthetic sliced input: [N, C_slice, H, W]
    ov::Shape sliced_in_shape = {in_shape_full[0], (size_t)C_slice,
                                  in_shape_full[2], in_shape_full[3]};
    auto fake_in  = std::make_shared<ov::op::v0::Parameter>(etype, sliced_in_shape);
    auto fake_flt = std::make_shared<ov::op::v0::Parameter>(etype, flt_shape);
    auto fake_bias = std::make_shared<ov::op::v0::Parameter>(etype, bias_shape);

    // Check if any downstream consumer of this node is a Swish op.
    // If yes: compile Slice+Conv+Bias+SiLU kernel (SiLU tracking skips downstream Swish).
    // If no:  compile plain Slice+Conv+Bias kernel (no activation).
    bool has_swish_consumer = false;
    for (const auto& out : node->outputs()) {
        for (const auto& tgt : out.get_target_inputs()) {
            if (ov::is_type<ov::op::v4::Swish>(tgt.get_node()->shared_from_this())) {
                has_swish_consumer = true; break;
            }
        }
        if (has_swish_consumer) break;
    }
    const ActivationMode act_mode = has_swish_consumer
                                    ? ActivationMode::SWISH
                                    : ActivationMode::NO_ACTIVATION;

    // Temporary FusedConvolution with correct sliced shapes
    auto tmp_fc = std::make_shared<FusedConvolution>(
        fake_in->output(0), fake_flt->output(0), fake_bias->output(0),
        slice_node->get_strides(),
        slice_node->get_pads_begin(),
        slice_node->get_pads_end(),
        slice_node->get_dilations(),
        slice_node->get_auto_pad(),
        act_mode);

    Convolution::Details::FusedConvolutionParams params{*tmp_fc};

#ifdef ENABLE_ROCMLIR
    return std::make_shared<FusedConvolutionRocMLIR>(
        context, *node,
        OperationBase::IndexCollection{inputIds},
        OperationBase::IndexCollection{outputIds},
        params);
#endif
    throw_ov_exception("FusedConvolutionSlice: ENABLE_ROCMLIR not defined");
}

OPERATION_REGISTER_FACTORY(fusedConvolutionSliceFactory, FusedConvolutionSlice);

}  // namespace rocm_gpu
}  // namespace ov
