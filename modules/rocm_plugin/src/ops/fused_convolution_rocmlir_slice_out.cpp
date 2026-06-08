// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Factory: FusedConvolutionSliceOut → FusedConvolutionRocMLIR.
// Compiles Conv+Bias+SliceOutput+SiLU+Add(skip) via MIGraphX MLIR path.
// MIGraphX pattern: mlir_convolution_broadcast_slice_add_sigmoid_mul_add
//
// Arg order (5 args): (data, filter, bias, skip_input, output)
// The "slice" of the conv output is encoded as a Slice transform in the rocMLIR IR.

#include <fmt/format.h>
#include <error.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/constant.hpp>

#include "rocm_operation_registry.hpp"
#include "fused_convolution_rocmlir.hpp"
#include "transformer/nodes/fused_convolution_slice_out.hpp"
#include "transformer/nodes/activation_type.hpp"
#include "ops/convolution_components/convolution_components.hpp"
#include "ops/convolution_components/convolution_miopen_components.hpp"

namespace ov {
namespace rocm_gpu {

static OperationBase::Ptr fusedConvolutionSliceOutFactory(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        OperationBase::IndexCollection&& inputIds,
        OperationBase::IndexCollection&& outputIds)
{
    using nodes::FusedConvolutionSliceOut;
    using nodes::FusedConvolution;
    using nodes::ActivationMode;

    auto slice_out_node = std::dynamic_pointer_cast<FusedConvolutionSliceOut>(node);
    OPENVINO_ASSERT(slice_out_node, "fusedConvolutionSliceOutFactory: wrong node type");

    // node inputs: 0=data, 1=filter, 2=bias, 3=skip_input
    const auto& flt_shape  = node->get_input_shape(1);   // [K_full, C/G, R, S]
    const auto& in_shape   = node->get_input_shape(0);   // [N, C, H, W]
    const auto& bias_shape = node->get_input_shape(2);   // [K_full]
    const auto& skip_shape = node->get_input_shape(3);   // [N, K_slice, OH, OW]
    const auto& out_shape  = node->get_output_shape(0);  // [N, K_slice, OH, OW]
    const ov::element::Type_t etype = node->get_input_element_type(0);

    const int K_full  = static_cast<int>(flt_shape[0]);
    const int K_slice = slice_out_node->get_c_out_end() - slice_out_node->get_c_out_start();
    const int C       = static_cast<int>(flt_shape[1]);

    // Build FusedConvolutionParams via a temporary FusedConvolution node (standard K_full output)
    // then override add_shape_ for skip connection
    ov::Shape full_out_shape = {in_shape[0], (size_t)K_full, out_shape[2], out_shape[3]};
    auto fake_in   = std::make_shared<ov::op::v0::Parameter>(etype, in_shape);
    auto fake_flt  = std::make_shared<ov::op::v0::Parameter>(etype, flt_shape);
    auto fake_bias = std::make_shared<ov::op::v0::Parameter>(etype, bias_shape);

    auto tmp_fc = std::make_shared<FusedConvolution>(
        fake_in->output(0), fake_flt->output(0), fake_bias->output(0),
        slice_out_node->get_strides(),
        slice_out_node->get_pads_begin(),
        slice_out_node->get_pads_end(),
        slice_out_node->get_dilations(),
        slice_out_node->get_auto_pad(),
        ActivationMode::SWISH);

    Convolution::Details::FusedConvolutionParams params{*tmp_fc};
    // Override: output is K_slice, not K_full; and we have a skip-add input
    params.conv_.output_shape_ = out_shape;
    params.add_shape_ = skip_shape;

    // Store output-slice metadata in rt_info so FusedConvolutionRocMLIR
    // compiles the slice-out-silu-add kernel
    const_cast<ov::Node&>(*node).get_rt_info()["rocmlir_slice_out_c_start"] =
        std::to_string(slice_out_node->get_c_out_start());
    const_cast<ov::Node&>(*node).get_rt_info()["rocmlir_slice_out_c_end"] =
        std::to_string(slice_out_node->get_c_out_end());
    const_cast<ov::Node&>(*node).get_rt_info()["rocmlir_slice_out_k_full"] =
        std::to_string(K_full);

#ifdef ENABLE_ROCMLIR
    try {
        return std::make_shared<FusedConvolutionRocMLIR>(
            context, *node,
            OperationBase::IndexCollection{inputIds},
            OperationBase::IndexCollection{outputIds},
            params);
    } catch (const std::exception& e) {
        throw_ov_exception(fmt::format(
            "FusedConvolutionSliceOut: rocMLIR failed: {}", e.what()));
    }
#endif
    throw_ov_exception("FusedConvolutionSliceOut: ENABLE_ROCMLIR not defined");
}

OPERATION_REGISTER_FACTORY(fusedConvolutionSliceOutFactory, FusedConvolutionSliceOut);

}  // namespace rocm_gpu
}  // namespace ov
