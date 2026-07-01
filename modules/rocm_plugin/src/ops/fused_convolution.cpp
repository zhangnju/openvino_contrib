// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <error.hpp>
#include <exception>
#include <openvino/core/except.hpp>
#include <memory>

#include "convolution_components/convolution_miopen_components.hpp"
#include "rocm_operation_registry.hpp"
#include "fused_convolution_miopen.hpp"
#include "fused_convolution_miopen_decomposed.hpp"
#include "transformer/nodes/activation_type.hpp"
#include "openvino/op/swish.hpp"
#ifdef ENABLE_miopen_BACKEND_API
#include "fused_convolution_miopen_be.hpp"
#endif  // ENABLE_miopen_BACKEND_API
#ifdef ENABLE_ROCMLIR
#include "fused_convolution_rocmlir.hpp"
#endif  // ENABLE_ROCMLIR

#ifdef ENABLE_CK_WMMA
#include "convolution_ck_wmma.hpp"
#endif  // ENABLE_CK_WMMA

#ifdef ENABLE_WINOGRAD
#include "winograd_conv_op.hpp"
#endif  // ENABLE_WINOGRAD

namespace ov {
namespace rocm_gpu {

// No-op op: used when a FusedGroupConvolution's computation is already handled
// by an upstream fused kernel (e.g., pe(V)+attn fusion in AV MatMul).
class NoOpConvOp : public OperationBase {
public:
    NoOpConvOp(const CreationContext& ctx, const ov::Node& node,
               IndexCollection&& inputs, IndexCollection&& outputs)
        : OperationBase(ctx, node, std::move(inputs), std::move(outputs)) {}

    void Execute(const InferenceRequestContext&, Inputs, Outputs, const Workbuffers&) const override {}
    rocmGraphCompatibility GetrocmGraphCompatibility() const override { return rocmGraphCompatibility::FULL; }
};

OperationBase::Ptr fusedConvolutionFactory(const CreationContext& context,
                                           const std::shared_ptr<ov::Node>& node,
                                           OperationBase::IndexCollection&& inputIds,
                                           OperationBase::IndexCollection&& outputIds) {
    // If tagged as pe(V) conv (computation handled by AV MatMul fused kernel), return no-op.
    if (node->get_rt_info().count("rocm_attn_pe_conv")) {
        return std::make_shared<NoOpConvOp>(context, *node,
            OperationBase::IndexCollection{inputIds}, OperationBase::IndexCollection{outputIds});
    }
    using ArgIndices = Convolution::Details::FusedConvolutionIndices;
    using IndexCollection = OperationBase::IndexCollection;
    const auto element_type = node->get_input_element_type(ArgIndices::input);
    OPENVINO_ASSERT(element_type == node->get_input_element_type(ArgIndices::filter));
    OPENVINO_ASSERT(element_type == node->get_input_element_type(ArgIndices::bias));
    OPENVINO_ASSERT(element_type == node->get_output_element_type(ArgIndices::output));
    const bool includesOnlyBiasAdd = node->inputs().size() == 3;
    const bool includesSecondAddition = node->inputs().size() == 4;
    OPENVINO_ASSERT(includesOnlyBiasAdd || includesSecondAddition);  // Conv input, filters, Bias and optional Add

    std::stringstream exception_msg;
    const auto fused_conv = std::dynamic_pointer_cast<nodes::FusedConvolution>(node);
    const auto fused_group_conv = std::dynamic_pointer_cast<nodes::FusedGroupConvolution>(node);
    OPENVINO_ASSERT(fused_conv || fused_group_conv);

    auto params = fused_conv ? Convolution::Details::FusedConvolutionParams{*fused_conv}
                            : Convolution::Details::FusedConvolutionParams{*fused_group_conv};

    // Detect downstream Swish pattern: FusedConv → Swish (single consumer, no add).
    // Override activation to SWISH so FusedConvolutionRocMLIR compiles Conv+Bias+SiLU.
    // The downstream SwishOp uses thread-local SiLU tracking to become a no-op.
    // Always enabled when ENABLE_ROCMLIR is set; ROCMLIR_EPILOGUE_FUSION controls
    // whether MIGraphX-guided compilation is used for the SiLU kernel.
#ifdef ENABLE_ROCMLIR
    if (params.activation_ == nodes::ActivationMode::NO_ACTIVATION && !params.add_shape_.has_value()) {
        // Detect any Swish consumer: FusedConv/FusedGroupConv → ... → Swish.
        // When any consumer is Swish, compile Conv+Bias+SiLU fused kernel.
        // The downstream SwishOp uses silu_tracking + in-place buffer to become a no-op.
        for (const auto& tgt : node->output(0).get_target_inputs()) {
            if (ov::is_type<ov::op::v4::Swish>(tgt.get_node())) {
                params.activation_ = nodes::ActivationMode::SWISH;
                break;
            }
        }
    }
#endif  // ENABLE_ROCMLIR

#ifdef ENABLE_WINOGRAD
    if (WinogradConvOp::isEligible(params)) {
        try {
            return std::make_shared<WinogradConvOp>(
                context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
        } catch (const std::exception& e) {
            exception_msg << fmt::format("WinogradConvOp failed: {}", e.what());
        }
    }
#endif  // ENABLE_WINOGRAD

    // Priority 0: CK WMMA with auto-profiling vs rocMLIR (RDNA4/gfx12xx + FP16 only)
    // First Execute: times both kernels and permanently selects the faster one.
#ifdef ENABLE_CK_WMMA
    try {
        return std::make_shared<FusedConvolutionCkWmma>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << fmt::format("FusedConvolutionCkWmma failed: {}", e.what());
    }
#endif  // ENABLE_CK_WMMA

    // Priority 1: rocMLIR backend (rock.conv → HSACO)
#ifdef ENABLE_ROCMLIR
    try {
        return std::make_shared<FusedConvolutionRocMLIR>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << fmt::format(
            "unsupported `{}` node: Failed to create "
            "FusedConvolutionRocMLIR impl: {}",
            node->get_type_info().name, e.what());
    }
#endif  // ENABLE_ROCMLIR

#ifdef ENABLE_miopen_BACKEND_API
    const bool should_try_backend = node->get_type_name() == std::string("FusedConvolution");
    if (should_try_backend) {
        try {
            return std::make_shared<FusedConvolutionmiopenBE>(
                context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
        } catch (const std::exception& e) {
            exception_msg << fmt::format(
                "unsupported `{}` node: Failed to create "
                "FusedConvolutionmiopenBE impl: {}",
                node->get_type_info().name,
                e.what());
        }
    }
#endif  // ENABLE_miopen_BACKEND_API

    const auto conv_descs{std::make_shared<Convolution::Details::ConvolutionDescriptorsmiopen>(context, params.conv_,
        std::vector<miopenDataType_t>{miopenHalf, miopenFloat})}; // 119703: investigate whether we need HALF here
    const auto bias_desc{Convolution::Details::MakeFusedAddDescriptor(params.bias_shape_, params.conv_.element_type_)};
    const auto activation_desc{Convolution::Details::MakeFusedActivationDescriptor(params.activation_)};
    const auto add_desc{params.add_shape_ ? Convolution::Details::MakeFusedAddDescriptor(params.add_shape_.value(),
                                                                                         params.conv_.element_type_)
                                          : nullptr};

    // miopenConvolutionBiasActivationForward() doesn't work properly with miopen_ACTIVATION_IDENTITY and any algorithm
    // other than miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, so we should decompose the convolution node and call
    // separate miopen functions.
    // For more information see:
    // https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenConvolutionBiasActivationForward
    const bool should_decompose = params.activation_ == ov::rocm_gpu::nodes::ActivationMode::NO_ACTIVATION &&
                                  conv_descs->Algo().fwd_algo != miopenConvolutionFwdAlgoImplicitGEMM;

    if (should_decompose) {
        try {
            return std::make_shared<FusedConvolutionmiopenDecomposed>(context,
                                                                     *node,
                                                                     IndexCollection{inputIds},
                                                                     IndexCollection{outputIds},
                                                                     conv_descs,
                                                                     bias_desc,
                                                                     add_desc,
                                                                     activation_desc);
        } catch (const std::exception& e) {
            throw_ov_exception(
                fmt::format("unsupported `{}` node: Failed to create "
                            "FusedConvolutionmiopenDecomposed impl: {}",
                            node->get_type_info().name,
                            e.what()));
        }
    }

    try {
        return std::make_shared<FusedConvolutionmiopen>(context,
                                                       *node,
                                                       IndexCollection{inputIds},
                                                       IndexCollection{outputIds},
                                                       conv_descs,
                                                       bias_desc,
                                                       add_desc,
                                                       activation_desc);
    } catch (const std::exception& e) {
        exception_msg << fmt::format(
            "unsupported `{}` node: Failed to create "
            "FusedConvolutionmiopen impl: {}",
            node->get_type_info().name,
            e.what());
    }

    throw_ov_exception(fmt::format("Convolution node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(fusedConvolutionFactory, FusedConvolution);
OPERATION_REGISTER_FACTORY(fusedConvolutionFactory, FusedGroupConvolution);

}  // namespace rocm_gpu
}  // namespace ov
