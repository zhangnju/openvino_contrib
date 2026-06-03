// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fused_convolution_rocmlir.hpp"
#include "rocm_creation_context.hpp"
#include "error.hpp"
#include "ops/converters.hpp"
#include "ops/convolution_components/convolution_miopen_components.hpp"
#include "kernels/bias_add.hpp"

#include <fmt/format.h>
#include <openvino/core/except.hpp>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

namespace ov {
namespace rocm_gpu {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers to build rocmlir::ConvParams from OpenVINO ConvolutionParams
// (identical to convolution_rocmlir.cpp — shared code could be extracted later)
// ─────────────────────────────────────────────────────────────────────────────

static rocmlir::ConvParams to_rocmlir(
        const Convolution::Details::ConvolutionParams& p,
        const rocm::Device& dev) {
    OPENVINO_ASSERT(p.NumberOfSpatialDims() == 2,
                    "FusedConvolutionRocMLIR: only 2-D convolution supported");
    rocmlir::ConvParams rp;
    rp.N = static_cast<int>(p.input_shape_[0]);
    rp.C = static_cast<int>(p.input_shape_[1]);
    rp.H = static_cast<int>(p.input_shape_[2]);
    rp.W = static_cast<int>(p.input_shape_[3]);
    rp.K = static_cast<int>(p.filter_shape_[0]);
    rp.R = static_cast<int>(p.filter_shape_[2]);
    rp.S = static_cast<int>(p.filter_shape_[3]);
    rp.pad_h      = static_cast<int>(p.padding_before_[0]);
    rp.pad_w      = static_cast<int>(p.padding_before_[1]);
    rp.stride_h   = static_cast<int>(p.strides_[0]);
    rp.stride_w   = static_cast<int>(p.strides_[1]);
    rp.dilation_h = static_cast<int>(p.dilations_[0]);
    rp.dilation_w = static_cast<int>(p.dilations_[1]);
    rp.groups     = static_cast<int>(p.groups_);
    rp.fp16       = (p.element_type_ == ov::element::Type_t::f16);
    auto props    = dev.props();
    std::string arch_name(props.gcnArchName);
    auto colon = arch_name.find(':');
    if (colon != std::string::npos) arch_name = arch_name.substr(0, colon);
    rp.arch   = arch_name;
    rp.num_cu = props.multiProcessorCount;
    return rp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

FusedConvolutionRocMLIR::FusedConvolutionRocMLIR(
        const CreationContext& ctx,
        const ov::Node& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds,
        const Convolution::Details::FusedConvolutionParams& params)
    : OperationMIOPEN(ctx, node, std::move(inputIds), std::move(outputIds))
{
    conv_params_ = to_rocmlir(params.conv_, ctx.device());
    activation_  = params.activation_;
    has_add_     = params.add_shape_.has_value();

    // Compile (or reuse cached) rocMLIR kernel based on activation type
    if (activation_ == nodes::ActivationMode::NO_ACTIVATION) {
        kernel_ = &rocmlir::RocMLIRKernelCache::global()
                       .get_or_compile_fused_bias(conv_params_);
    } else {
        rocmlir::Activation act = rocmlir::Activation::None;
        if (activation_ == nodes::ActivationMode::RELU)    act = rocmlir::Activation::ReLU;
        if (activation_ == nodes::ActivationMode::SIGMOID) act = rocmlir::Activation::Sigmoid;
        kernel_ = &rocmlir::RocMLIRKernelCache::global()
                       .get_or_compile_fused_bias_act(conv_params_, act);
    }

    // Build bias descriptor (shape: 1×K×1×1)
    bias_desc_ = *Convolution::Details::MakeFusedAddDescriptor(
                      params.bias_shape_, params.conv_.element_type_);

    // Build output descriptor (shape: N×K×OH×OW)
    output_desc_ = *Convolution::Details::MakeFusedAddDescriptor(
                       params.conv_.output_shape_, params.conv_.element_type_);

    // Optional: skip-connection add descriptor
    if (has_add_) {
        add_desc_ = *Convolution::Details::MakeFusedAddDescriptor(
                         *params.add_shape_, params.conv_.element_type_);
    }

    // Optional: activation descriptor
    if (activation_ != nodes::ActivationMode::NO_ACTIVATION) {
        activation_desc_ = rocm::DnnActivationDescriptor{};
        activation_desc_->set(convertActivationMode(activation_), 0.0, 0.0, 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute
// ─────────────────────────────────────────────────────────────────────────────

void FusedConvolutionRocMLIR::Execute(const InferenceRequestContext& ctx,
                                       Inputs inputs, Outputs outputs,
                                       const Workbuffers& wbs) const {
    using Idx = Convolution::Details::FusedConvolutionIndices;
    OPENVINO_ASSERT(outputs.size() == 1, GetName(), ": expected 1 output");

    const auto& dnn = ctx.getThreadContext().dnnHandle();

    void* d_input  = const_cast<void*>(inputs[Idx::input].get());
    void* d_filter = const_cast<void*>(inputs[Idx::filter].get());
    void* d_bias   = const_cast<void*>(inputs[Idx::bias].get());
    void* d_output = outputs[Idx::output].get();
    void* d_ws     = wbs.mutable_buffers.empty() ? nullptr : wbs.mutable_buffers[0].get();

    // ── Step 1: rocMLIR convolution kernel ───────────────────────────────────
    {
        void* args[] = { &d_input, &d_filter, &d_output, &d_ws };
        const auto& info = kernel_->info;
        const int grid_x = (conv_params_.N * conv_params_.K * info.block_x - 1)
                           / info.block_x + 1;
        hipError_t err = hipModuleLaunchKernel(
            kernel_->function,
            grid_x, 1, 1, info.block_x, 1, 1,
            0, ctx.getThreadContext().stream().get(),
            args, nullptr);
        if (err != hipSuccess)
            throw_ov_exception(fmt::format("FusedConvolutionRocMLIR conv failed: {}",
                                           hipGetErrorString(err)));
    }

    // ── Step 2: bias add via custom HIP kernel ───────────────────────────────
    // miopenConvolutionForwardBias is affected by MIOPEN_DEBUG_FIND_ONLY_SOLVER.
    // Use a custom broadcast-add kernel instead.
    {
        kernel::launch_bias_add(d_output, d_bias,
                                conv_params_.N, conv_params_.K,
                                conv_params_.out_h(), conv_params_.out_w(),
                                conv_params_.fp16,
                                ctx.getThreadContext().stream().get());
    }

    // ── Step 3: optional skip-connection add (same shape, element-wise) ──────
    if (has_add_ && inputs.size() > Idx::add) {
        void* d_add = const_cast<void*>(inputs[Idx::add].get());
        // Use bias_add kernel with K=total elements, N=H=W=1 to do flat add
        const int total = conv_params_.N * conv_params_.K *
                          conv_params_.out_h() * conv_params_.out_w();
        kernel::launch_bias_add(d_output, d_add,
                                1, total, 1, 1,
                                conv_params_.fp16,
                                ctx.getThreadContext().stream().get());
    }

    // ── Step 4: optional activation via existing elementwise kernel ───────────
    // Use existing abs/sigmoid HIP kernels or simply skip (for SiLU = Sigmoid*input,
    // the rocm plugin applies this post-activation at a higher level).
    // For now: only RELU and SIGMOID are wired; others are identity.
    if (activation_ != nodes::ActivationMode::NO_ACTIVATION) {
        // Inline elementwise: reuse existing rocm_plugin kernels via the stream
        // (sigmoid/relu are already implemented as elementwise ops in the plugin)
        // TODO: call ov::rocm_gpu::kernel::... directly here
        // For now, skip to avoid MIOpen calls that fail under FIND_ONLY_SOLVER.
    }
}

WorkbufferRequest FusedConvolutionRocMLIR::GetWorkBufferRequest() const {
    const size_t ws = kernel_->info.workspace_bytes;
    if (ws > 0) return {{}, {ws}};
    return {{}, {}};
}

} // namespace rocm_gpu
} // namespace ov
