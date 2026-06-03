// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "convolution_rocmlir.hpp"
#include "rocm_creation_context.hpp"
#include "error.hpp"

#include <openvino/core/except.hpp>
#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <cstring>

namespace ov {
namespace rocm_gpu {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: OpenVINO ConvolutionParams → rocmlir::ConvParams
// ─────────────────────────────────────────────────────────────────────────────

static rocmlir::ConvParams to_rocmlir_params(
        const Convolution::Details::ConvolutionParams& p,
        const rocm::Device& dev) {

    OPENVINO_ASSERT(p.NumberOfSpatialDims() == 2,
                    "ConvolutionRocMLIR: only 2-D convolution supported");

    rocmlir::ConvParams rp;
    // Input N×C×H×W
    rp.N = static_cast<int>(p.input_shape_[0]);
    rp.C = static_cast<int>(p.input_shape_[1]);
    rp.H = static_cast<int>(p.input_shape_[2]);
    rp.W = static_cast<int>(p.input_shape_[3]);
    // Filter K×C×R×S
    rp.K = static_cast<int>(p.filter_shape_[0]);
    rp.R = static_cast<int>(p.filter_shape_[2]);
    rp.S = static_cast<int>(p.filter_shape_[3]);
    // Padding, stride, dilation
    rp.pad_h     = static_cast<int>(p.padding_before_[0]);
    rp.pad_w     = static_cast<int>(p.padding_before_[1]);
    rp.stride_h  = static_cast<int>(p.strides_[0]);
    rp.stride_w  = static_cast<int>(p.strides_[1]);
    rp.dilation_h = static_cast<int>(p.dilations_[0]);
    rp.dilation_w = static_cast<int>(p.dilations_[1]);
    rp.groups    = static_cast<int>(p.groups_);
    // Data type
    rp.fp16 = (p.element_type_ == ov::element::Type_t::f16);
    // Architecture from HIP device props
    auto props = dev.props();
    // gcnArchName is e.g. "gfx950:sramecc+:xnack-"; strip suffix for rocmlir
    std::string arch_name(props.gcnArchName);
    auto colon = arch_name.find(':');
    if (colon != std::string::npos) arch_name = arch_name.substr(0, colon);
    rp.arch   = arch_name;
    rp.num_cu = props.multiProcessorCount;

    return rp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor: compile (or reuse cached) kernel
// ─────────────────────────────────────────────────────────────────────────────

ConvolutionRocMLIR::ConvolutionRocMLIR(
        const CreationContext& ctx,
        const ov::Node& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds,
        const Convolution::Details::ConvolutionParams& params)
    : OperationBase(ctx, node, std::move(inputIds), std::move(outputIds))
{
    conv_params_ = to_rocmlir_params(params, ctx.device());
    // compile_conv or fetch from cache
    kernel_ = &rocmlir::RocMLIRKernelCache::global().get_or_compile(conv_params_);
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute: launch the compiled kernel
// ─────────────────────────────────────────────────────────────────────────────

void ConvolutionRocMLIR::Execute(const InferenceRequestContext& ctx,
                                  Inputs inputs, Outputs outputs,
                                  const Workbuffers& wbs) const {
    using ArgIdx = Convolution::Details::ConvArgIndices;
    OPENVINO_ASSERT(inputs.size()  == 2, GetName(), ": expected 2 inputs");
    OPENVINO_ASSERT(outputs.size() == 1, GetName(), ": expected 1 output");

    void* d_input  = const_cast<void*>(inputs[ArgIdx::input].get());
    void* d_filter = const_cast<void*>(inputs[ArgIdx::filter].get());
    void* d_output = outputs[ArgIdx::output].get();
    void* d_ws     = wbs.mutable_buffers.empty() ? nullptr : wbs.mutable_buffers[0].get();

    // rocMLIR kernels expect arguments: (input, filter, output[, workspace])
    void* args[] = { &d_input, &d_filter, &d_output, &d_ws };
    const int nargs = (d_ws != nullptr) ? 4 : 3;

    const auto& info = kernel_->info;

    // Grid: kernel-specific; use a safe default if not parsed
    const int grid_x  = (conv_params_.N * conv_params_.K * info.block_x - 1) / info.block_x + 1;

    hipError_t err = hipModuleLaunchKernel(
        kernel_->function,
        grid_x, 1, 1,               // grid
        info.block_x, 1, 1,         // block
        0,                           // shared mem
        ctx.getThreadContext().stream().get(),
        args, nullptr);

    if (err != hipSuccess)
        throw_ov_exception(fmt::format("ConvolutionRocMLIR::Execute failed: {}",
                                       hipGetErrorString(err)));
}

WorkbufferRequest ConvolutionRocMLIR::GetWorkBufferRequest() const {
    const size_t ws = kernel_->info.workspace_bytes;
    if (ws > 0) return {{}, {ws}};
    return {{}, {}};
}

} // namespace rocm_gpu
} // namespace ov
