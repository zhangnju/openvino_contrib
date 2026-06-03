// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedConvolution backed by rocMLIR:
//   Conv + Bias (+ optional Add + optional Activation)
//
// Step 1: Conv kernel via rocMLIR (rock.conv).
// Step 2: Bias add via miopenConvolutionForwardBias (supports NC broadcast).
// Step 3: Optional skip-connection add via miopenOpTensor.
// Step 4: Optional activation via miopenActivationForward.

#pragma once

#include <rocm_operation_base.hpp>
#include "rocm/rocmlir_compiler.hpp"
#include "rocm/rocmlir_kernel_cache.hpp"
#include "rocm/dnn.hpp"
#include "ops/convolution_components/convolution_components.hpp"
#include "transformer/nodes/activation_type.hpp"

namespace ov {
namespace rocm_gpu {

class FusedConvolutionRocMLIR : public OperationMIOPEN {
public:
    FusedConvolutionRocMLIR(const CreationContext& ctx,
                             const ov::Node& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds,
                             const Convolution::Details::FusedConvolutionParams& params);

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

private:
    rocmlir::ConvParams conv_params_;
    const rocmlir::KernelEntry* kernel_ = nullptr;

    // Bias descriptor for miopenConvolutionForwardBias (shape: 1×K×1×1)
    rocm::DnnTensorDescriptor bias_desc_;
    // Output tensor descriptor (reused for bias add + activation)
    rocm::DnnTensorDescriptor output_desc_;
    // Optional: add descriptor (skip connection)
    std::optional<rocm::DnnTensorDescriptor> add_desc_;
    // Optional: activation
    std::optional<rocm::DnnActivationDescriptor> activation_desc_;

    bool has_add_ = false;
    nodes::ActivationMode activation_ = nodes::ActivationMode::NO_ACTIVATION;
};

} // namespace rocm_gpu
} // namespace ov
