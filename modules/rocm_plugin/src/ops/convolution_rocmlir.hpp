// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Convolution operator backed by rocMLIR:
//   ConvolutionParams → rock.conv IR → rocmlir-driver → HSACO → hipLaunchKernel

#pragma once

#include <rocm_operation_base.hpp>
#include <rocm/rocmlir_compiler.hpp>
#include <rocm/rocmlir_kernel_cache.hpp>
#include <hip/hip_runtime.h>
#include <string>

#include "ops/convolution_components/convolution_components.hpp"

namespace ov {
namespace rocm_gpu {

class ConvolutionRocMLIR : public OperationBase {
public:
    ConvolutionRocMLIR(const CreationContext& context,
                       const ov::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds,
                       const Convolution::Details::ConvolutionParams& params);

    ~ConvolutionRocMLIR() = default;

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

private:
    rocmlir::ConvParams conv_params_;
    // Pointer into global cache — valid for the lifetime of the cache
    const rocmlir::KernelEntry* kernel_ = nullptr;
};

} // namespace rocm_gpu
} // namespace ov
