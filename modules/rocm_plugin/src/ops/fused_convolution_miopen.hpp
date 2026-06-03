// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "convolution_components/convolution_miopen_components.hpp"
#include "rocm/dnn.hpp"
#include "rocm_operation_base.hpp"

namespace ov {
namespace rocm_gpu {

class FusedConvolutionmiopen : public OperationMIOPEN {
public:
    FusedConvolutionmiopen(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds,
                          Convolution::Details::FusedConvolutionParams params);

    FusedConvolutionmiopen(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds,
                          std::shared_ptr<Convolution::Details::ConvolutionDescriptorsmiopen> conv_descs_,
                          std::shared_ptr<rocm::DnnTensorDescriptor> bias_desc_,
                          std::shared_ptr<rocm::DnnTensorDescriptor> add_desc_,
                          std::shared_ptr<rocm::DnnActivationDescriptor> activation_desc_);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;
    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    void ThrowIfShouldDecompose() const;
    WorkbufferIds workbuffer_ids_;
    std::shared_ptr<Convolution::Details::ConvolutionDescriptorsmiopen> conv_descs_;
    std::shared_ptr<rocm::DnnTensorDescriptor> bias_desc_;
    std::shared_ptr<rocm::DnnTensorDescriptor> add_desc_;
    std::shared_ptr<rocm::DnnActivationDescriptor> activation_desc_;
};

}  // namespace rocm_gpu
}  // namespace ov
