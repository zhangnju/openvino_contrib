// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "convolution_components/convolution_miopen_components.hpp"
#include "rocm/dnn.hpp"
#include "rocm_operation_base.hpp"

namespace ov {
namespace rocm_gpu {

/**
 * @brief This class was created as a workaround for the following miopen behavior:
 * miopenConvolutionBiasActivationForward() doesn't work properly with miopen_ACTIVATION_IDENTITY and any algorithm
 * other than miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, so we should decompose the convolution node and call
 * separate miopen functions.
 * For more information see:
 * https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenConvolutionBiasActivationForward
 */
class FusedConvolutionmiopenDecomposed : public OperationMIOPEN {
public:
    FusedConvolutionmiopenDecomposed(const CreationContext& context,
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
    void ThrowIfShouldNotDecompose() const;

    WorkbufferIds workbuffer_ids_;
    std::shared_ptr<Convolution::Details::ConvolutionDescriptorsmiopen> conv_descs_;
    std::shared_ptr<rocm::DnnTensorDescriptor> bias_desc_;
    std::shared_ptr<rocm::DnnTensorDescriptor> add_desc_;
    std::shared_ptr<rocm::DnnActivationDescriptor> activation_desc_;
};

}  // namespace rocm_gpu
}  // namespace ov
