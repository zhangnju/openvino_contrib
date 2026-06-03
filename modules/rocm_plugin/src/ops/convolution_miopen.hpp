// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_components/convolution_miopen_components.hpp"
#include "rocm_operation_base.hpp"

namespace ov {
namespace rocm_gpu {

/**
 * @brief Implements `ov::op::v1::Convolution` using miopen API
 * which doesn't support asymmetric padding.
 */
class Convolutionmiopen : public OperationMIOPEN {
public:
    Convolutionmiopen(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds,
                     const Convolution::Details::ConvolutionParams& params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    Convolution::Details::ConvolutionDescriptorsmiopen descs_;
};

}  // namespace rocm_gpu
}  // namespace ov
