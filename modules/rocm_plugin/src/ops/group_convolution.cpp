// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_convolution.hpp"

#include <rocm_operation_registry.hpp>

#include "convolution_components/convolution_components.hpp"

namespace ov {
namespace rocm_gpu {

GroupConvolutionOp::GroupConvolutionOp(const CreationContext &context,
                                       const NodeOp &node,
                                       IndexCollection &&inputIds,
                                       IndexCollection &&outputIds)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      convolution_(context, node, {}, {}, Convolution::Details::ConvolutionParams{node}) {}

void GroupConvolutionOp::Execute(const InferenceRequestContext &context,
                                 Inputs inputTensors,
                                 Outputs outputTensors,
                                 const Workbuffers &buffers) const {
    convolution_.Execute(context, inputTensors, outputTensors, buffers);
}

rocmGraphCompatibility GroupConvolutionOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

WorkbufferRequest GroupConvolutionOp::GetWorkBufferRequest() const { return convolution_.GetWorkBufferRequest(); }

OPERATION_REGISTER(GroupConvolutionOp, GroupConvolution);

}  // namespace rocm_gpu
}  // namespace ov
