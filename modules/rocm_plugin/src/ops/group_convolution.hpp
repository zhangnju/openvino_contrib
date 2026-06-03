// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory.h>

#include <openvino/op/group_conv.hpp>

#include "convolution_miopen.hpp"
#include "rocm_operation_base.hpp"

namespace ov {
namespace rocm_gpu {

class GroupConvolutionOp : public OperationMIOPEN {
public:
    using NodeOp = ov::op::v1::GroupConvolution;
    GroupConvolutionOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override final;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override final;

private:
    Convolutionmiopen convolution_;
};

}  // namespace rocm_gpu
}  // namespace ov
