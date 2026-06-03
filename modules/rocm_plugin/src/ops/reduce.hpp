// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

namespace ov {
namespace rocm_gpu {

class ReduceOp : public OperationMIOPEN {
public:
    ReduceOp(const CreationContext& context,
             const ov::Node& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds,
             const rocm::DnnReduceTensorDescriptor& reduce_desc);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    static miopenDataType_t reduceCompType(const ov::Node& node);

private:
    miopenDataType_t comp_type_;
    rocm::DnnReduceTensorDescriptor reduce_desc_;
    rocm::DnnTensorDescriptor a_desc_;
    rocm::DnnTensorDescriptor c_desc_;
    size_t workspace_size_;
};

inline WorkbufferRequest ReduceOp::GetWorkBufferRequest() const {
    return {{}, {workspace_size_}};  // TODO: find a way to allocate buffers from constructor
}

}  // namespace rocm_gpu
}  // namespace ov
