// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <miopen/miopen.h>

#include <rocm/device_pointers.hpp>
#include <rocm_operation_base.hpp>
#include <openvino/op/softmax.hpp>

namespace ov {
namespace rocm_gpu {

class SoftmaxOp : public OperationMIOPEN {
public:
    using NodeOp = ov::op::v1::Softmax;

    SoftmaxOp(const CreationContext& context,
              const NodeOp& node,
              IndexCollection&& inputIds,
              IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    void mapRankAxis(const ov::Shape& shape, int axis);
    std::array<int, 4> shape_;
    miopenDataType_t type_;
    rocm::DnnTensorDescriptor tensor_descriptor_;
    bool use_custom_kernel_ = false;
    size_t custom_rows_ = 0;
    int custom_cols_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
