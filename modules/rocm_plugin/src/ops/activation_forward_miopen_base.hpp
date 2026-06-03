// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <miopen/miopen.h>

#include <rocm/dnn.hpp>
#include <rocm_operation_base.hpp>
#include <initializer_list>

namespace ov {
namespace rocm_gpu {

class ActivationForwardmiopenOpBase : public OperationMIOPEN {
public:
    static constexpr std::size_t max_shape_size = 5;

    static constexpr std::initializer_list<miopenDataType_t> supported_types{
        miopenFloat, miopenDouble, miopenHalf, miopenInt8};

    ActivationForwardmiopenOpBase(std::unique_ptr<rocm::DnnActivationDescriptor> opDesc,
                                 const CreationContext& context,
                                 const ov::Node& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

protected:
    std::unique_ptr<rocm::DnnActivationDescriptor> op_desc_;
    rocm::DnnTensorDescriptor x_desc_;
    rocm::DnnTensorDescriptor y_desc_;
    miopenDataType_t data_type_;
};

}  // namespace rocm_gpu
}  // namespace ov
