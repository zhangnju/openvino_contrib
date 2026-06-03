// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

namespace ov {
namespace rocm_gpu {

class miopenTensorOpBase : public OperationMIOPEN {
public:
    static constexpr std::size_t max_supported_shape_size = 5;

    miopenTensorOpBase(const CreationContext& context,
                      const std::shared_ptr<ov::Node>& node,
                      IndexCollection&& inputIds,
                      IndexCollection&& outputIds,
                      const miopenTensorOp_t& opType,
                      const miopenNanPropagation_t& nanPropogationType = miopenNanPropagation_t::MIOPEN_PROPAGATE_NAN);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    struct IoParams {
        const miopenDataType_t type_;
        const ov::Shape shape_;
        std::array<int, 5> array_;
        rocm::DnnTensorDescriptor desc_;
        enum class Type { INPUT, OUTPUT };

        IoParams(const ov::Node& node, const Type& io_type, int index);
    };
    /*
    static rocm::DnnTensorDescriptor makeDnnOpTensorDescriptor(miopenTensorOp_t opType,
                                                                 miopenDataType_t dataType,
                                                                 miopenNanPropagation_t nanPropogationType) {
        return rocm::DnnTensorDescriptor{}.set(opType, dataType, nanPropogationType);
    }
    */
    IoParams in0;
    IoParams in1;
    IoParams out;
    rocm::DnnTensorDescriptor op_desc_;
    miopenTensorOp_t op_type_;
    int bias_index_ = 0;
    int dest_index_ = 1;
};
}  // namespace rocm_gpu
}  // namespace ov
