// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_forward_miopen_base.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <rocm/constant_factory.hpp>
#include <rocm/descriptor_utils.hpp>

#include "converters.hpp"

namespace ov {
namespace rocm_gpu {

ActivationForwardmiopenOpBase::ActivationForwardmiopenOpBase(std::unique_ptr<rocm::DnnActivationDescriptor> opDesc,
                                                           const CreationContext& context,
                                                           const ov::Node& node,
                                                           IndexCollection&& inputIds,
                                                           IndexCollection&& outputIds)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      op_desc_{std::move(opDesc)},
      x_desc_{rocm::makeInputDnnTensorDescr(node, 0)},
      y_desc_{rocm::makeOutputDnnTensorDescr(node, 0)},
      data_type_{convertDataType<miopenDataType_t>(node.get_input_element_type(0))} {
    OPENVINO_ASSERT(node.get_input_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    if (std::find(supported_types.begin(), supported_types.end(), data_type_) == supported_types.end()) {
        throw_ov_exception(
            fmt::format("ov::rocm_gpu::ActivationForwardmiopenOpBase: unsupported data type: {}", toString(data_type_)));
    }

    const auto& shape = node.get_input_shape(0);
    OPENVINO_ASSERT(node.get_output_shape(0) == shape, "Node name: ", GetName());

    const auto in_shape_size = node.get_input_shape(0).size();
    if (in_shape_size > max_shape_size) {
        throw_ov_exception(
            fmt::format("ov::rocm_gpu::ActivationForwardmiopenOpBase: in_shape_size > max_shape_size: in_shape_size = {}, "
                        "max_shape_size = {}",
                        in_shape_size,
                        max_shape_size));
    }
}

void ActivationForwardmiopenOpBase::Execute(const InferenceRequestContext& context,
                                           Inputs inputTensors,
                                           Outputs outputTensors,
                                           const Workbuffers&) const {
    context.getThreadContext().dnnHandle().activationForward(*op_desc_,
                                                             &rocm::NumericConst<rocm::constants::one>(data_type_),
                                                             x_desc_,
                                                             inputTensors[0].get(),
                                                             &rocm::NumericConst<rocm::constants::zero>(data_type_),
                                                             y_desc_,
                                                             outputTensors[0].get());
}

rocmGraphCompatibility ActivationForwardmiopenOpBase::GetrocmGraphCompatibility() const {
    return rocmGraphCompatibility::FULL;
}

}  // namespace rocm_gpu
}  // namespace ov
