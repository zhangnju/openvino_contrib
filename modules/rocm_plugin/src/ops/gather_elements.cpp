// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/op/gather_elements.hpp>
#include "kernels/gather_elements.hpp"

namespace ov {
namespace rocm_gpu {

GatherElementsOp::GatherElementsOp(const CreationContext& context,
                                   const std::shared_ptr<ov::Node>& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto ge = ov::as_type<const ov::op::v6::GatherElements>(node.get());
    OPENVINO_ASSERT(ge, "Node name: ", GetName());

    auto data_shape = node->get_input_shape(0);
    auto out_shape = node->get_output_shape(0);
    ndim_ = static_cast<int32_t>(data_shape.size());
    axis_ = static_cast<int32_t>(ge->get_axis());
    if (axis_ < 0) axis_ += ndim_;
    element_size_ = node->get_input_element_type(0).size();

    for (auto d : data_shape) data_shape_.push_back(static_cast<int64_t>(d));

    // compute out strides
    out_strides_.resize(ndim_);
    out_strides_[ndim_ - 1] = 1;
    for (int i = ndim_ - 2; i >= 0; --i) out_strides_[i] = out_strides_[i + 1] * out_shape[i + 1];
    total_out_ = 1;
    for (auto d : out_shape) total_out_ *= d;

    // compute data strides
    data_strides_.resize(ndim_);
    data_strides_[ndim_ - 1] = 1;
    for (int i = ndim_ - 2; i >= 0; --i) data_strides_[i] = data_strides_[i + 1] * data_shape[i + 1];
}

void GatherElementsOp::Execute(const InferenceRequestContext& context,
                                Inputs inputTensors,
                                Outputs outputTensors,
                                const Workbuffers&) const {
    OPENVINO_ASSERT(inputTensors.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());

    kernel::launchGatherElements(inputTensors[0].get(),
                                  inputTensors[1].get(),
                                  outputTensors[0].get(),
                                  data_strides_.data(),
                                  out_strides_.data(),
                                  axis_,
                                  ndim_,
                                  total_out_,
                                  element_size_,
                                  context.getThreadContext().stream().get());
}

OPERATION_REGISTER(GatherElementsOp, GatherElements);

}  // namespace rocm_gpu
}  // namespace ov
