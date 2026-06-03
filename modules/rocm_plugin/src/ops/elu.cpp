// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "elu.hpp"
#include "converters.hpp"
#include "rocm_operation_registry.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace rocm_gpu {

EluOp::EluOp(const CreationContext& context,
             const ov::Node& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 1 || node.get_input_size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());
    const auto input_element_type = node.get_input_element_type(0);
    const auto output_element_type = node.get_output_element_type(0);
    OPENVINO_ASSERT(input_element_type == output_element_type, "Node name: ", GetName());
    const auto input_shape = node.get_input_shape(0);
    const auto output_shape = node.get_output_shape(0);
    OPENVINO_ASSERT(input_shape == output_shape, "Node name: ", GetName());
    size_t num_elements = ov::shape_size(input_shape);
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    const auto elu = dynamic_cast<const ov::op::v0::Elu*>(&node);
    OPENVINO_ASSERT(elu, "Node name: ", GetName());
    const auto alpha = static_cast<float>(elu->get_alpha());
    kernel_ = kernel::Elu{
        convertDataType<ov::rocm_gpu::kernel::Type_t>(input_element_type), max_threads_per_block, num_elements, alpha};
}

void EluOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
    OPENVINO_ASSERT(inputTensors.size() >= 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());
    const auto& stream = context.getThreadContext().stream();
    (*kernel_)(stream.get(), inputTensors[0].get(), outputTensors[0].get());
}

rocmGraphCompatibility EluOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

OPERATION_REGISTER(EluOp, Elu);
}  // namespace rocm_gpu
}  // namespace ov
