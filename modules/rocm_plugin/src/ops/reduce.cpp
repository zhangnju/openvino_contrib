// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "rocm/descriptor_utils.hpp"
#include "converters.hpp"
#include "reduce_sum.hpp"

namespace ov {
namespace rocm_gpu {

miopenDataType_t ReduceOp::reduceCompType(const ov::Node& node) {
    const auto in_shape = node.input(0).get_shape();
    const auto out_shape = node.output(0).get_shape();
    OPENVINO_ASSERT(in_shape.size() == out_shape.size(), "Node name: ", node.get_friendly_name());
    const auto in_type = convertDataType<miopenDataType_t>(node.get_input_element_type(0));
    const auto out_type = convertDataType<miopenDataType_t>(node.get_output_element_type(0));
    OPENVINO_ASSERT(in_type == out_type, "Node name: ", node.get_friendly_name());
    switch (in_type) {
        case miopenFloat:
        case miopenHalf:
            // TODO: it's unclear from documentation, whether it can be half when both tensors are
            // half, or int8 when both tensors are int8. we'll have to test it
            return miopenFloat;
        case miopenDouble:
            return miopenDouble;
        default:
            throw_ov_exception(fmt::format("ov::rocm_gpu::reduceCompType(): Unsupported data types: in0 = {}, in1 = {}",
                                         toString(in_type),
                                         toString(out_type)));
    }
}

ReduceOp::ReduceOp(const CreationContext& context,
                   const ov::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds,
                   const rocm::DnnReduceTensorDescriptor& reduce_desc)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      comp_type_{reduceCompType(node)},
      a_desc_{rocm::makeInputDnnTensorDescr(node, 0)},
      c_desc_{rocm::makeOutputDnnTensorDescr(node, 0)},
      reduce_desc_(reduce_desc),
      workspace_size_{context.dnnHandle().getReductionWorkspaceSize(reduce_desc_, a_desc_, c_desc_)} {}

void ReduceOp::Execute(const InferenceRequestContext& context,
                          Inputs inputTensors,
                          Outputs outputTensors,
                          const Workbuffers& workbuffers) const {
    context.getThreadContext().dnnHandle().reduceTensor(reduce_desc_,
                                                        workbuffers.createMutableSpanFrom<0>(workspace_size_),
                                                        rocm::DnnScaleFactorOne{comp_type_},
                                                        a_desc_,
                                                        inputTensors[0],
                                                        rocm::DnnScaleFactorZero{comp_type_},
                                                        c_desc_,
                                                        outputTensors[0]);
}

rocmGraphCompatibility ReduceOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

}  // namespace rocm_gpu
}  // namespace ov
