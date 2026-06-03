// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "avgpool.hpp"

#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/avg_pool.hpp>

namespace ov {
namespace rocm_gpu {

AvgPoolOp::AvgPoolOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationMIOPEN(context, node, std::move(inputIds), std::move(outputIds)),
      impl_{dynamic_cast<const ov::op::v1::AvgPool&>(*node)} {}

void AvgPoolOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    impl_.Execute(context.getThreadContext().dnnHandle(),
                  inputs[PoolingImpl::input_index].get(),
                  outputs[PoolingImpl::output_index].get());
}

rocmGraphCompatibility AvgPoolOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

OPERATION_REGISTER(AvgPoolOp, AvgPool);

}  // namespace rocm_gpu
}  // namespace ov
