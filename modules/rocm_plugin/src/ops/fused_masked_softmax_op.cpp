// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedMaskedSoftmaxOp: native HIP fused masked softmax kernel.
// No MIGraphX API calls.

#include "fused_masked_softmax_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "kernels/fused_reduce.hpp"

namespace ov {
namespace rocm_gpu {

FusedMaskedSoftmaxOp::FusedMaskedSoftmaxOp(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {

    auto ms = std::dynamic_pointer_cast<nodes::FusedMaskedSoftmax>(node);
    OPENVINO_ASSERT(ms, "FusedMaskedSoftmaxOp: expected FusedMaskedSoftmax node");

    heads_ = static_cast<int>(ms->get_heads());
    sq_    = static_cast<int>(ms->get_sq());
    sk_    = static_cast<int>(ms->get_sk());
}

void FusedMaskedSoftmaxOp::Execute(
        const InferenceRequestContext& context,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2 && outputs.size() == 1);

    kernel::launch_masked_softmax_fused(
        context.getThreadContext().stream().get(),
        inputs[0].get(), inputs[1].get(), outputs[0].get(),
        heads_, sq_, sk_);
}

OPERATION_REGISTER(FusedMaskedSoftmaxOp, FusedMaskedSoftmax);

}  // namespace rocm_gpu
}  // namespace ov
