// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Fused LayerNorm using native HIP kernel (no MIGraphX API).
// Algorithm from MIGraphX's fuse_pointwise_reduce design:
//   One workgroup per row; block-level parallel reduction via warp shuffles.

#include "fused_layernorm_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "kernels/fused_reduce.hpp"

namespace ov {
namespace rocm_gpu {

FusedLayerNormOp::FusedLayerNormOp(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {

    auto ln = std::dynamic_pointer_cast<nodes::FusedLayerNorm>(node);
    OPENVINO_ASSERT(ln, "FusedLayerNormOp: expected FusedLayerNorm node");

    rows_ = static_cast<int>(ln->get_seq_len());
    cols_ = static_cast<int>(ln->get_hidden());
    has_residual_ = ln->get_has_residual();

    // With residual: inputs = [x, skip, gamma, beta]
    // Without:       inputs = [x, gamma, beta]
    has_scale_ = true;
    has_bias_  = true;
}

void FusedLayerNormOp::Execute(
        const InferenceRequestContext& context,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    OPENVINO_ASSERT(outputs.size() == 1);

    // With residual: inputs = [x, skip, gamma, beta]
    // Without:       inputs = [x, gamma, beta]
    const int offset      = has_residual_ ? 1 : 0;
    const void* skip_ptr  = has_residual_ ? inputs[1].get() : nullptr;
    const void* gamma_ptr = has_scale_    ? inputs[1 + offset].get() : nullptr;
    const void* beta_ptr  = has_bias_     ? inputs[2 + offset].get() : nullptr;

    kernel::launch_layernorm_fused(
        context.getThreadContext().stream().get(),
        inputs[0].get(), skip_ptr, gamma_ptr, beta_ptr, outputs[0].get(),
        rows_, cols_, epsilon_);
}

OPERATION_REGISTER(FusedLayerNormOp, FusedLayerNorm);

}  // namespace rocm_gpu
}  // namespace ov
