// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedLayerNormOp: single-pass fused LayerNorm kernel.
// Uses native HIP implementation inspired by MIGraphX's fuse_pointwise_reduce design.
// No MIGraphX API calls - pure HIP kernel in kernels/fused_reduce.hip.
//
// Inputs: [0]=x[rows,cols]  [1]=gamma[cols]  [2]=beta[cols]
// Output: [0]=LayerNorm(x, gamma, beta)[rows, cols]

#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fused_layernorm_node.hpp>

namespace ov {
namespace rocm_gpu {

class FusedLayerNormOp : public OperationBase {
public:
    FusedLayerNormOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

    WorkbufferRequest GetWorkBufferRequest() const override { return {}; }

private:
    int   rows_{0};
    int   cols_{0};
    float epsilon_{1e-12f};
    bool  has_scale_{false};
    bool  has_bias_{false};
    bool  has_residual_{false};
};

}  // namespace rocm_gpu
}  // namespace ov
