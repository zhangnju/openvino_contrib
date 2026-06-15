// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedLayerNormOp: uses hipRTC JIT-compiled register-cached kernel when eligible,
// with fallback to the native HIP kernel in fused_reduce.hip.
#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fused_layernorm_node.hpp>
#include "kernels/fused_reduce.hpp"
#include <memory>

namespace ov {
namespace rocm_gpu {

struct LNKernel;  // forward decl for hipRTC compiled kernel

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
    bool  has_bias_{false};      // LN beta parameter (always true)
    bool  has_residual_{false};  // has skip/residual input
    bool  has_add_bias_{false};  // 3-input mode: has additive bias constant
    std::shared_ptr<LNKernel> jit_kernel_;  // hipRTC JIT kernel (nullptr = use fallback)
};

}  // namespace rocm_gpu
}  // namespace ov
