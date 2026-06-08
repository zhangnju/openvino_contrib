// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <rocm_operation_base.hpp>
#include <vector>

#include "transformer/nodes/fused_elementwise_node.hpp"
#include "kernels/fused_elementwise.hpp"

namespace ov {
namespace rocm_gpu {

// ROCm Plugin operation for FusedElementwise nodes.
// Uses immutable workbuffers to store the GPU-side ops/params/aux-pointer arrays.
class FusedElementwiseOp : public OperationBase {
public:
    FusedElementwiseOp(const CreationContext& context,
                       const std::shared_ptr<ov::Node>& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);

    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

private:
    std::vector<uint8_t>          host_ops_;
    std::vector<float>            host_params_;
    std::vector<bool>             step_has_aux_;  // whether each step needs aux input
    int                           chain_len_{0};
    bool                          is_fp16_{false};
    int64_t                       num_elements_{0};
    int                           num_aux_{0};     // number of auxiliary inputs

};

}  // namespace rocm_gpu
}  // namespace ov
