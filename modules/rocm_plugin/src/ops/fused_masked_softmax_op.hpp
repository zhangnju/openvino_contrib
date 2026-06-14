// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedMaskedSoftmaxOp: add(mask) + softmax in a single native HIP kernel.
// No MIGraphX API calls.
//
// Inputs: [0]=scores[1,heads,sq,sk]  [1]=mask[sq,sk]
// Output: [0]=softmax(scores+mask)[1,heads,sq,sk]
#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fused_masked_softmax_node.hpp>

namespace ov {
namespace rocm_gpu {

class FusedMaskedSoftmaxOp : public OperationBase {
public:
    FusedMaskedSoftmaxOp(const CreationContext& context,
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
    int heads_{0};
    int sq_{0};
    int sk_{0};
};

}  // namespace rocm_gpu
}  // namespace ov
