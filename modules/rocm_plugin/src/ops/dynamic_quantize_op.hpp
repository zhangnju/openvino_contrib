// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GPU executor for nodes::DynamicQuantize — a single fused kernel that computes
// the per-tensor (global) uint8 dynamic quantization of an activation tensor.

#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/dynamic_quantize_node.hpp>

namespace ov {
namespace rocm_gpu {

class DynamicQuantizeOp : public OperationBase {
public:
    DynamicQuantizeOp(const CreationContext& context,
                      const std::shared_ptr<ov::Node>& node,
                      IndexCollection&& inputIds,
                      IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

private:
    size_t n_elems_{0};
    ov::element::Type x_type_{ov::element::f32};
};

}  // namespace rocm_gpu
}  // namespace ov
