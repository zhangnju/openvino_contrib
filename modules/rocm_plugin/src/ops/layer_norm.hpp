// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <rocm_operation_base.hpp>
#include <rocm/dnn.hpp>
#include <transformer/nodes/layer_norm_node.hpp>

namespace ov {
namespace rocm_gpu {

// GPU executor for LayerNorm via miopenLayerNormForward.
// Inputs: [0]=x, [1]=scale (optional), [2]=bias (optional)
// Output: [0]=normalized y
class LayerNormOp : public OperationMIOPEN {
public:
    LayerNormOp(const CreationContext& context,
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
    int32_t normalized_dim_{0};
    float   epsilon_{1e-5f};
    bool    has_scale_{false};
    bool    has_bias_{false};
    size_t  mean_bytes_{0};
    size_t  rstd_bytes_{0};

    rocm::DnnTensorDescriptor x_desc_;
    rocm::DnnTensorDescriptor y_desc_;
    rocm::DnnTensorDescriptor mean_desc_;
    rocm::DnnTensorDescriptor rstd_desc_;
    rocm::DnnTensorDescriptor scale_desc_;
    rocm::DnnTensorDescriptor bias_desc_;
};

}  // namespace rocm_gpu
}  // namespace ov
