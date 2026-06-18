// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Detects the onnx DynamicQuantizeLinear decomposition emitted by the OV ONNX
// frontend and replaces it with a single fused nodes::DynamicQuantize op
// (one HIP kernel instead of ~8 ReduceMin/Max/Sub/Div/Round/Clamp/Convert kernels).

#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class DynamicQuantizeFusionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("DynamicQuantizeFusionPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
