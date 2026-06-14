// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Detects FullyConnected → Gelu and fuses into FusedFCGELU kernel.
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class FusedFCGELUPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FusedFCGELUPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
