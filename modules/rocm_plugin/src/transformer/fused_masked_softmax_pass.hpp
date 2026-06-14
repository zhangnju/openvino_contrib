// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Fuses Add(scores, mask) → Softmax into FusedMaskedSoftmax kernel.
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class FusedMaskedSoftmaxPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FusedMaskedSoftmaxPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
