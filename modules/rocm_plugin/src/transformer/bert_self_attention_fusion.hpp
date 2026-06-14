// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Detects BERT self-attention pattern and replaces with a single fused kernel.
// Eliminates: Reshape+Transpose (×4), scale Multiply, Q×K^T MatMul,
//             mask Add, Softmax, A×V MatMul, context Transpose+Reshape
// All replaced by one rocMLIR rock.attention kernel.

#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class BertSelfAttentionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("BertSelfAttentionFusion", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
