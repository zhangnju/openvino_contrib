// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Fuses Reshape+Transpose+MatMul in BERT attention into a stride-aware GEMM
// that avoids materializing the transposed tensor.
//
// Pattern matched:
//   input[1,256,768] → Reshape[1,256,12,64] → Transpose[0,2,1,3] → MatMul
//
// Replaced by:
//   MatMulStridedOp with lda=768, stride_a=64, batch_count=12
//   operating directly on the input[1,256,768] buffer
//
// This eliminates 61 Transpose kernel launches (1.06ms) in BERT inference.
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class BertAttentionTransposeFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("BertAttentionTransposeFusion", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
