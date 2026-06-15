// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedQKVProjectionPass: detects 3 FullyConnected nodes feeding Q, K, V into
// BertSelfAttention where all 3 share the same primary input x.
// Merges them into a single FusedQKVProjection node with combined weights [3H, H].
// BertSelfAttentionOp then uses the combined QKV buffer via ptr+offset, avoiding
// all D2D copy overhead and replacing 3×256x768x768 GEMMs with 1×256x2304x768.
//
// Must run AFTER BertSelfAttentionFusion (which creates BertSelfAttention nodes).
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class FusedQKVProjectionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FusedQKVProjectionPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
