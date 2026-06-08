// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RocmAttentionFusionPass: replaces the AAttn (Area Attention) pattern
//   Reshape → Split(Q,K,V) → Transpose(Q) → MatMul(Q^T,K) → ... → MatMul(V,softmax)
// with RocmAttentionMatMul ops that call MIGraphX-compiled MLIR kernels.
//
// Controlled by ROCM_FUSE_ATTENTION env var (default: enabled).

#pragma once
#include <openvino/pass/pass.hpp>

namespace ov::rocm_gpu::pass {

class RocmAttentionFusionPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RocmAttentionFusionPass", "0");
    // arch: GPU arch string (e.g. "gfx950", "gfx1201").
    // pe_fusion: enable pe(V) conv fusion (only stable on gfx1201).
    explicit RocmAttentionFusionPass(const std::string& arch = "", bool pe_fusion = false)
        : arch_(arch), pe_fusion_(pe_fusion) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
private:
    std::string arch_;
    bool pe_fusion_ = false;
};

}  // namespace ov::rocm_gpu::pass
