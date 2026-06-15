// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RocmDCE: Dead Code Elimination pass for OV ROCm Plugin.
//
// OV's built-in NopElimination only removes identity/passthrough ops.
// This pass does proper backward-reachability DCE: nodes not reachable
// backward from any Result are dead and can be removed.
//
// Required for residual+LayerNorm fusion: LayerNormFusionPass replaces
// the TF-style LN output with a LayerNorm node, but leaves the internal
// ReduceMean/Sub/Mul nodes alive. These "dead" nodes are still consumed
// by ElementwiseFusionPass, preventing FusedLayerNormPass from absorbing
// the residual Add into a single FusedLayerNorm kernel.
//
// After RocmDCE:
// - Dead TF-LN internal nodes are removed
// - residual Add has exactly 1 consumer (LayerNorm)
// - FusedLayerNormPass can fuse: Add(fc_out, residual) → FusedLayerNorm
// - Result: 25 single-kernel FusedLayerNorm (vs 26 FusedEW + 25 FusedLN)
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class RocmDCE : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("RocmDCE", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
