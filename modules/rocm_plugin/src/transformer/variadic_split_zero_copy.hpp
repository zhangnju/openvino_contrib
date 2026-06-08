// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// VariadicSplitZeroCopyPass: replaces eligible VariadicSplit nodes with
// VariadicSplitAlias so the memory planner can alias their outputs into
// sub-ranges of the input buffer (zero-copy split).
//
// Eligibility conditions (conservative, safe to always apply):
//   1. axis == 1 (channel axis in NCHW) — splits are contiguous in memory
//   2. Input is 4D NCHW
//   3. All split lengths > 0 (no dynamic splits)
//
// Controlled by env var ROCM_ZEROCOPY_SPLIT=1 (default: enabled).

#pragma once

#include <openvino/pass/pass.hpp>

namespace ov::rocm_gpu::pass {

class VariadicSplitZeroCopyPass : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("VariadicSplitZeroCopyPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    // Returns true if zero-copy split is enabled via ROCM_ZEROCOPY_SPLIT env var.
    static bool isEnabled();
};

}  // namespace ov::rocm_gpu::pass
