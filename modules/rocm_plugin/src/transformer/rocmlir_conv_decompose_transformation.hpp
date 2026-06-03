// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RocMLIRConvDecompose: converts standard Convolution (R>1 or S>1) into an
// equivalent K×C-group GroupConvolution to work around a rocMLIR 7.2 bug on
// gfx950 (MI350) where large-kernel convolutions with non-zero data produce
// GPU memory faults.
//
// Transformation (mathematically equivalent):
//   Conv(input[N,C,H,W], filter[K,C,R,S], stride, pad)
//     → Reshape(filter,  [K*C, 1, R, S])               // group filter
//     → Tile(input, K)   → [N, K*C, H, W]               // broadcast input
//     → GroupConvolution(groups=K*C)  → [N, K*C, OH, OW]
//     → Reshape           → [N, K, C, OH, OW]
//     → ReduceSum(axis=2) → [N, K, OH, OW]              // accumulate over C
//
// Reference: MIGraphX fuse_mlir pass generates identical rock.conv IR by
// setting G=K*C groups in the Rock dialect, which uses stable tiling paths.

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::rocm_gpu::pass {

class RocMLIRConvDecompose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RocMLIRConvDecompose", "0");
    RocMLIRConvDecompose();
};

class RocMLIRGroupConvDecompose : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RocMLIRGroupConvDecompose", "0");
    RocMLIRGroupConvDecompose();
};

} // namespace ov::rocm_gpu::pass
