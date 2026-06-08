// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::rocm_gpu::pass {

class rocmFuseMarkUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("rocmFuseMarkUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class rocmFuseCleanUpNodesOrder : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("rocmFuseCleanUpNodesOrder", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class FuseConvolutionWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseConvolutionWithBiasAdd", "0");
    FuseConvolutionWithBiasAdd();
};

class FuseGroupConvolutionWithBiasAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGroupConvolutionWithBiasAdd", "0");
    FuseGroupConvolutionWithBiasAdd();
};

class FuseConvolutionWithBiasAddAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseConvolutionWithBiasAddAdd", "0");
    FuseConvolutionWithBiasAddAdd();
};

class FuseGroupConvolutionWithBiasAddAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseGroupConvolutionWithBiasAddAdd", "0");
    FuseGroupConvolutionWithBiasAddAdd();
};

class SinkActivationToFusedConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SinkActivationToFusedConvolution", "0");
    SinkActivationToFusedConvolution();
};

// Fuses FusedConvolution → Swish → Add(skip) into FusedConvolution(4-inputs, SWISH).
// This enables rocMLIR to compile a single Conv+Bias+SiLU+SkipAdd kernel, eliminating
// two separate kernel launches (SwishOp + elementwise Add).
class SinkSwishAddToFusedConvolution : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SinkSwishAddToFusedConvolution", "0");
    SinkSwishAddToFusedConvolution();
};

class rocmConvolutionFusion : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("rocmConvolutionFusion", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class rocmFuseConvBackpropDataAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("rocmFuseConvBackpropDataAdd", "0");
    rocmFuseConvBackpropDataAdd();
};

}  // namespace ov::rocm_gpu::pass
