// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::rocm_gpu::pass {

class FullyConnectedTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FullyConnectedTransformation", "0");
    FullyConnectedTransformation();
};

}  // namespace ov::rocm_gpu::pass
