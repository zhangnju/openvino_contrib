// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce.hpp"

namespace ov {
namespace rocm_gpu {

class ReduceMeanOp : public ReduceOp {
public:
    explicit ReduceMeanOp(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds);
};

}  // namespace rocm_gpu
}  // namespace ov
