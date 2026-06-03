// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "reduce.hpp"

namespace ov {
namespace rocm_gpu {

class ReduceProdOp : public ReduceOp {
public:
    explicit ReduceProdOp(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds);
};

}  // namespace rocm_gpu
}  // namespace ov
