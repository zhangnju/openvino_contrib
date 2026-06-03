// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "comparison.hpp"

namespace ov {
namespace rocm_gpu {

class GreaterEqualOp : public Comparison {
public:
    GreaterEqualOp(const CreationContext& context,
                   const ov::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);
};

}  // namespace rocm_gpu
}  // namespace ov
