// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/clamp.hpp>

#include "activation_forward_miopen_base.hpp"

namespace ov {
namespace rocm_gpu {

class ClippedRelumiopenOp : public ActivationForwardmiopenOpBase {
public:
    using NodeOp = ov::op::v0::Clamp;

    ClippedRelumiopenOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);
};

}  // namespace rocm_gpu
}  // namespace ov
