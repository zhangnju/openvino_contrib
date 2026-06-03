// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "activation_forward_miopen_base.hpp"

namespace ov {
namespace rocm_gpu {

class TanhOp : public ActivationForwardmiopenOpBase {
public:
    TanhOp(const CreationContext& context,
           const std::shared_ptr<ov::Node>& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds);
};

}  // namespace rocm_gpu
}  // namespace ov
