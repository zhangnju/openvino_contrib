// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/multiply.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace rocm_gpu {

using MultiplyrocmOpBase = ElementwiseBinaryOp<ov::op::v1::Multiply, kernel::Multiply>;
class MultiplyrocmOp : public MultiplyrocmOpBase {
public:
    using NodeOp = ov::op::v1::Multiply;
    MultiplyrocmOp(const CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);
};

}  // namespace rocm_gpu
}  // namespace ov
