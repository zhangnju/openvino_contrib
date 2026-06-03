// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/squared_difference.hpp"
#include "openvino/op/squared_difference.hpp"

namespace ov {
namespace rocm_gpu {

class SquaredDifferenceOp : public ElementwiseBinaryOp<ov::op::v0::SquaredDifference, kernel::SquaredDifference> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
