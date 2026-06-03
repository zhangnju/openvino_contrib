// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/power.hpp>

#include "elementwise_binary.hpp"
#include "kernels/power.hpp"

namespace ov {
namespace rocm_gpu {

class PowerOp : public ElementwiseBinaryOp<ov::op::v1::Power, kernel::Power> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
