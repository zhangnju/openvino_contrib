// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/floor_mod.hpp"
#include "openvino/op/floor_mod.hpp"

namespace ov {
namespace rocm_gpu {

class FloorModOp : public ElementwiseBinaryOp<ov::op::v1::FloorMod, kernel::FloorMod> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
