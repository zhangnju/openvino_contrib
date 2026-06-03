// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/add.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace rocm_gpu {

class AddrocmOp : public ElementwiseBinaryOp<ov::op::v1::Add, kernel::Add> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
