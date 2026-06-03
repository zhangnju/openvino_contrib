// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

#include "elementwise_binary.hpp"
#include "kernels/prelu.hpp"
#include "openvino/op/prelu.hpp"

namespace ov {
namespace rocm_gpu {

class PReluOp : public ElementwiseBinaryOp<ov::op::v0::PRelu, kernel::PRelu> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
