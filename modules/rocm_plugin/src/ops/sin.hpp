// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/sin.hpp"
#include "openvino/op/sin.hpp"

namespace ov {
namespace rocm_gpu {

class SinOp : public ElementwiseUnaryOp<ov::op::v0::Sin, kernel::Sin> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
