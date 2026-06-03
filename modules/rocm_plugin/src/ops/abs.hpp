// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/abs.hpp"
#include "openvino/op/abs.hpp"

namespace ov {
namespace rocm_gpu {

class AbsOp : public ElementwiseUnaryOp<ov::op::v0::Abs, kernel::Abs> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
