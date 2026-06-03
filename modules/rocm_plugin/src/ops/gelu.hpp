// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/gelu.hpp"
#include "openvino/op/gelu.hpp"

namespace ov {
namespace rocm_gpu {

class GeluOp : public ElementwiseUnaryOp<ov::op::v0::Gelu, kernel::GeluErf> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

class GeluErfOp : public ElementwiseUnaryOp<ov::op::v7::Gelu, kernel::GeluErf> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

class GeluTanhOp : public ElementwiseUnaryOp<ov::op::v7::Gelu, kernel::GeluTanh> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace rocm_gpu
}  // namespace ov
