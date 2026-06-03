// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nop_op.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

OPERATION_REGISTER(NopOp, Constant);
OPERATION_REGISTER(NopOp, Reshape);
OPERATION_REGISTER(NopOp, Squeeze);
OPERATION_REGISTER(NopOp, Unsqueeze);
OPERATION_REGISTER(NopOp, ConcatOptimized);

}  // namespace rocm_gpu
}  // namespace ov
