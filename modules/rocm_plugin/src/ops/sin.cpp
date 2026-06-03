// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sin.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

OPERATION_REGISTER(SinOp, Sin);

}  // namespace rocm_gpu
}  // namespace ov
