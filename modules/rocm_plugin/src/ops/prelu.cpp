// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prelu.hpp"

#include "rocm_operation_registry.hpp"

namespace ov {
namespace rocm_gpu {

OPERATION_REGISTER(PReluOp, PRelu)

}  // namespace rocm_gpu
}  // namespace ov
