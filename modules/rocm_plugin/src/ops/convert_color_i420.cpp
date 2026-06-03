// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_i420.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

OPERATION_REGISTER(I420toRGBOp, I420toRGB);
OPERATION_REGISTER(I420toBGROp, I420toBGR);

}  // namespace rocm_gpu
}  // namespace ov
