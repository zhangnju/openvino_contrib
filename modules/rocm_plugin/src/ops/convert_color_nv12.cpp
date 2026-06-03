// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_nv12.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

OPERATION_REGISTER(NV12toRGBOp, NV12toRGB);
OPERATION_REGISTER(NV12toBGROp, NV12toBGR);

}  // namespace rocm_gpu
}  // namespace ov
