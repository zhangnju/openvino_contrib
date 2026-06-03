// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file template_itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace ov {
namespace rocm_gpu {
namespace itt {
namespace domains {
OV_ITT_DOMAIN(rocm_gpu);
}
}  // namespace itt
}  // namespace rocm_gpu
}  // namespace ov
