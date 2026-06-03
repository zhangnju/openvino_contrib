// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_plugin.hpp"

namespace ov {
namespace rocm_gpu {
namespace {

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_rocm_gpu_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace
}  // namespace rocm_gpu
}  // namespace ov
