// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <hip/hip_runtime.h>

namespace ov { namespace rocm_gpu { namespace kernel {

void launch_bias_add(void* output, const void* bias,
                      int N, int K, int H, int W,
                      bool fp16, hipStream_t stream);

} } }
