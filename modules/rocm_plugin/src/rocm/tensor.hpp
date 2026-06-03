// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
/*
#include <hiptensor/hiptensor.h>

#include "runtime.hpp"

inline void throwIfError(
    cutensorStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUTENSOR_STATUS_SUCCESS) ov::rocm_gpu::throw_ov_exception(cutensorGetErrorString(err), location);
}

inline void logIfError(
    cutensorStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUTENSOR_STATUS_SUCCESS) ov::rocm_gpu::logError(cutensorGetErrorString(err), location);
}

namespace rocm {

class CuTensorHandle : public Handle<cutensorHandle_t> {
public:
    CuTensorHandle() : Handle(cutensorInit, nullptr) {}
};

}  // namespace rocm
*/