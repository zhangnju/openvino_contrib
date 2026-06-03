// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "kernels/details/error.hpp"

namespace ov {
namespace rocm_gpu {
[[gnu::cold, noreturn]] void throw_ov_exception(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());
[[gnu::cold]] void logError(
    const std::string& msg,
    const std::experimental::source_location& location = std::experimental::source_location::current());
}  // namespace rocm_gpu
}  // namespace ov
