// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "variadic_split_alias.hpp"
#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

OPERATION_REGISTER(VariadicSplitAliasOp, VariadicSplitAlias);

}  // namespace rocm_gpu
}  // namespace ov
