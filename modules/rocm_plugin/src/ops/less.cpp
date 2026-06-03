// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "less.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

LessOp::LessOp(const CreationContext& context,
               const ov::Node& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : Comparison(context, node, std::move(inputIds), std::move(outputIds), kernel::Comparison::Op_t::LESS) {}

OPERATION_REGISTER(LessOp, Less);

}  // namespace rocm_gpu
}  // namespace ov
