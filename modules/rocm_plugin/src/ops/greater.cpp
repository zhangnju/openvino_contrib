// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "greater.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

GreaterOp::GreaterOp(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : Comparison(context, node, std::move(inputIds), std::move(outputIds), kernel::Comparison::Op_t::GREATER) {}

OPERATION_REGISTER(GreaterOp, Greater);

}  // namespace rocm_gpu
}  // namespace ov
