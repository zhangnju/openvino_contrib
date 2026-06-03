// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_operation_registry.hpp"
#include "reduce_max.hpp"

namespace ov {
namespace rocm_gpu {

ReduceMaxOp::ReduceMaxOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, move(inputIds), move(outputIds), rocm::DnnReduceMaxDescriptor(reduceCompType(node))) {}

OPERATION_REGISTER(ReduceMaxOp, ReduceMax);

}  // namespace rocm_gpu
}  // namespace ov
