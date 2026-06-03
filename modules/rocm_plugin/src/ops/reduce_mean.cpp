// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_operation_registry.hpp"
#include "reduce_mean.hpp"

namespace ov {
namespace rocm_gpu {

ReduceMeanOp::ReduceMeanOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, move(inputIds), move(outputIds), rocm::DnnReduceAvgDescriptor(reduceCompType(node))) {}

OPERATION_REGISTER(ReduceMeanOp, ReduceMean);

}  // namespace rocm_gpu
}  // namespace ov
