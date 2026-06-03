// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_operation_registry.hpp"
#include "reduce_prod.hpp"

namespace ov {
namespace rocm_gpu {

ReduceProdOp::ReduceProdOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, std::move(inputIds), std::move(outputIds), rocm::DnnReduceMulDescriptor(reduceCompType(node))) {}

OPERATION_REGISTER(ReduceProdOp, ReduceProd);

}  // namespace rocm_gpu
}  // namespace ov
