// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "relu.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

ReluOp::ReluOp(const CreationContext& context,
               const std::shared_ptr<ov::Node>& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : ActivationForwardmiopenOpBase{
          std::make_unique<rocm::ReluDescriptor>(), context, *node, std::move(inputIds), std::move(outputIds)} {}

OPERATION_REGISTER(ReluOp, Relu);
}  // namespace rocm_gpu
}  // namespace ov
