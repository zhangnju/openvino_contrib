// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sigmoid.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

SigmoidOp::SigmoidOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : ActivationForwardmiopenOpBase{
          std::make_unique<rocm::SigmoidDescriptor>(), context, *node, std::move(inputIds), std::move(outputIds)} {}

OPERATION_REGISTER(SigmoidOp, Sigmoid);
}  // namespace rocm_gpu
}  // namespace ov
