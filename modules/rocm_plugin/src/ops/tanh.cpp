// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "tanh.hpp"

#include <rocm/dnn.hpp>
#include <rocm_operation_registry.hpp>

#include "converters.hpp"
#include "rocm/constant_factory.hpp"

namespace ov {
namespace rocm_gpu {

TanhOp::TanhOp(const CreationContext& context,
               const std::shared_ptr<ov::Node>& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : ActivationForwardmiopenOpBase{
          std::make_unique<rocm::TanhDescriptor>(), context, *node, std::move(inputIds), std::move(outputIds)} {}

OPERATION_REGISTER(TanhOp, Tanh);
}  // namespace rocm_gpu
}  // namespace ov
