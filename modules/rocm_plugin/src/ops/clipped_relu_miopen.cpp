// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clipped_relu_miopen.hpp"

#include <fmt/format.h>

#include <rocm/dnn.hpp>
#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

ClippedRelumiopenOp::ClippedRelumiopenOp(const CreationContext& context,
                                       const NodeOp& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds)
    : ActivationForwardmiopenOpBase{std::make_unique<rocm::ClippedReluDescriptor>(node.get_max()),
                                   context,
                                   node,
                                   std::move(inputIds),
                                   std::move(outputIds)} {
    const auto min = node.get_min();
    const auto max = node.get_max();
    if (min != 0.0) {
        throw_ov_exception(fmt::format("ov::rocm_gpu::ClippedRelumiopenOp: Clamp min != 0.0, min = {}", min));
    }
    if (max < 0.0) {
        throw_ov_exception(fmt::format("ov::rocm_gpu::ClippedRelumiopenOp: Clamp max < 0.0, max = {}", max));
    }
}

}  // namespace rocm_gpu
}  // namespace ov
