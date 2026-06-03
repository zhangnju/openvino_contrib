// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_runtime.h>

#include <fmt/ostream.h>
#include "multiply_rocm.hpp"
namespace ov {
namespace rocm_gpu {

MultiplyrocmOp::MultiplyrocmOp(const CreationContext& context,
                               const NodeOp& node,
                               IndexCollection&& inputIds,
                               IndexCollection&& outputIds)
    : MultiplyrocmOpBase{context, node, std::move(inputIds), std::move(outputIds)} {
    const auto broatcast_type = node.get_autob().m_type;
    switch (broatcast_type) {
        case ov::op::AutoBroadcastType::NONE:
        case ov::op::AutoBroadcastType::NUMPY:
            break;
        default:
            throw_ov_exception(fmt::format("MultiplyrocmOp: unsupported broadcast type: {}", broatcast_type));
    }
}

}  // namespace rocm_gpu
}  // namespace ov
