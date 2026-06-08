// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <rocm_operation_base.hpp>
#include <rocm_operation_registry.hpp>
#include <openvino/op/clamp.hpp>
#include <sstream>

#include <hip/hip_runtime.h>
//#include "clamp_miopen.hpp"
#include "clamp_rocm.hpp"
#include "clipped_relu_miopen.hpp"

namespace ov {
namespace rocm_gpu {

using IndexCollection = OperationBase::IndexCollection;

static OperationBase::Ptr clampFactory(const CreationContext& context,
                                       const std::shared_ptr<ov::Node>& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds) {
    const ov::op::v0::Clamp& node_op{downcast<const ov::op::v0::Clamp>(node)};

    const IndexCollection inputs{inputIds};
    const IndexCollection outputs{outputIds};

    std::stringstream exception_msg;
    try {
        return std::make_shared<ClamprocmOp>(context, node_op, IndexCollection{inputIds}, IndexCollection{outputIds});
    } catch (const std::exception& e) {
        exception_msg << "Failed to create ClamprocmOp implementation: " << e.what();
    }
    try {
        return std::make_shared<ClippedRelumiopenOp>(
            context, node_op, IndexCollection{inputIds}, IndexCollection{outputIds});
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create ClippedRelumiopenOp implementation: " << e.what();
    }
    throw_ov_exception(fmt::format("Clamp node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(clampFactory, Clamp)

}  // namespace rocm_gpu
}  // namespace ov
