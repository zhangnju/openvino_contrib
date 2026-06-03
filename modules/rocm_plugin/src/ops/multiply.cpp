// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <openvino/core/except.hpp>
#include <sstream>

#include "rocm_operation_registry.hpp"
#include <hip/hip_runtime.h>
#include "multiply_miopen.hpp"
#include "multiply_rocm.hpp"

namespace ov {
namespace rocm_gpu {

static OperationBase::Ptr multiplyFactory(const CreationContext& context,
                                          const std::shared_ptr<ov::Node>& in_node,
                                          OperationBase::IndexCollection&& inputIds,
                                          OperationBase::IndexCollection&& outputIds) {
    auto node = std::dynamic_pointer_cast<ov::op::v1::Multiply>(in_node);
    OPENVINO_ASSERT(node);

    const OperationBase::IndexCollection inputs{inputIds};
    const OperationBase::IndexCollection outputs{outputIds};

    std::stringstream exception_msg;
    try {
        return std::make_shared<MultiplymiopenOp>(
            context, node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "Failed to create Multiplymiopen impl: " << e.what();
    }
    try {
        return std::make_shared<MultiplyrocmOp>(
            context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create Multiplyrocm impl: " << e.what();
    }
    throw_ov_exception(fmt::format("Multiply node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(multiplyFactory, Multiply)

}  // namespace rocm_gpu
}  // namespace ov
