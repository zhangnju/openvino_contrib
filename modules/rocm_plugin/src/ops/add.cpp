// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <openvino/core/except.hpp>
#include <sstream>

#include <hip/hip_runtime.h>
#include "add_miopen.hpp"
#include "add_rocm.hpp"
#include "rocm_operation_registry.hpp"

namespace ov {
namespace rocm_gpu {

static OperationBase::Ptr addFactory(const CreationContext& context,
                                     const std::shared_ptr<ov::Node>& in_node,
                                     OperationBase::IndexCollection&& inputIds,
                                     OperationBase::IndexCollection&& outputIds) {
    auto node = std::dynamic_pointer_cast<ov::op::v1::Add>(in_node);
    OPENVINO_ASSERT(node);

    const OperationBase::IndexCollection inputs{inputIds};
    const OperationBase::IndexCollection outputs{outputIds};

    std::stringstream exception_msg;
    try {
        return std::make_shared<AddmiopenOp>(
            context, node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "Failed to create Addmiopen impl: " << e.what();
    }
    try {
        return std::make_shared<AddrocmOp>(
            context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create Addrocm impl: " << e.what();
    }
    throw_ov_exception(fmt::format("Add node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(addFactory, Add)

}  // namespace rocm_gpu
}  // namespace ov
