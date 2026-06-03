// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "add_miopen.hpp"

#include <rocm_operation_registry.hpp>

namespace ov {
namespace rocm_gpu {

AddmiopenOp::AddmiopenOp(const CreationContext& context,
                       const std::shared_ptr<ov::Node>& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds)
    : miopenTensorOpBase{context, node, std::move(inputIds), std::move(outputIds), miopenTensorOp_t::miopenTensorOpAdd} {}

}  // namespace rocm_gpu
}  // namespace ov
