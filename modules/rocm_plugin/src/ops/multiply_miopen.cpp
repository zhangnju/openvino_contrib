// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "multiply_miopen.hpp"

namespace ov {
namespace rocm_gpu {

MultiplymiopenOp::MultiplymiopenOp(const CreationContext& context,
                                 const std::shared_ptr<ov::Node>& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds)
    : miopenTensorOpBase{context, node, std::move(inputIds), std::move(outputIds), miopenTensorOp_t::miopenTensorOpMul} {}

}  // namespace rocm_gpu
}  // namespace ov
