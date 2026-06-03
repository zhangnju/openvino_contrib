// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile.hpp"
#include <rocm_operation_registry.hpp>
#include "kernels/tile.hpp"

namespace ov {
namespace rocm_gpu {

TileOp::TileOp(const CreationContext& context,
               const std::shared_ptr<ov::Node>& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto in_shape = node->get_input_shape(0);
    auto out_shape = node->get_output_shape(0);
    ndim_ = static_cast<int32_t>(out_shape.size());
    for (auto d : in_shape) in_shape_.push_back(static_cast<int64_t>(d));
    for (auto d : out_shape) out_shape_.push_back(static_cast<int64_t>(d));
    element_size_ = node->get_input_element_type(0).size();
}

void TileOp::Execute(const InferenceRequestContext& context,
                     Inputs inputTensors,
                     Outputs outputTensors,
                     const Workbuffers&) const {
    OPENVINO_ASSERT(inputTensors.size() >= 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());

    kernel::launchTile(inputTensors[0].get(),
                       outputTensors[0].get(),
                       in_shape_.data(),
                       out_shape_.data(),
                       ndim_,
                       element_size_,
                       context.getThreadContext().stream().get());
}

OPERATION_REGISTER(TileOp, Tile);

}  // namespace rocm_gpu
}  // namespace ov
