// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>
#include <vector>

namespace ov {
namespace rocm_gpu {

class TileOp : public OperationBase {
public:
    TileOp(const CreationContext& context,
           const std::shared_ptr<ov::Node>& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    std::vector<int64_t> in_shape_;
    std::vector<int64_t> out_shape_;
    size_t element_size_;
    int32_t ndim_;
};

}  // namespace rocm_gpu
}  // namespace ov
