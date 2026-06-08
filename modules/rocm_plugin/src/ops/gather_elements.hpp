// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>
#include <vector>

namespace ov {
namespace rocm_gpu {

class GatherElementsOp : public OperationBase {
public:
    GatherElementsOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);

    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    std::vector<int64_t> data_shape_;
    int32_t axis_;
    int32_t ndim_;
    size_t element_size_;
    size_t indices_element_size_;
    int64_t total_out_;
    std::vector<int64_t> out_strides_;
    std::vector<int64_t> data_strides_;
};

}  // namespace rocm_gpu
}  // namespace ov
