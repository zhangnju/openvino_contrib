// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rocm_operation_base.hpp"
#include "kernels/gather_nd.hpp"

namespace ov {
namespace rocm_gpu {

class GatherNDOp : public OperationBase {
public:
    GatherNDOp(const CreationContext& context,
               const ov::Node& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

private:
    std::optional<kernel::GatherND> kernel_;
    std::vector<size_t> data_dim_padding_;
};

}  // namespace rocm_gpu
}  // namespace ov
