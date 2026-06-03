// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rocm_operation_base.hpp"
#include "kernels/interpolate_cubic.hpp"
#include "openvino/op/interpolate.hpp"

namespace ov {
namespace rocm_gpu {

class InterpolateCubicOp : public OperationBase {
public:
    using NodeOp = ov::op::v4::Interpolate;

    InterpolateCubicOp(const CreationContext& context,
                       const NodeOp& stridedSliceOp,
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
    std::optional<kernel::InterpolateCubic> interpolate_;
};

}  // namespace rocm_gpu
}  // namespace ov
