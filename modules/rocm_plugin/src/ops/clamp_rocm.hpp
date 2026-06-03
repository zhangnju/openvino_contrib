// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>
#include <openvino/op/clamp.hpp>

#include "kernels/clamp.hpp"

namespace ov {
namespace rocm_gpu {

class ClamprocmOp : public OperationBase {
public:
    using NodeOp = ov::op::v0::Clamp;

    ClamprocmOp(const CreationContext& context,
                const NodeOp& node,
                IndexCollection&& inputIds,
                IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    std::optional<kernel::Clamp> kernel_;
};

}  // namespace rocm_gpu
}  // namespace ov
