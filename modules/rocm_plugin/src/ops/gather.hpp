// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>
#include <kernels/gather.hpp>

namespace ov {
namespace rocm_gpu {

class GatherOp : public OperationBase {
public:
    GatherOp(const CreationContext& context,
             const ov::Node& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    std::optional<kernel::Gather> gather_kernel_;
};

}  // namespace rocm_gpu
}  // namespace ov
