// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>
#include <openvino/op/swish.hpp>

#include "kernels/swish.hpp"

namespace ov {
namespace rocm_gpu {

class SwishOp : public OperationBase {
public:
    SwishOp(const CreationContext& context,
            const ov::Node& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    std::optional<kernel::Swish> kernel_;
};

}  // namespace rocm_gpu
}  // namespace ov
