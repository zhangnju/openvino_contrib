// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "rocm_operation_base.hpp"
#include "kernels/elu.hpp"
#include "openvino/op/elu.hpp"

namespace ov {
namespace rocm_gpu {

class EluOp : public OperationBase {
public:
    EluOp(const CreationContext& context,
          const ov::Node& node,
          IndexCollection&& inputIds,
          IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    std::optional<kernel::Elu> kernel_;
};

}  // namespace rocm_gpu
}  // namespace ov
