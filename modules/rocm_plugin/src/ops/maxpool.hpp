// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

#include "pooling_impl.hpp"

namespace ov {
namespace rocm_gpu {

class MaxPoolOp : public OperationMIOPEN {
public:
    explicit MaxPoolOp(const CreationContext& context,
                       const std::shared_ptr<ov::Node>& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    PoolingImpl impl_;
};

}  // namespace rocm_gpu
}  // namespace ov
