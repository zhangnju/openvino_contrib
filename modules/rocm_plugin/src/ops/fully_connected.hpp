// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/device_pointers.hpp>
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fully_connected.hpp>

#include "rocm/constant_factory.hpp"
#include "matmul.hpp"

namespace ov {
namespace rocm_gpu {

class FullyConnectedOp : public OperationRocBlas {
public:
    using NodeOp = nodes::FullyConnected;
    FullyConnectedOp(const CreationContext& context,
                     const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    MatMulOp matmul_op_;
    size_t bias_size_ = 0;
    size_t batch_bias_count_ = 0;
    size_t bias_cols_ = 0;  // number of bias elements (= output cols)
    size_t bias_rows_ = 0;  // number of rows to broadcast bias over
    bool   is_fp16_   = false;
};

}  // namespace rocm_gpu
}  // namespace ov
