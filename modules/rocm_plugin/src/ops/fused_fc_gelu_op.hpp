// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedFCGELUOp: FullyConnected + GELU in two passes:
//   1. rocBLAS GEMM:  gemm_out = x[seq,in] × W[in,out]  (+ bias applied in pass 2)
//   2. Native HIP:    out = GELU(gemm_out + bias)
//
// Inputs: [0]=x[seq,in_dim]  [1]=W[in_dim,out_dim]  [2]=bias[out_dim]
// Output: [0]=GELU(x×W+bias)[seq,out_dim]
#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fused_fc_gelu_node.hpp>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {

class FusedFCGELUOp : public OperationBase {
public:
    FusedFCGELUOp(const CreationContext& context,
                  const std::shared_ptr<ov::Node>& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds);
    ~FusedFCGELUOp();

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

    WorkbufferRequest GetWorkBufferRequest() const override { return {}; }

private:
    int   seq_{0};
    int   in_dim_{0};
    int   out_dim_{0};

    // Pre-allocated intermediate buffer: GEMM output before bias+GELU
    void* gemm_buf_{nullptr};
    size_t gemm_buf_bytes_{0};

    rocblas_operation trans_x_{rocblas_operation_none};
    rocblas_operation trans_w_{rocblas_operation_transpose};
};

}  // namespace rocm_gpu
}  // namespace ov
