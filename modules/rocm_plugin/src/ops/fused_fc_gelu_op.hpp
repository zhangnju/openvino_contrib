// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedFCGELUOp: GEMM + bias + GELU in a single hipBLASLt kernel.
// Uses HIPBLASLT_EPILOGUE_GELU_BIAS (value=36) — same conv as FullyConnectedOp.
// Falls back to rocBLAS + native HIP bias+GELU if hipBLASLt unavailable.
//
// Note: hipBLASLt GELU uses the tanh approximation; our fallback uses the erf
// form. The difference is within FP16 quantization noise for BERT inference.
//
// Inputs: [0]=x[seq,in_dim]  [1]=W[out_dim,in_dim]  [2]=bias[out_dim]
// Output: [0]=GELU(x×W^T+bias)[seq,out_dim]
#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fused_fc_gelu_node.hpp>
#include "rocm/rocmlir_gemm.hpp"
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <memory>

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

    // ── hipBLASLt GEMM+bias+GELU path (single kernel) ────────────────────────
    hipblasLtHandle_t       lt_handle_{nullptr};
    hipblasLtMatmulDesc_t   lt_desc_{nullptr};
    hipblasLtMatrixLayout_t lt_layout_W_{nullptr};
    hipblasLtMatrixLayout_t lt_layout_X_{nullptr};
    hipblasLtMatrixLayout_t lt_layout_D_{nullptr};
    hipblasLtMatmulAlgo_t   lt_algo_{};
    void*                   lt_workspace_{nullptr};
    size_t                  lt_workspace_bytes_{0};
    bool                    use_hipblaslt_{false};
    mutable bool            lt_tuned_{false};
    std::string             arch_;

    // ── Fallback: rocMLIR or rocBLAS + native GELU kernel ────────────────────
    std::shared_ptr<rocmlir_gemm::GemmKernel> rocmlir_kernel_;
    void*  gemm_buf_{nullptr};
    size_t gemm_buf_bytes_{0};
};

}  // namespace rocm_gpu
}  // namespace ov
