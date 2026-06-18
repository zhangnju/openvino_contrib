// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedQKVProjectionOp: replaces 3×256×768×768 FC projections with a single
// 256×2304×768 hipBLASLt GEMM+bias kernel.
// Output: QKV_combined[seq, 3*hidden] — Q at [0], K at [hidden], V at [2*hidden]
// BertSelfAttentionOp reads Q, K, V via ptr+offset (no D2D copy).
#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fused_qkv_node.hpp>
#include <hipblaslt/hipblaslt.h>
#include "rocm/rocmlir_gemm.hpp"
#include <memory>

namespace ov {
namespace rocm_gpu {

class FusedQKVProjectionOp : public OperationBase {
public:
    FusedQKVProjectionOp(const CreationContext& context,
                         const std::shared_ptr<ov::Node>& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds);
    ~FusedQKVProjectionOp();

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

    WorkbufferRequest GetWorkBufferRequest() const override { return {}; }

private:
    int   seq_len_{0};
    int   hidden_{0};   // per-head group size (e.g. 768)

    // hipBLASLt GEMM+bias for 256×(3*768)×768
    hipblasLtHandle_t       lt_handle_{nullptr};
    hipblasLtMatmulDesc_t   lt_desc_{nullptr};
    hipblasLtMatrixLayout_t lt_layout_W_{nullptr};  // weight [3H, H]
    hipblasLtMatrixLayout_t lt_layout_X_{nullptr};  // input  [H, M]
    hipblasLtMatrixLayout_t lt_layout_D_{nullptr};  // output [3H, M]
    hipblasLtMatmulAlgo_t   lt_algo_{};
    void*                   lt_workspace_{nullptr};
    size_t                  lt_workspace_bytes_{0};

    // Lazy algo tuning
    mutable bool        lt_tuned_{false};
    std::string         arch_;
    int                 num_cu_{0};

    // Tuned fused rocMLIR GEMM+bias. Decided at construction by cache presence
    // (a tuned perf_config exists ⇒ use rocMLIR); Execute is pure dispatch, never
    // benchmarks. No tuned config ⇒ hipBLASLt (no regression).
    std::shared_ptr<rocmlir_gemm::GemmKernel> rocmlir_kernel_;
    bool                use_rocmlir_{false};
};

}  // namespace rocm_gpu
}  // namespace ov
