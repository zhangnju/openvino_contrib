// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/device_pointers.hpp>
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fully_connected.hpp>

#include "rocm/constant_factory.hpp"
#include "rocm/rocmlir_gemm.hpp"
#include "matmul.hpp"
#include <hipblaslt/hipblaslt.h>

namespace ov {
namespace rocm_gpu {

class FullyConnectedOp : public OperationRocBlas {
public:
    using NodeOp = nodes::FullyConnected;
    FullyConnectedOp(const CreationContext& context,
                     const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);
    ~FullyConnectedOp();

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    MatMulOp matmul_op_;
    size_t bias_size_ = 0;
    size_t batch_bias_count_ = 0;
    size_t bias_cols_ = 0;
    size_t bias_rows_ = 0;
    bool   is_fp16_   = false;

    // ── hipBLASLt GEMM+bias path (fuses bias into GEMM epilogue) ──────────────
    // Eliminates the separate broadcast_bias_add kernel when available.
    hipblasLtHandle_t         lt_handle_{nullptr};
    hipblasLtMatmulDesc_t     lt_desc_{nullptr};
    hipblasLtMatrixLayout_t   lt_layout_B_{nullptr};  // weight matrix [N,K]
    hipblasLtMatrixLayout_t   lt_layout_A_{nullptr};  // input  matrix [K,M]
    hipblasLtMatrixLayout_t   lt_layout_D_{nullptr};  // output matrix [N,M]
    hipblasLtMatmulAlgo_t     lt_algo_{};
    void*                     lt_workspace_{nullptr};
    size_t                    lt_workspace_bytes_{0};
    bool                      use_hipblaslt_{false};
    bool                      lt_tuned_{false};       // algo tuning done on first Execute
    int                       lt_M_{0}, lt_N_{0}, lt_K_{0};
    bool                      lt_transB_{false};

    // ── rocMLIR GEMM-only backend (fallback when hipBLASLt unavailable) ────────
    std::shared_ptr<rocmlir_gemm::GemmKernel> rocmlir_kernel_;
    bool use_rocmlir_      = false;
    bool rocmlir_benchmarked_ = false;
    std::string arch_;
    int num_cu_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
