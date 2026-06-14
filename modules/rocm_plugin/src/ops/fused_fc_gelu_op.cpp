// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedFCGELUOp: rocBLAS GEMM + native HIP bias+GELU kernel.
// No MIGraphX API calls.

#include "fused_fc_gelu_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "kernels/fused_reduce.hpp"
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {

FusedFCGELUOp::FusedFCGELUOp(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {

    auto fc_gelu = std::dynamic_pointer_cast<nodes::FusedFCGELU>(node);
    OPENVINO_ASSERT(fc_gelu, "FusedFCGELUOp: expected FusedFCGELU node");

    seq_     = static_cast<int>(fc_gelu->get_seq());
    in_dim_  = static_cast<int>(fc_gelu->get_in_dim());
    out_dim_ = static_cast<int>(fc_gelu->get_out_dim());

    // Allocate intermediate buffer for GEMM output (f16)
    gemm_buf_bytes_ = (size_t)seq_ * out_dim_ * sizeof(__fp16);
    auto hip_err = hipMalloc(&gemm_buf_, gemm_buf_bytes_);
    OPENVINO_ASSERT(hip_err == hipSuccess,
        "FusedFCGELUOp: hipMalloc failed: ", hipGetErrorString(hip_err));
}

FusedFCGELUOp::~FusedFCGELUOp() {
    if (gemm_buf_) hipFree(gemm_buf_);
}

void FusedFCGELUOp::Execute(
        const InferenceRequestContext& context,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 3 && outputs.size() == 1);

    const void* x_ptr    = inputs[0].get();
    const void* W_ptr    = inputs[1].get();
    const void* bias_ptr = inputs[2].get();
    void*       out_ptr  = outputs[0].get();

    auto& handle = context.getThreadContext().rocBlasHandle();
    hipStream_t stream = context.getThreadContext().stream().get();
    rocblas_set_stream(handle.get(), stream);

    // ── Pass 1: GEMM  gemm_buf = x[seq,in] × W^T[out,in] → [seq,out] ─────────
    // rocBLAS is column-major: C = A×B → we compute C^T = B^T × A^T
    // Our row-major: C[seq,out] = x[seq,in] × W^T  ←→  col-major: C^T[out,seq] = W[out,in] × x^T[in,seq]
    // So: rocBLAS(N, T, out, seq, in, W, x, gemm_buf)
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    auto err = rocblas_gemm_ex(
        handle.get(),
        rocblas_operation_none,       // W is [out,in] in col-major → treat as-is
        rocblas_operation_transpose,  // x^T
        out_dim_, seq_, in_dim_,
        &alpha,
        W_ptr,       rocblas_datatype_f16_r, out_dim_,
        x_ptr,       rocblas_datatype_f16_r, in_dim_,
        &beta,
        gemm_buf_,   rocblas_datatype_f16_r, out_dim_,
        gemm_buf_,   rocblas_datatype_f16_r, out_dim_,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    OPENVINO_ASSERT(err == rocblas_status_success,
        "FusedFCGELUOp: rocblas_gemm_ex failed: ", err);

    // ── Pass 2: bias+GELU  out = GELU(gemm_buf + bias) ───────────────────────
    kernel::launch_bias_gelu_fused(
        stream,
        gemm_buf_, bias_ptr, out_ptr,
        seq_, out_dim_);
}

OPERATION_REGISTER(FusedFCGELUOp, FusedFCGELU);

}  // namespace rocm_gpu
}  // namespace ov
