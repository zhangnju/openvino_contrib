// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/device_pointers.hpp>
#include <rocm_operation_base.hpp>
#include <transformer/nodes/fully_connected.hpp>

#include "rocm/constant_factory.hpp"
#include "rocm/rocmlir_gemm.hpp"
#include "openvino/op/matmul.hpp"
#include <memory>
#include <string>

namespace ov {
namespace rocm_gpu {

class MatMulOp : public OperationRocBlas {
public:
    using NodeOp = ov::op::v0::MatMul;

    template <typename TOperation>
    MatMulOp(const CreationContext& context,
             const TOperation& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

    int GetBatchCount()  const { return batch_count_; }
    int GetM()           const { return m_; }
    int GetN()           const { return n_; }
    int GetK()           const { return k_; }
    rocblas_operation GetTransposeB() const { return rocblas_transpose_b_; }

    /**
     * Get number of batches that equals to product between dimensions in range [matrixShape.begin(),
     * matrixShape.end()-2)
     * @param matrixShape Matrix to calculate number of batches
     * @return Number of batches
     */
    static int GetMatrixNumBatches(const ov::Shape& matrixShape);
    /**
     * Broadcast some shape to Matrix
     * For example:
     * {} -> {1, 1}
     * {2} -> {1, 2}
     * {3, 2} -> {3, 2} // Not changed
     * @param shape Shape to broadcast to matrix
     */
    static void BroadcastToMatrix(ov::Shape& shape);

    /**
     * Get compute type according A/B matrix data type and C matrix data type
     * @param abDataType A/B matrix data type
     * @param cDataType C matrix data type
     * @return Available compute type
     */
    static hipDataType GetComputeType(hipDataType abDataType, hipDataType cDataType);

private:
    /**
     * Broadcast input shapes according OpenVINO documentation:
     * @reference https://docs.openvinotoolkit.org/latest/openvino_docs_ops_matrix_MatMul_1.html
     * @param matrixAShape Shape of matrix A
     * @param matrixBShape Shape of matrix B
     * @param matrixCShape Shape of matrix C
     */
    static void BroadcastShapes(
        ov::Shape& matrixAShape, bool& transposeA, ov::Shape& matrixBShape, bool& transposeB, ov::Shape& matrixCShape);

    hipDataType data_type_ = hipDataType::HIP_R_32F;
    hipDataType compute_type_ = hipDataType::HIP_R_32F;
    int m_ = 0;
    int k_ = 0;
    int n_ = 0;
    int ld_a_ = 0;
    int ld_b_ = 0;
    int ld_c_ = 0;
    long long stride_a_ = 0;
    long long stride_b_ = 0;
    long long stride_c_ = 0;
    int batch_count_ = 0;
    const rocm::constants::AnyNumeric* beta_ = nullptr;
    rocblas_operation rocblas_transpose_a_ = rocblas_operation_none;
    rocblas_operation rocblas_transpose_b_ = rocblas_operation_none;

    // Optional tuned rocMLIR GEMM for plain 2D fp16 (e.g. BERT FFN-down/attn-out).
    // Decided at construction by tuning-cache presence; Execute is pure dispatch.
    // A one-time numeric check vs rocBLAS guards correctness (generic op).
    mutable std::shared_ptr<rocmlir_gemm::GemmKernel> rocmlir_kernel_;
    mutable bool use_rocmlir_ = false;
    mutable bool rocmlir_checked_ = false;
    std::string arch_;
    int num_cu_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
