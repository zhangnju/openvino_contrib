// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.hpp"

#include <rocblas/rocblas.h>
#include <rocm/blas.hpp>
#include <rocm/float16.hpp>
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <transformer/nodes/fully_connected.hpp>
#include <utility>

#include "converters.hpp"
#include "rocm/constant_factory.hpp"

namespace ov {
namespace rocm_gpu {

template <typename TOperation>
MatMulOp::MatMulOp(const CreationContext& context,
                   const TOperation& op,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationRocBlas(context, op, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(op.get_input_size() >= 2, "Node name: ", GetName());
    OPENVINO_ASSERT(op.get_output_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(convertDataType<hipDataType>(op.get_input_element_type(0)) ==
                        convertDataType<hipDataType>(op.get_input_element_type(1)),
                    "Node name: ",
                    GetName());
    data_type_ = convertDataType<hipDataType>(op.get_input_element_type(0));
    compute_type_ = GetComputeType(data_type_, convertDataType<hipDataType>(op.get_output_element_type(0)));
    auto inputAShape = op.get_input_shape(0);
    auto inputBShape = op.get_input_shape(1);
    auto outputCShape = op.get_output_shape(0);
    OPENVINO_ASSERT(inputAShape.size() > 0, "Node name: ", GetName());
    OPENVINO_ASSERT(inputBShape.size() > 0, "Node name: ", GetName());
    bool transposeA = op.get_transpose_a();
    bool transposeB = op.get_transpose_b();
    const int batchACount = GetMatrixNumBatches(inputAShape);
    const int batchBCount = GetMatrixNumBatches(inputBShape);
    BroadcastShapes(inputAShape, transposeA, inputBShape, transposeB, outputCShape);
    batch_count_ = std::max(batchACount, batchBCount);
    const size_t rowsA = *(inputAShape.end() - !transposeA - 1);
    const size_t colsA = *(inputAShape.end() - transposeA - 1);
    const size_t rowsB = *(inputBShape.end() - !transposeB - 1);
    const size_t colsB = *(inputBShape.end() - transposeB - 1);
    OPENVINO_ASSERT(colsA == rowsB, "Node name: ", GetName());
    m_ = rowsA;
    k_ = colsA;
    n_ = colsB;
    ld_a_ = *(inputAShape.end() - 1);
    ld_b_ = *(inputBShape.end() - 1);
    ld_c_ = *(outputCShape.end() - 1);
    stride_a_ = (batchACount > 1) ? (m_ * k_) : 0;
    stride_b_ = (batchBCount > 1) ? (k_ * n_) : 0;
    stride_c_ = (m_ * n_);
    #if 0
    rocblas_transpose_a_ = transposeA ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_transpose_b_ = transposeB ? rocblas_operation_transpose : rocblas_operation_none;
    #endif 
    if constexpr (std::is_same_v<TOperation, nodes::FullyConnected>) {
        beta_ = &rocm::NumericConst<rocm::constants::one>(compute_type_);
    } else {
        beta_ = &rocm::NumericConst<rocm::constants::zero>(compute_type_);
    }
    OPENVINO_ASSERT(m_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(k_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(n_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_a_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_b_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_c_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(batch_count_ != 0, "Node name: ", GetName());
}
template MatMulOp::MatMulOp(const CreationContext& context,
                            const ov::op::v0::MatMul&,
                            IndexCollection&&,
                            IndexCollection&&);
template MatMulOp::MatMulOp(const CreationContext& context,
                            const nodes::FullyConnected&,
                            IndexCollection&&,
                            IndexCollection&&);

hipDataType MatMulOp::GetComputeType(const hipDataType abDataType, const hipDataType cDataType) {
    constexpr auto SwitchCase = [](hipDataType a, hipDataType b) constexpr { return (a << 16) + b; };
    /**
     * NOTE: This switch is an implementation of CuBlas table for available compute types:
     * @reference https://docs.rocm.com/rocm/cublas/index.html#cublas-GemmStridedBatchedEx
     */
    switch (SwitchCase(abDataType, cDataType)) {
        case SwitchCase(HIP_R_16F, HIP_R_16F): {
            return HIP_R_16F;
        }
        case SwitchCase(HIP_R_8I, HIP_R_32I): {
            return HIP_R_32I;
        }
#ifdef rocm_HAS_BF16_TYPE
        case SwitchCase(HIP_R_16BF, HIP_R_16BF):
        case SwitchCase(HIP_R_16BF, HIP_R_32F):
#endif
        case SwitchCase(HIP_R_8I, HIP_R_32F):
        case SwitchCase(HIP_R_16F, HIP_R_32F):
        case SwitchCase(HIP_R_32F, HIP_R_32F):
        case SwitchCase(HIP_C_8I, HIP_C_32F):
        case SwitchCase(HIP_C_32F, HIP_C_32F): {
            return HIP_R_32F;
        }
        case SwitchCase(HIP_R_64F, HIP_R_64F):
        case SwitchCase(HIP_C_64F, HIP_C_64F): {
            return HIP_R_64F;
        }
        default:
            throw_ov_exception(
                fmt::format("Not supported combination of A and B types [{}] "
                            "with C type [{}]",
                            abDataType,
                            cDataType));
    }
}

int MatMulOp::GetMatrixNumBatches(const ov::Shape& matrixShape) {
    return matrixShape.size() >= 2
               ? std::accumulate(matrixShape.begin(), matrixShape.end() - 2, 1, std::multiplies<size_t>())
               : 1;
}

void MatMulOp::BroadcastShapes(
    ov::Shape& matrixAShape, bool& transposeA, ov::Shape& matrixBShape, bool& transposeB, ov::Shape& matrixCShape) {
    /**
     * NOTE: See NGraph documentation for broadcasting:
     * @reference https://docs.openvinotoolkit.org/latest/openvino_docs_ops_matrix_MatMul_1.html
     */
    if (matrixAShape.size() == 1 && matrixBShape.size() == 1) {
        // 1D x 1D: [X] x [X] -> [1, X] x [X, 1] -> [1, 1] => [] (scalar)
        matrixAShape = ov::Shape{1, matrixAShape[0]};
        matrixBShape = ov::Shape{matrixBShape[0], 1};
        transposeA = false;
        transposeB = false;
    } else if (matrixAShape.size() == 1 && matrixBShape.size() > 1) {
        // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
        matrixAShape = ov::Shape{1, matrixAShape[0]};
        transposeA = false;
    } else if (matrixAShape.size() > 1 && matrixBShape.size() == 1) {
        // ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]
        matrixBShape = ov::Shape{matrixBShape[0], 1};
        transposeB = false;
    } else if (matrixAShape.size() > 1 && matrixBShape.size() > 1) {
        // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
        auto broadcastNdToMd = [](const auto& shapeToBroadcast, auto& broadcastShape) {
            OPENVINO_ASSERT(shapeToBroadcast.size() >= broadcastShape.size());
            std::vector<size_t> newAxies;
            newAxies.reserve(shapeToBroadcast.size());
            newAxies.insert(newAxies.end(), shapeToBroadcast.begin(), shapeToBroadcast.end() - 2);
            newAxies.insert(newAxies.end(), broadcastShape.end() - 2, broadcastShape.end());
            broadcastShape = ov::Shape{newAxies};
        };
        const size_t batchA = GetMatrixNumBatches(matrixAShape);
        const size_t batchB = GetMatrixNumBatches(matrixBShape);
        if (batchA > batchB) {
            broadcastNdToMd(matrixAShape, matrixBShape);
        } else if (batchA < batchB) {
            broadcastNdToMd(matrixBShape, matrixAShape);
        }
        OPENVINO_ASSERT(GetMatrixNumBatches(matrixAShape) == GetMatrixNumBatches(matrixBShape));
    }
    OPENVINO_ASSERT(*(matrixAShape.end() - transposeA - 1) == *(matrixBShape.end() - !transposeB - 1));
    if (matrixAShape.size() > matrixBShape.size()) {
        matrixCShape = matrixAShape;
    } else {
        matrixCShape = matrixBShape;
    }
    *(matrixCShape.end() - 2) = *(matrixAShape.end() - !transposeA - 1);
    *(matrixCShape.end() - 1) = *(matrixBShape.end() - transposeB - 1);
}

void MatMulOp::BroadcastToMatrix(ov::Shape& shape) {
    if (shape.size() < 2) {
        shape.insert(shape.begin(), 2 - shape.size(), 1);
    }
}

// NOTE: Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n), C is stored as row-major matrix
void MatMulOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    auto& RocBlasHandle = context.getThreadContext().rocBlasHandle();
    auto matrixA = inputs[0];
    auto matrixB = inputs[1];
    auto matrixC = outputs[0];

    // Map hipDataType → rocblas_datatype
    auto to_rocblas_dt = [](hipDataType dt) -> rocblas_datatype {
        switch (dt) {
            case HIP_R_16F: return rocblas_datatype_f16_r;
            case HIP_R_32F: return rocblas_datatype_f32_r;
            case HIP_R_64F: return rocblas_datatype_f64_r;
            case HIP_R_8I:  return rocblas_datatype_i8_r;
            default:        return rocblas_datatype_f32_r;
        }
    };
    const rocblas_datatype dt       = to_rocblas_dt(data_type_);
    const rocblas_datatype ct       = to_rocblas_dt(compute_type_);

    // rocBLAS uses column-major internally.
    // We compute Ct = Bt × At so the result C is row-major.
    throwIfError(rocblas_gemm_strided_batched_ex(
        RocBlasHandle.get(),
        rocblas_transpose_b_,   // transa for Bt
        rocblas_transpose_a_,   // transb for At
        n_,                     // m  (rows of Bt / Ct)
        m_,                     // n  (cols of At / Ct)
        k_,                     // k  (inner dimension)
        &rocm::NumericConst<rocm::constants::one>(compute_type_),
        matrixB.get(), dt, ld_b_, stride_b_,  // B (acts as A in col-major call)
        matrixA.get(), dt, ld_a_, stride_a_,  // A (acts as B in col-major call)
        beta_,
        matrixC.get(), dt, ld_c_, stride_c_,  // C
        matrixC.get(), dt, ld_c_, stride_c_,  // D (same as C for in-place)
        batch_count_,
        ct,
        rocblas_gemm_algo_standard,
        0,   // solution_index
        0)); // flags
}

rocmGraphCompatibility MatMulOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

// See below (after closing namespace) for the attention-aware factory

}  // namespace rocm_gpu
}  // namespace ov

// ── Attention-aware MatMul factory (outside namespace) ───────────────────────
// When ROCM_FUSE_ATTENTION != "0" and a MatMul node has been tagged by
// RocmAttentionFusionPass (via rt_info["rocm_attn_kind"]), create
// RocmAttentionMatMulOp instead of standard rocBLAS MatMulOp.
// Set ROCM_FUSE_ATTENTION=0 to revert to standard rocBLAS (no regression).
#include "rocm_attention_matmul.hpp"
namespace {
ov::rocm_gpu::OperationBase::Ptr matmulAttentionFactory(
        const ov::rocm_gpu::CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        ov::rocm_gpu::OperationBase::IndexCollection&& inputIds,
        ov::rocm_gpu::OperationBase::IndexCollection&& outputIds)
{
    const auto& rt = node->get_rt_info();
    if (ov::rocm_gpu::RocmAttentionMatMulOp::isEnabled() && rt.count("rocm_attn_kind")) {
        try {
            return std::make_shared<ov::rocm_gpu::RocmAttentionMatMulOp>(
                context, *node,
                ov::rocm_gpu::OperationBase::IndexCollection{inputIds},
                ov::rocm_gpu::OperationBase::IndexCollection{outputIds});
        } catch (const std::exception& e) {
            std::cerr << "[AttnFusion] Fallback to rocBLAS MatMul: " << e.what() << "\n";
        }
    }
    // Cast to MatMul to access get_transpose_a/b (required by template)
    auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
    OPENVINO_ASSERT(matmul_node, "matmulAttentionFactory: node is not MatMul");
    return std::make_shared<ov::rocm_gpu::MatMulOp>(
        context, *matmul_node,
        ov::rocm_gpu::OperationBase::IndexCollection{inputIds},
        ov::rocm_gpu::OperationBase::IndexCollection{outputIds});
}
}  // anonymous namespace
namespace ov { namespace rocm_gpu {
OPERATION_REGISTER_FACTORY(matmulAttentionFactory, MatMul);
}}  // namespace ov::rocm_gpu
