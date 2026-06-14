// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include <rocm/blas.hpp>
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/matmul.hpp>
#include <utility>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "converters.hpp"
#include "rocm/constant_factory.hpp"
#include "matmul.hpp"

namespace {
// Vectorized f16 bias broadcast: processes 2 elements per thread via __half2.
// cols must be even (BERT hidden=768/3072 always is). Falls back to scalar otherwise.
__global__ void broadcast_bias_add_h2(
        __half2* __restrict__ out_h2,
        const __half2* __restrict__ bias_h2,
        int rows, int cols2) {           // cols2 = cols/2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols2;
    if (idx >= total) return;
    int col2 = idx % cols2;             // pair index within a row
    out_h2[idx] = __hadd2(out_h2[idx], bias_h2[col2]);
}

// Scalar fallback for f32 or odd cols
template<typename T>
__global__ void broadcast_bias_add_scalar(T* out, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    out[idx] += bias[idx % cols];
}

void launch_broadcast_bias_add(hipStream_t stream, void* out, const void* bias,
                               int rows, int cols, bool is_fp16) {
    constexpr int THREADS = 256;
    if (is_fp16 && (cols % 2 == 0)) {
        int cols2  = cols / 2;
        int total2 = rows * cols2;
        int blocks = (total2 + THREADS - 1) / THREADS;
        hipLaunchKernelGGL(broadcast_bias_add_h2, blocks, THREADS, 0, stream,
            reinterpret_cast<__half2*>(out),
            reinterpret_cast<const __half2*>(bias),
            rows, cols2);
    } else if (is_fp16) {
        int blocks = (rows * cols + THREADS - 1) / THREADS;
        hipLaunchKernelGGL(broadcast_bias_add_scalar<__half>, blocks, THREADS, 0, stream,
            static_cast<__half*>(out), static_cast<const __half*>(bias), rows, cols);
    } else {
        int blocks = (rows * cols + THREADS - 1) / THREADS;
        hipLaunchKernelGGL(broadcast_bias_add_scalar<float>, blocks, THREADS, 0, stream,
            static_cast<float*>(out), static_cast<const float*>(bias), rows, cols);
    }
}
}  // namespace

namespace ov {
namespace rocm_gpu {

FullyConnectedOp::FullyConnectedOp(const CreationContext& context,
                                   const NodeOp& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationRocBlas(context, node, std::move(inputIds), std::move(outputIds)),
      matmul_op_{
          context, node, IndexCollection{input_ids_.begin(), input_ids_.end() - 1}, IndexCollection(output_ids_)} {
    bias_size_ = node.get_input_tensor(2).size();
    auto biasShape = node.get_input_shape(2);
    auto matrixShape = node.get_output_shape(0);
    OPENVINO_ASSERT(biasShape.size() > 0, "Node name: ", GetName());
    MatMulOp::BroadcastToMatrix(biasShape);
    const auto biasShapeSize = ov::shape_size(biasShape);
    const auto matrixShapeSize = ov::shape_size(matrixShape);
    OPENVINO_ASSERT(matrixShapeSize >= biasShapeSize, "Node name: ", GetName());
    auto batchBiasCount = MatMulOp::GetMatrixNumBatches(biasShape);
    auto matMulBatchCount = matmul_op_.GetBatchCount();
    OPENVINO_ASSERT(matMulBatchCount >= batchBiasCount, "Node name: ", GetName());
    batch_bias_count_ = matrixShapeSize / biasShapeSize;
    // Number of rows = total output elements / bias cols
    bias_cols_ = biasShapeSize;
    bias_rows_ = batch_bias_count_;  // rows to broadcast over
    is_fp16_ = (node.get_output_element_type(0) == ov::element::f16);
}

void FullyConnectedOp::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    // Step 1: GEMM: C = A*B (beta=0, no bias yet)
    matmul_op_.Execute(context, inputs.first(inputs.size() - 1), outputs, workbuffers);
    // Step 2: Broadcast-add bias: C[i,j] += bias[j]  (single kernel, no copy loop)
    auto& stream = context.getThreadContext().stream();
    launch_broadcast_bias_add(stream.get(), outputs[0].get(), inputs[2].get(),
                              static_cast<int>(bias_rows_), static_cast<int>(bias_cols_), is_fp16_);
}

rocmGraphCompatibility FullyConnectedOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

OPERATION_REGISTER(FullyConnectedOp, FullyConnected);
}  // namespace rocm_gpu
}  // namespace ov
