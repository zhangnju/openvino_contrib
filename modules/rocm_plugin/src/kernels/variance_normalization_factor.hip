#include "hip/hip_runtime.h"
// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_fp16.h>
#include "rocm/math.hpp"
#include "kernels/variance_normalization_factor.hpp"

namespace ov {
namespace rocm_gpu {

namespace kernel {

// FP32 kernel: 1/sqrt(var + eps) or 1/(sqrt(var) + eps)
static __global__ void varianceNormalizationFactor_f32(
        double epsilon, size_t data_size, float* data, bool epsilon_inside_sqrt) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data_size) return;
    if (epsilon_inside_sqrt)
        data[i] = powf(sqrtf(data[i] + (float)epsilon), -1.f);
    else
        data[i] = powf(sqrtf(data[i]) + (float)epsilon, -1.f);
}

// FP16 kernel: operates on __half data in-place
static __global__ void varianceNormalizationFactor_f16(
        double epsilon, size_t data_size, __half* data, bool epsilon_inside_sqrt) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data_size) return;
    float v = __half2float(data[i]);
    float result;
    if (epsilon_inside_sqrt)
        result = powf(sqrtf(v + (float)epsilon), -1.f);
    else
        result = powf(sqrtf(v) + (float)epsilon, -1.f);
    data[i] = __float2half(result);
}

// ── FP32 caller ──────────────────────────────────────────────────────────────
void kernelCaller_f32(hipStream_t stream,
                       unsigned blocks_number,
                       unsigned threads_per_block,
                       double epsilon,
                       size_t data_size,
                       float* data, bool epsilon_inside_sqrt) {
    varianceNormalizationFactor_f32
        <<<blocks_number, threads_per_block, 0, stream>>>(
            epsilon, data_size, data, epsilon_inside_sqrt);
}

// ── FP16 caller ──────────────────────────────────────────────────────────────
void kernelCaller_f16(hipStream_t stream,
                       unsigned blocks_number,
                       unsigned threads_per_block,
                       double epsilon,
                       size_t data_size,
                       float* data_as_float, bool epsilon_inside_sqrt) {
    // data_as_float is actually __half* cast to float* for the generic interface
    __half* data = reinterpret_cast<__half*>(data_as_float);
    varianceNormalizationFactor_f16
        <<<blocks_number, threads_per_block, 0, stream>>>(
            epsilon, data_size, data, epsilon_inside_sqrt);
}

// ── Constructor ───────────────────────────────────────────────────────────────
VarianceNormalizationFactor::VarianceNormalizationFactor(unsigned blocks_number,
                                                         unsigned threads_per_block,
                                                         double epsilon,
                                                         size_t data_size,
                                                         Type_t data_type,
                                                         bool epsilon_inside_sqrt)
    : blocks_number_{blocks_number}, threads_per_block_{threads_per_block},
      epsilon_{epsilon}, data_size_{data_size} {
    switch (data_type) {
        case Type_t::f16:
            epsilon_inside_sqrt_ = epsilon_inside_sqrt;
            func_ptr_ = kernelCaller_f16;
            break;
        case Type_t::f32:
            epsilon_inside_sqrt_ = epsilon_inside_sqrt;
            func_ptr_ = kernelCaller_f32;
            break;
        default:
            throw_ov_exception(
                fmt::format("ov::rocm_gpu::MvnOp: unsupported data type, must be f16 or f32."));
    }
}

void VarianceNormalizationFactor::operator()(hipStream_t stream, void* data) const {
    float* data_ptr = static_cast<float*>(data);
    func_ptr_(stream, blocks_number_, threads_per_block_, epsilon_, data_size_,
              data_ptr, epsilon_inside_sqrt_);
}

}  // namespace kernel

}  // namespace rocm_gpu
}  // namespace ov
