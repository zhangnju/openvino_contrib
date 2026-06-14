// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gelu.hpp"
#include "rocm/math.hpp"
#include <hip/hip_fp16.h>
namespace ov {
namespace rocm_gpu {
namespace kernel {

namespace rocmath = rocm::math;

// f32 GELU-Erf kernel
static __global__ void GeluErfOpImpl_fp32(const float* in, float* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = 0.5f * in[i] * (1.0f + rocmath::erff(in[i] * 0.7071067811865475f));
}

// f16 GELU-Erf kernel (accumulates in f32 for precision)
static __global__ void GeluErfOpImpl_fp16(const __half* in, __half* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = __half2float(in[i]);
    float y = 0.5f * x * (1.0f + rocmath::erff(x * 0.7071067811865475f));
    out[i] = __float2half(y);
}

// f32 GELU-Tanh kernel
static __global__ void GeluTanhOpImpl_fp32(const float* in, float* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[i];
    out[i] = 0.5f * x * (1.0f + rocmath::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// f16 GELU-Tanh kernel (accumulates in f32 for precision)
static __global__ void GeluTanhOpImpl_fp16(const __half* in, __half* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = __half2float(in[i]);
    float y = 0.5f * x * (1.0f + rocmath::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    out[i] = __float2half(y);
}
GeluErf::GeluErf(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void GeluErf::operator()(hipStream_t stream, const void* in0, void* out) const {
    switch(element_type_) {
        case Type_t::f32:
            GeluErfOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const float*>(in0), static_cast<float*>(out));
            break;
        case Type_t::f16:
            GeluErfOpImpl_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const __half*>(in0), static_cast<__half*>(out));
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by GeluErf operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

GeluTanh::GeluTanh(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void GeluTanh::operator()(hipStream_t stream, const void* in0, void* out) const {
    switch(element_type_) {
        case Type_t::f32:
            GeluTanhOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const float*>(in0), static_cast<float*>(out));
            break;
        case Type_t::f16:
            GeluTanhOpImpl_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const __half*>(in0), static_cast<__half*>(out));
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by GeluTanh operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}
}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
