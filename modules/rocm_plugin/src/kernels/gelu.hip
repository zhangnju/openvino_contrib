// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gelu.hpp"
#include "rocm/math.hpp"
namespace ov {
namespace rocm_gpu {
namespace kernel {

namespace rocmath = rocm::math;
/*
template <typename T>
struct GeluErfOpImpl {
    __device__ static inline T op(T x) {
        return static_cast<T>(0.5) * x * (static_cast<T>(1.0) +
            static_cast<T>(rocmath::erff(x / static_cast<T>(rocmath::sqrt(static_cast<T>(2.0))))));
    }
};
*/
static __global__ void GeluErfOpImpl_fp32(const float* in, float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    out[i] = static_cast<float>(0.5) * in[i] * (static_cast<float>(1.0) +
            static_cast<float>(rocmath::erff(in[i] / static_cast<float>(rocmath::sqrt(static_cast<float>(2.0))))));
}
/*
template <typename T>
struct GeluTanhOpImpl {
    __device__ static inline T op(T x) {
        return static_cast<T>(0.5) * x * (static_cast<T>(1.0) +
            static_cast<T>(rocmath::tanh(static_cast<T>(0.7978845608028654) * (x + static_cast<T>(0.044715) * x * x * x))));
    }
};
*/
static __global__ void GeluTanhOpImpl_fp32(const float* in, float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    out[i] = static_cast<float>(0.5) * in[i] * (static_cast<float>(1.0) +
            static_cast<float>(rocmath::tanh(static_cast<float>(0.7978845608028654) * (in[i] + static_cast<float>(0.044715) * in[i] * in[i] * in[i]))));
}
GeluErf::GeluErf(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void GeluErf::operator()(hipStream_t stream, const void* in0, void* out) const {
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            out_fp32 = (float*) out;
            GeluErfOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32,out_fp32);
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
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            out_fp32 = (float*) out;
            GeluTanhOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32,out_fp32);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by GeluTanh operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}
}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
