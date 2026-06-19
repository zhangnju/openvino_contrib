// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/float16.hpp>
#include <hip/hip_fp16.h>

#include "details/tensor_helpers.hpp"
#include "elu.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {
/*
//namespace cumath = rocm::math;
template <typename T>
struct EluOpImpl {
    __device__ static inline T op(T x, float alpha) {
        return x >= static_cast<T>(0.0) ? x : static_cast<T>(alpha) * (exp(x) - static_cast<T>(1.0));
    }
};
*/

static __global__ void EluOpImpl_fp32(const float* in,float alpha, float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[i] ;
    out[i] = ( x >= static_cast<float>(0.0) ) ? in[i] : (static_cast<float>(alpha) * (expf(x) - static_cast<float>(1.0)));
}

static __global__ void EluOpImpl_fp16(const __half* in, float alpha, __half* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = __half2float(in[i]);
    out[i] = __float2half( (x >= 0.0f) ? x : (alpha * (expf(x) - 1.0f)) );
}
Elu::Elu(Type_t element_type, size_t max_threads_per_block, size_t num_elements, float alpha)
    : alpha_{alpha} 
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void Elu::operator()(hipStream_t stream, const void* in, void* out) const 
{ 
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in;
            out_fp32 = (float*) out;
            EluOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32, alpha_,out_fp32);
            break;
        case Type_t::f16:
            EluOpImpl_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>((const __half*)in, alpha_, (__half*)out);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Elu operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
 }

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
