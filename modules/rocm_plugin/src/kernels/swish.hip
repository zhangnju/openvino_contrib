// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/float16.hpp>

#include "details/tensor_helpers.hpp"
#include "swish.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

//namespace cumath = rocm::math;
/*
template <typename T>
struct SwishOpImpl {
    __device__ static inline T op(T x, double beta) {
        return x / (static_cast<T>(1.0f) + exp(-x * static_cast<T>(beta)));
    }
};
*/

static __global__ void SwishOpImpl_fp32(const float* in, double beta, float* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = in[i];
    out[i] = x / (1.0f + expf(-x * static_cast<float>(beta)));
}

static __global__ void SwishOpImpl_fp16(const __half* in, double beta, __half* out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = __half2float(in[i]);
    out[i] = __float2half(x / (1.0f + expf(-x * static_cast<float>(beta))));
}
Swish::Swish(Type_t element_type, size_t max_threads_per_block, size_t num_elements, double beta)
    : beta_{beta} 
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void Swish::operator()(hipStream_t stream, const void* in, void* out) const 
{ 
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            SwishOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const float*)in, beta_, (float*)out);
            break;
        case Type_t::f16:
            SwishOpImpl_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const __half*)in, beta_, (__half*)out);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Swish operation !!",
                                         static_cast<Type_t>(element_type_)));
    } 
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
