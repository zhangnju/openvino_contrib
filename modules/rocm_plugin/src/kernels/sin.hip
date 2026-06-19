// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sin.hpp"
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

//namespace cumath = rocm::math;
/*
template <typename T>
struct SinOpImpl {
    __device__ static inline T op(T x) {
        return sin(x);
    }
};
*/
static __global__ void SinOpImpl_fp32(const float* in, float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    out[i] = sinf(in[i]);
}

static __global__ void SinOpImpl_fp16(const __half* in, __half* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = __float2half(sinf(__half2float(in[i])));
}
Sin::Sin(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void Sin::operator()(hipStream_t stream, const void* in0, void* out) const {
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            out_fp32 = (float*) out;
            SinOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32,out_fp32);
            break;
        case Type_t::f16:
            SinOpImpl_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>((const __half*)in0, (__half*)out);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Sin operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
