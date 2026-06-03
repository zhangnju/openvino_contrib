// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm/float16.hpp"
#include "rocm/math.hpp"
#include "round.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {
/*
template <typename T>
struct RoundOpImpl {
    __device__ static inline T op(T x) { return rocm::math::round(x); }
};
*/
static __global__ void RoundOpImpl_fp32(const float* in, float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    out[i] = rocm::math::round(in[i]);
}

Round::Round(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void Round::operator()(hipStream_t stream, const void* in, void* out) const 
{ 
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in;
            out_fp32 = (float*) out;
            RoundOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32,out_fp32);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Round operation !!",
                                         static_cast<Type_t>(element_type_)));
    } 
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
