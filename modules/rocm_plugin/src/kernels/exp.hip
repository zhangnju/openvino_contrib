// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "exp.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

//namespace cumath = rocm::math;
/*
template <typename T>
struct ExpOpImpl {
    __device__ static inline T op(T x) {
        return exp(x);
    }
};
*/
static __global__ void ExpOpImpl_fp32(const float* in,float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    out[i]=exp(in[i]);
}
Exp::Exp(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void Exp::operator()(hipStream_t stream, const void* in0, void* out) const {
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            out_fp32 = (float*) out;
            ExpOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32, out_fp32);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Cos operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
