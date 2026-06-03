// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/float16.hpp>

#include "multiply.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
struct MultiplyOpImpl {
    __device__ static inline T op(T in0, T in1) { return in0 * in1; }
};

/*
static __global__ void MultiplyOpImpl_fp32(const float* in0, const float* in1,float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    float x = in0[i];
    float y = in1[i];

    out[i] = x * y;
}
*/

Multiply::Multiply(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}
/*
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = out_num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);
}
*/
void Multiply::operator()(hipStream_t stream,
                          const void* in0,
                          const NumpyBroadcastMapper& in0_mapper,
                          const void* in1,
                          const NumpyBroadcastMapper& in1_mapper,
                          void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
    /*
    const float* in0_fp32=nullptr;
    const float* in1_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            in1_fp32 = (const float*) in1;
            out_fp32 = (float*) out;
            MultiplyOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32, in1_fp32,out_fp32);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Multiply operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
    */
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
