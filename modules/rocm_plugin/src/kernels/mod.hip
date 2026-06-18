// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/float16.hpp>
#include <rocm/math.hpp>
#include <type_traits>

#include "mod.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

namespace {
/*
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T mod(T x, T y) {
    // NOTE: for y = 0 and any x returning int value similar to nan
    if (y == 0) {
        return rocm::math::limit_min<T>();
    }
    return x % y;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T mod(T x, T y) {
    return x - y * trunc(x / y);
}
*/


}  // namespace
/*
template <typename T>
struct ModOpImpl {
    __device__ static inline T op(T in0, T in1) { return mod(in0, in1); }
};
*/

static __global__ void ModOpImpl_fp32(const float* in0, NumpyBroadcastMapper m0,
                                      const float* in1, NumpyBroadcastMapper m1,
                                      float* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in0[m0.srcIndex(i)];
    float y = in1[m1.srcIndex(i)];
    out[i] = (y == 0) ? 0.f : fmodf(x, y);
}
Mod::Mod(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = out_num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);
}

void Mod::operator()(hipStream_t stream,
                     const void* in0,
                     const NumpyBroadcastMapper& in0_mapper,
                     const void* in1,
                     const NumpyBroadcastMapper& in1_mapper,
                     void* out) const {
    const float* in0_fp32=nullptr;
    const float* in1_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            in1_fp32 = (const float*) in1;
            out_fp32 = (float*) out;
            ModOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                in0_fp32, in0_mapper, in1_fp32, in1_mapper, out_fp32, (unsigned)num_elements_);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Mod operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
