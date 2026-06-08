// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/float16.hpp>
#include "rocm/math.hpp"
#include <type_traits>

#include "floor_mod.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

namespace {
/*
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T floor_mod(T x, T y) {
    // NOTE: return value according to reference implementation in
    // openvino/ngraph/core/reference/include/ngraph/runtime/reference/floor_mod.hpp
    // const double divisor = static_cast<double>(y);
    // return x - y * std::floor(x / divisor);
    // for y = 0 it is static_cast<T>(-nan) which is minimal limit for integers
    if (y == 0) {
        return rocm::math::limit_min<T>();
    }
    const T mod = x % y;
    if (mod == 0) {
        return mod;
    }
    T div = x / y;
    if ((x < 0) != (y < 0)) {
        --div;
    }
    return x - y * div;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T floor_mod(T x, T y) {
    return x - y * rocm::math::floor(x / y);
}
*/
static __global__ void FloorModOpImpl_fp32(const float* in0, const float* in1, float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = in0[i] - in1[i] * rocm::math::floor(in0[i] / in1[i]);
}

static __global__ void FloorModOpImpl_i32(const int32_t* in0, const int32_t* in1, int32_t* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (in1[i] == 0) { out[i] = INT32_MIN; return; }
    int32_t mod = in0[i] % in1[i];
    if (mod == 0) { out[i] = 0; return; }
    int32_t d = in0[i] / in1[i];
    if ((in0[i] < 0) != (in1[i] < 0)) --d;
    out[i] = in0[i] - in1[i] * d;
}

static __global__ void FloorModOpImpl_i64(const int64_t* in0, const int64_t* in1, int64_t* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (in1[i] == 0) { out[i] = INT64_MIN; return; }
    int64_t mod = in0[i] % in1[i];
    if (mod == 0) { out[i] = 0; return; }
    int64_t d = in0[i] / in1[i];
    if ((in0[i] < 0) != (in1[i] < 0)) --d;
    out[i] = in0[i] - in1[i] * d;
}
}  // namespace
/*
template <typename T>
struct FloorModOpImpl {
    __device__ static inline T op(T in0, T in1) { return floor_mod(in0, in1); }
};
*/
FloorMod::FloorMod(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = out_num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);
}

void FloorMod::operator()(hipStream_t stream,
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
            FloorModOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32, in1_fp32 ,out_fp32);
            break;
        case Type_t::i32:
            FloorModOpImpl_i32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const int32_t*)in0, (const int32_t*)in1, (int32_t*)out);
            break;
        case Type_t::i64:
            FloorModOpImpl_i64<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const int64_t*)in0, (const int64_t*)in1, (int64_t*)out);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by FloorMod operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
