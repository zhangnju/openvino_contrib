// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm/float16.hpp"
#include "rocm/math.hpp"
#include <type_traits>

#include "divide.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

namespace {

/*
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T div(T x, T y) {
    // NOTE: as by default rocm math returns -1 for integer division by 0
    // this behavior was implemented to be more similar to floating point division:
    // inf ->limit_max, -inf ->limit_min, nan -> limit_min
    if (y == 0) {
        return x > 0 ? rocm::math::limit_max<T>() : rocm::math::limit_min<T>();
    }
    return x / y;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T div(T x, T y) {
    return x / y;
}
*/
static __global__ void DivOpImpl_fp32(const float* in0, NumpyBroadcastMapper m0,
                                      const float* in1, NumpyBroadcastMapper m1,
                                      float* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fdividef(in0[m0.srcIndex(i)], in1[m1.srcIndex(i)]);
}

static __global__ void DivOpImpl_i32(const int32_t* in0, NumpyBroadcastMapper m0,
                                     const int32_t* in1, NumpyBroadcastMapper m1,
                                     int32_t* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t a = in0[m0.srcIndex(i)], b = in1[m1.srcIndex(i)];
    out[i] = b != 0 ? a / b : (a > 0 ? INT32_MAX : INT32_MIN);
}

static __global__ void DivOpImpl_i64(const int64_t* in0, NumpyBroadcastMapper m0,
                                     const int64_t* in1, NumpyBroadcastMapper m1,
                                     int64_t* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t a = in0[m0.srcIndex(i)], b = in1[m1.srcIndex(i)];
    out[i] = b != 0 ? a / b : (a > 0 ? INT64_MAX : INT64_MIN);
}

static __global__ void PythonDivOpImpl_i32(const int32_t* in0, NumpyBroadcastMapper m0,
                                           const int32_t* in1, NumpyBroadcastMapper m1,
                                           int32_t* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t a = in0[m0.srcIndex(i)], b = in1[m1.srcIndex(i)];
    if (b == 0) { out[i] = a > 0 ? INT32_MAX : INT32_MIN; return; }
    int32_t d = a / b;
    int32_t mod = a % b;
    if (mod != 0 && ((a < 0) != (b < 0))) d -= 1;
    out[i] = d;
}

static __global__ void PythonDivOpImpl_i64(const int64_t* in0, NumpyBroadcastMapper m0,
                                           const int64_t* in1, NumpyBroadcastMapper m1,
                                           int64_t* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t a = in0[m0.srcIndex(i)], b = in1[m1.srcIndex(i)];
    if (b == 0) { out[i] = a > 0 ? INT64_MAX : INT64_MIN; return; }
    int64_t d = a / b;
    int64_t mod = a % b;
    if (mod != 0 && ((a < 0) != (b < 0))) d -= 1;
    out[i] = d;
}
/*
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T python_div(T x, T y) {
    // NOTE: as by default rocm math returns -1 for integer division by 0
    // this behavior was implemented to be more similar to floating point division:
    // inf ->limit_max, -inf ->limit_min, nan -> limit_min
    if (y == 0) {
        return x > 0 ? rocm::math::limit_max<T>() : rocm::math::limit_min<T>();
    }
    const T div = x / y;
    const T mod = x % y;
    if (mod != 0 && (x < 0) != (y < 0)) {
        return div - 1;
    }
    return div;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T python_div(T x, T y) {
    return div(x, y);
}
*/
}  // namespace

template <typename T>
struct DivideOpImpl {
    __device__ static inline T op(T in0, T in1) { return div(in0, in1); }
};

template <typename T>
struct PythonDivideOpImpl {
    __device__ static inline T op(T in0, T in1) { return python_div(in0, in1); }
};

Divide::Divide(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = out_num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);
}

void Divide::operator()(hipStream_t stream,
                        const void* in0,
                        const NumpyBroadcastMapper& in0_mapper,
                        const void* in1,
                        const NumpyBroadcastMapper& in1_mapper,
                        void* out) const {
    switch(element_type_)
    {
        case Type_t::f32:
            DivOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const float*)in0, in0_mapper, (const float*)in1, in1_mapper, (float*)out, num_elements_);
            break;
        case Type_t::i32:
            DivOpImpl_i32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const int32_t*)in0, in0_mapper, (const int32_t*)in1, in1_mapper, (int32_t*)out, num_elements_);
            break;
        case Type_t::i64:
            DivOpImpl_i64<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const int64_t*)in0, in0_mapper, (const int64_t*)in1, in1_mapper, (int64_t*)out, num_elements_);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Div operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

PythonDivide::PythonDivide(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = out_num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);  
}

void PythonDivide::operator()(hipStream_t stream,
                              const void* in0,
                              const NumpyBroadcastMapper& in0_mapper,
                              const void* in1,
                              const NumpyBroadcastMapper& in1_mapper,
                              void* out) const {
    switch(element_type_)
    {
        case Type_t::f32:
            DivOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const float*)in0, in0_mapper, (const float*)in1, in1_mapper, (float*)out, num_elements_);
            break;
        case Type_t::i32:
            PythonDivOpImpl_i32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const int32_t*)in0, in0_mapper, (const int32_t*)in1, in1_mapper, (int32_t*)out, num_elements_);
            break;
        case Type_t::i64:
            PythonDivOpImpl_i64<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const int64_t*)in0, in0_mapper, (const int64_t*)in1, in1_mapper, (int64_t*)out, num_elements_);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Python-Div operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
