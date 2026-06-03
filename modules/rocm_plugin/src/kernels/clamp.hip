// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "hip/hip_runtime.h"
#include "rocm/math.hpp"

#include "clamp.hpp"
#include "details/error.hpp"
#include "details/tensor_helpers.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

//namespace cumath = rocm::math;
/*
template <typename T>
struct ClampOpImpl {
    __device__ static inline T op(T x, T min_value, T max_value) {
        return rocm::math::min(rocm::math::max(x, min_value), max_value);
    }
};

template <typename ElementTypes>
struct MinMaxSwitch {
    template <typename T>
    void case_(
        const EWClamp& ewclamp, hipStream_t stream, const void* in, void* out, double d_min, double d_max) const {
        //T min_value = double_round_cast<T>(d_min, std::ceil);
        //T max_value = double_round_cast<T>(d_max, std::floor);
        ewclamp.callKernel<T>(stream, in, out, double_round_cast<T>(d_min, std::ceil), double_round_cast<T>(d_max, std::floor));
    }

    template <typename T>
    void default_(T t, const EWClamp&, hipStream_t, const void*, void*, double, double) const {
        throw_ov_exception(fmt::format("Element type = {} is not supported.", t));
    }
};
*/
static __global__ void ClampOpImpl_fp32(const float* in,float min_value, float max_value, float* out)
{
     const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

     out[i] = fminf(fmaxf(in[i], min_value), max_value);
}
Clamp::Clamp(Type_t element_type, size_t max_threads_per_block, size_t num_elements, double min, double max)
    : element_type_{element_type},
      min_{min},
      max_{max} 
{
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
}

void Clamp::operator()(hipStream_t stream, const void* in, void* out) const {
    //MinMaxSwitch<AllElementTypesSwitch> switcher{};
    //AllElementTypesSwitch::switch_(element_type_, switcher, ew_clamp_, stream, in, out, min_, max_);
    const float* in0_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in;
            out_fp32 = (float*) out;
            ClampOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32, static_cast<float>(min_), static_cast<float>(max_), out_fp32);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Clamp operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
