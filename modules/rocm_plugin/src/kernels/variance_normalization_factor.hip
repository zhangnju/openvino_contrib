#include "hip/hip_runtime.h"
// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm/math.hpp"

//#include "convert.hpp"
//#include "details/typed_functor.hpp"
#include "kernels/variance_normalization_factor.hpp"

namespace ov {
namespace rocm_gpu {

namespace kernel {

//template <typename T, bool epsilon_inside_sqrt>
static __global__ void varianceNormalizationFactor(double epsilon, size_t data_size, float* data, bool epsilon_inside_sqrt) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data_size) {
        return;
    }
    
    if (epsilon_inside_sqrt) {
            data[i] = powf(sqrtf(data[i] + epsilon), -1);
    } else {
            data[i] = powf(sqrtf(data[i]) + epsilon, -1);
    }
}

void kernelCaller(hipStream_t stream,
                  unsigned blocks_number,
                  unsigned threads_per_block,
                  double epsilon,
                  size_t data_size,
                  float *data, bool epsilon_inside_sqrt) {
    varianceNormalizationFactor<<<blocks_number, threads_per_block, 0, stream>>>(epsilon, data_size, data, epsilon_inside_sqrt);
    //    <<<blocks_number, threads_per_block, 0, stream>>>(epsilon, data_size, static_cast<T *>(data));
}

VarianceNormalizationFactor::VarianceNormalizationFactor(unsigned blocks_number,
                                                         unsigned threads_per_block,
                                                         double epsilon,
                                                         size_t data_size,
                                                         Type_t data_type,
                                                         bool epsilon_inside_sqrt)
    : blocks_number_{blocks_number}, threads_per_block_{threads_per_block}, epsilon_{epsilon}, data_size_{data_size}  {
    switch (data_type) {
        /*
        //case Type_t::bf16:
        case Type_t::f16:
            element_size_=  2 ;
            epsilon_inside_sqrt_ = epsilon_inside_sqrt;
            func_ptr_ = kernelCaller;
            break;
        */
        case Type_t::f32:
            epsilon_inside_sqrt_ = epsilon_inside_sqrt;
            func_ptr_ = kernelCaller;
            break;
        /*
        case Type_t::f64:
            element_size_=  8 ;
            epsilon_inside_sqrt_ = epsilon_inside_sqrt;
            func_ptr_ = kernelCaller;
            break;
        */
        default:
            throw_ov_exception(
                fmt::format("ov::rocm_gpu::MvnOp: unsupported data type, must be any float point type."));
    }

}

void VarianceNormalizationFactor::operator()(hipStream_t stream, void *data) const {
    float* data_fp32 = (float*) data;
    func_ptr_(stream, blocks_number_, threads_per_block_, epsilon_, data_size_, data_fp32,epsilon_inside_sqrt_);
}

}  // namespace kernel

}  // namespace rocm_gpu
}  // namespace ov
