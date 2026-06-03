#include "hip/hip_runtime.h"
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <rocm/float16.hpp>
#include <math.h>
#include <type_traits>

#include "convert_color_nv12.hpp"
#include "details/error.hpp"
#include "details/tensor_helpers.hpp"
//#include "details/type_validator.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <ColorConversion ColorFormat>
__global__ void color_convert_nv12(const float* arg_y,
                                   const float* arg_uv,
                                   unsigned char* out_ptr,
                                   const size_t batch_size,
                                   const size_t image_h,
                                   const size_t image_w,
                                   const size_t stride_y,
                                   const size_t stride_uv) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (batch_size * image_h * image_w)) {
        return;
    }

    const unsigned w = idx % image_w;
    const unsigned w_left = idx / image_w;
    const unsigned h = w_left % image_h;
    const unsigned h_left = w_left / image_h;
    const unsigned batch = h_left % batch_size;

    unsigned char* out = out_ptr + batch * image_w * image_h * 3;
    const float* y_ptr = arg_y + batch * stride_y;
    const float* uv_ptr = arg_uv + batch * stride_uv;
    const size_t y_index = h * image_w + w;
    const auto y_val = static_cast<float>(y_ptr[y_index]);
    const size_t uv_index = (h / 2) * image_w + (w / 2) * 2;
    const auto u_val = static_cast<float>(uv_ptr[uv_index]);
    const auto v_val = static_cast<float>(uv_ptr[uv_index + 1]);

    unsigned char r, g, b;
    yuv_pixel_to_rgb(y_val, u_val, v_val, r, g, b);
    if (ColorFormat == ColorConversion::RGB) {
        out[y_index * 3] = r;
        out[y_index * 3 + 1] = g;
        out[y_index * 3 + 2] = b;
    } else if (ColorFormat == ColorConversion::BGR) {
        out[y_index * 3] = b;
        out[y_index * 3 + 1] = g;
        out[y_index * 3 + 2] = r;
    }
}

template <ColorConversion Conversion>
NV12ColorConvert<Conversion>::NV12ColorConvert(const Type_t element_type,
                                               const size_t max_threads_per_block,
                                               const size_t batch_size,
                                               const size_t image_h,
                                               const size_t image_w,
                                               const size_t stride_y,
                                               const size_t stride_uv)
    : element_type_{element_type},
      batch_size_{batch_size},
      image_h_{image_h},
      image_w_{image_w},
      stride_y_{stride_y},
      stride_uv_{stride_uv} {
    //TypeValidator<Switcher>::check(element_type_);
    std::tie(num_blocks_, threads_per_block_) =
        calculateElementwiseGrid(batch_size * image_h * image_w, max_threads_per_block);
}

template <ColorConversion Conversion>
void NV12ColorConvert<Conversion>::operator()(hipStream_t stream, const void* in, void* out) const {
    callKernel(stream, in, out);
}

template <ColorConversion Conversion>
void NV12ColorConvert<Conversion>::operator()(hipStream_t stream, const void* in0, const void* in1, void* out) const {
    callKernel(stream, in0, in1, out);
}
/*
template <ColorConversion Conversion>
template <typename T, typename... Args>
constexpr void NV12ColorConvert<Conversion>::case_(hipStream_t stream, Args&&... args) const noexcept {
    callKernel<T>(stream, std::forward<Args>(args)...);
}

template <ColorConversion Conversion>
template <typename T, typename... Args>
void NV12ColorConvert<Conversion>::default_(T t, hipStream_t, Args&&...) const noexcept {
    throw_ov_exception(fmt::format("Element type = {} is not supported by NV12ColorConvert operation.", t));
}
*/
template <ColorConversion Conversion>
void NV12ColorConvert<Conversion>::callKernel(const hipStream_t stream, const void* in, void* out) const {
    return color_convert_nv12<Conversion>
        <<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const float*>(in),
                                                         static_cast<const float*>(in) + image_w_ * image_h_,
                                                         static_cast<unsigned char*>(out),
                                                         batch_size_,
                                                         image_h_,
                                                         image_w_,
                                                         stride_y_,
                                                         stride_uv_);
}

template <ColorConversion Conversion>
void NV12ColorConvert<Conversion>::callKernel(const hipStream_t stream,
                                              const void* in0,
                                              const void* in1,
                                              void* out) const {
    return color_convert_nv12<Conversion><<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const float*>(in0),
                                                                                          static_cast<const float*>(in1),
                                                                                          static_cast<unsigned char*>(out),
                                                                                          batch_size_,
                                                                                          image_h_,
                                                                                          image_w_,
                                                                                          stride_y_,
                                                                                          stride_uv_);
}

template class NV12ColorConvert<ColorConversion::RGB>;
template class NV12ColorConvert<ColorConversion::BGR>;

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
