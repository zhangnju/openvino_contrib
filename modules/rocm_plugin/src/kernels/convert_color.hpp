// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "rocm/math.hpp"
#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp" 
#include "details/rocm_type_traits.hpp"
#include <rocm/float16.hpp>

namespace ov {
namespace rocm_gpu {
namespace kernel {

enum class ColorConversion { RGB, BGR };


__device__ void yuv_pixel_to_rgb(const float y_val, const float u_val, const float v_val, unsigned char &r, unsigned char &g, unsigned char &b) {
    const float c = y_val - 16.f;
    const float d = u_val - 128.f;
    const float e = v_val - 128.f;
    constexpr float lo = 0.f;
    constexpr float hi = 255.f;
    auto clip = [lo, hi](const float a) -> unsigned char {
        if (std::is_integral<unsigned char>::value) {
            return static_cast<unsigned char>(fminf(fmaxf(roundf(a), lo), hi));
        } else {
            return static_cast<float>(fminf(fmaxf(a, lo), hi));
        }
    };
    b = clip(1.164f * c + 2.018f * d);
    g = clip(1.164f * c - (0.391f) * d - (0.813f) * e);
    r = clip(1.164f * c + (1.596f) * e);
}


}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
