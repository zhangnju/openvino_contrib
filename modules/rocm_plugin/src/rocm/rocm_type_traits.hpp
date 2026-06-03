// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#ifdef __rocmCC__
#include <rocm/float16.hpp>
#endif

namespace rocm {

template <ov::element::Type_t Type>
struct rocm_type_traits : ov::element_type_traits<Type> {};

#ifdef __rocmCC__
template <>
struct rocm_type_traits<ov::element::Type_t::f16> { /* 5bit exponent, 10bit mantissa */
    using value_type = __half;                      // 16-bit half-precision floating point (FP16) representation:
                                                    // 1 sign bit, 5 exponent bits, and 10 mantissa bits.
};

template <>
struct rocm_type_traits<ov::element::Type_t::bf16> { /* 8bit exponent, 7bit mantissa */
    using value_type = __nv_bfloat16;
};

#endif

template <ov::element::Type_t Type>
using rocm_type_traits_t = typename rocm_type_traits<Type>::value_type;

}  // namespace rocm
