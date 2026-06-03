// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <rocm/float16.hpp>

namespace ov {
namespace rocm_gpu {
namespace kernel {

enum class Type_t : int {
    boolean,
    bf16,
    f16,
    f32,
    f64,
    i4,
    i8,
    i16,
    i32,
    i64,
    u1,
    u4,
    u8,
    u16,
    u32,
    u64
};
constexpr int type_t_first_value = static_cast<int>(Type_t::boolean);
constexpr int type_t_last_value = static_cast<int>(Type_t::u64);

#if 0
enum class Type_t : int {
    f32,
    u8
};

constexpr int type_t_first_value = static_cast<int>(Type_t::f32);
constexpr int type_t_last_value = static_cast<int>(Type_t::u8);
#endif 
#if 0
template <Type_t>
struct rocm_type_traits {
    using value_type = void;
};
#endif 
template <Type_t>
struct rocm_type_traits {
    using value_type = float;
};
template <Type_t Type>
using rocm_type_traits_t = typename rocm_type_traits<Type>::value_type;

template <>
struct rocm_type_traits<Type_t::boolean> {
    using value_type = char;
};

template <>
struct rocm_type_traits<Type_t::f32> {
    using value_type = float;
};

template <>
struct rocm_type_traits<Type_t::f64> {
    using value_type = double;
};

template <>
struct rocm_type_traits<Type_t::i4> {
    using value_type = int8_t;
};

template <>
struct rocm_type_traits<Type_t::i8> {
    using value_type = int8_t;
};

template <>
struct rocm_type_traits<Type_t::i16> {
    using value_type = int16_t;
};

template <>
struct rocm_type_traits<Type_t::i32> {
    using value_type = int32_t;
};

template <>
struct rocm_type_traits<Type_t::i64> {
    using value_type = int64_t;
};

template <>
struct rocm_type_traits<Type_t::u1> {
    using value_type = int8_t;
};

template <>
struct rocm_type_traits<Type_t::u4> {
    using value_type = int8_t;
};


template <>
struct rocm_type_traits<Type_t::u8> {
    using value_type = uint8_t;
};

template <>
struct rocm_type_traits<Type_t::u16> {
    using value_type = uint16_t;
};

template <>
struct rocm_type_traits<Type_t::u32> {
    using value_type = uint32_t;
};

template <>
struct rocm_type_traits<Type_t::u64> {
    using value_type = uint64_t;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
