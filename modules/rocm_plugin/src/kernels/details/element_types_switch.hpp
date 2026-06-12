// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rocm_type_traits.hpp"
#include "switch.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <Type_t... Types>
struct ElementTypesSwitch {
    static constexpr std::integer_sequence<int, static_cast<int>(Types)...> indices{};
    template <Type_t I, typename Switch, typename... Args>
    constexpr decltype(auto) case_(Switch&& switch_, Args&&... args) const noexcept(
        noexcept(std::forward<Switch>(switch_).template case_<rocm_type_traits_t<I>>(std::forward<Args>(args)...))) {
        return std::forward<Switch>(switch_).template case_<rocm_type_traits_t<I>>(std::forward<Args>(args)...);
    }
    template <typename Switch, typename... Args, typename TypeT>
    constexpr decltype(auto) default_(TypeT t, Switch&& switch_, Args&&... args) const
        noexcept(noexcept(std::forward<Switch>(switch_).default_(t, std::forward<Args>(args)...))) {
        return std::forward<Switch>(switch_).default_(t, std::forward<Args>(args)...);
    }
    template <typename Switch, typename... Args>
    static constexpr decltype(auto) switch_(Type_t v, Switch&& switchObj, Args&&... args) noexcept(
        noexcept(templateSwitch(
            indices, v, ElementTypesSwitch{}, std::forward<Switch>(switchObj), std::forward<Args>(args)...))) {
        return templateSwitch(
            indices, v, ElementTypesSwitch{}, std::forward<Switch>(switchObj), std::forward<Args>(args)...);
    }
};

using AllElementTypesSwitch = ElementTypesSwitch<
    Type_t::boolean,  // Added: support bool for attention masks and comparison results
    Type_t::f16,
    Type_t::f32,
    Type_t::i8,
    Type_t::i16,
    Type_t::i32,
    Type_t::i64,
    Type_t::u8,
    Type_t::u16,
    Type_t::u32,
    Type_t::u64>;

using FloatElementTypesSwitch = ElementTypesSwitch<Type_t::f16, Type_t::f32>;
}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
