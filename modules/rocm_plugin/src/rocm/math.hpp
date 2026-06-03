// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "float16.hpp"

namespace rocm {
namespace math {

/* =================== limit_min =================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, bool> = true>
inline __device__ __host__ T limit_min() {
    return 0;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
inline __device__ __host__ T limit_min() {
    return static_cast<T>(1) << (sizeof(T) * 8 - 1);
}
/* ================================================= */

/* =================== limit_max =================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, bool> = true>
inline __device__ __host__ T limit_max() {
    return static_cast<T>(-1);
}

template <typename T, std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value, bool> = true>
inline __device__ __host__ T limit_max() {
    return ~limit_min<T>();
}
/* ================================================= */

template <typename T>
inline __device__ T round(T x) {
    return ::round(x);
}

/* ===================== floor ===================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline __device__ T floor(T x) {
    return x;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline __device__ T floor(T x) {
    return ::floor(x);
}

inline __device__ float floor(float x) { return ::floorf(x); }
/* ================================================= */

/* ===================== trunc ===================== */
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline __device__ T trunc(T x) {
    return x;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
inline __device__ T trunc(T x) {
    return ::trunc(x);
}

inline __device__ float trunc(float x) { return ::truncf(x); }
/* ================================================= */

template <typename T>
inline __device__ T min(T x, T y) {
    return x < y ? x : y;
}

template <typename T>
inline __device__ T max(T x, T y) {
    return x > y ? x : y;
}

template <typename T>
inline __device__ T exp(T x) {
    return static_cast<T>(::exp(static_cast<float>(x)));
}

template <typename T>
inline __device__ T pow(T x, T y) {
    return static_cast<T>(powf(static_cast<float>(x), static_cast<float>(y)));
}

template <typename T>
inline __device__ T sqrt(T a) {
    return static_cast<T>(::sqrtf(static_cast<float>(a)));
}

template <typename T>
inline __device__ T abs(T a) {
    return static_cast<T>(::fabsf(static_cast<float>(a)));
}

template <typename T>
inline __device__ T tanh(T a) {
    return static_cast<T>(::tanhf(static_cast<float>(a)));
}

template <typename T>
inline __device__ T erff(T a) {
    return ::erff(a);
}

template <typename T>
inline __device__ T sin(T a) {
    return static_cast<T>(::sinf(static_cast<float>(a)));
}

template <typename T>
inline __device__ T sinh(T a) {
    return static_cast<T>(::sinhf(static_cast<float>(a)));
}

template <typename T>
inline __device__ T cos(T a) {
    return static_cast<T>(::cosf(static_cast<float>(a)));
}

template <typename T>
inline __device__ T cosh(T a) {
    return static_cast<T>(::coshf(static_cast<float>(a)));
}

template <typename T>
inline __device__ T log(T a) {
    return static_cast<T>(::logf(static_cast<float>(a)));
}

#ifdef __rocmCC__
/* ==================== __half ===================== */
template <>
inline __device__ __half round(__half x) {
    return ::round(static_cast<float>(x));
}

#if defined(rocm_HAS_HALF_MATH)
inline __device__ __half floor(__half x) { return ::hfloor(x); }

inline __device__ __half trunc(__half x) { return ::htrunc(x); }

template <>
inline __device__ __half exp<__half>(__half x) {
    return ::hexp(x);
}

template <>
inline __device__ __half sqrt<__half>(__half x) {
    return ::hsqrt(x);
}

template <>
inline __device__ __half abs<__half>(__half x) {
    return ::__habs(x);
}

template <>
inline __device__ __half sin<__half>(__half x) {
    return ::hsin(x);
}

template <>
inline __device__ __half cos<__half>(__half x) {
    return ::hcos(x);
}

template <>
inline __device__ __half log<__half>(__half x) {
    return ::hlog(x);
}

#else  // defined (rocm_HAS_HALF_MATH)

inline __device__ __half floor(__half x) { return floor(static_cast<float>(x)); }

inline __device__ __half trunc(__half x) { return trunc(static_cast<float>(x)); }

template <>
inline __device__ __half min<__half>(__half x, __half y) {
    return min<float>(static_cast<float>(x), static_cast<float>(y));
}

template <>
inline __device__ __half max<__half>(__half x, __half y) {
    return max<float>(static_cast<float>(x), static_cast<float>(y));
}

template <>
inline __device__ __half exp<__half>(__half x) {
    return exp<float>(static_cast<float>(x));
}

template <>
inline __device__ __half sqrt<__half>(__half x) {
    return ::sqrt(static_cast<float>(x));
}

template <>
inline __device__ __half abs<__half>(__half x) {
    return ::abs(static_cast<float>(x));
}

template <>
inline __device__ __half sin<__half>(__half x) {
    return ::sin(static_cast<float>(x));
}

template <>
inline __device__ __half cos<__half>(__half x) {
    return ::cos(static_cast<float>(x));
}

template <>
inline __device__ __half log<__half>(__half x) {
    return ::log(static_cast<float>(x));
}

#endif  // defined (rocm_HAS_HALF_MATH)


#endif  // __rocmCC__

}  // namespace math
}  // namespace rocm
