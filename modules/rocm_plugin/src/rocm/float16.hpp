// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#include <half/half.hpp>

using half_float::half;
typedef half float16;

using half    = half_float::half;
using float16 = half_float::half;
using __half    = half_float::half;


// Taken from:
// https://github.com/dmlc/mshadow/blob/master/mshadow/half.h
union Bits {
  float f;
  int32_t si;
  uint32_t ui;
};
static int const shift = 13;
static int const shiftSign = 16;

static int32_t const infN = 0x7F800000;   // flt32 infinity
static int32_t const maxN = 0x477FE000;   // max flt16 normal as a flt32
static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
static int32_t const signN = 0x80000000;  // flt32 sign bit

static int32_t const infC = infN >> shift;
static int32_t const nanN = (infC + 1) << shift;  // minimum flt16 nan as a flt32
static int32_t const maxC = maxN >> shift;
static int32_t const minC = minN >> shift;
static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
static int32_t const norC = 0x00400;  // min flt32 normal down shifted

static int32_t const maxD = infC - maxC - 1;
static int32_t const minD = minC - subC - 1;

// Host version of device function __float2half_rn()
uint16_t inline float2half(float value) {
  Bits v, s;
  v.f = value;
  uint32_t sign = v.si & signN;
  v.si ^= sign;
  sign >>= shiftSign;  // logical shift
  s.si = mulN;
  s.si = s.f * v.f;  // correct subnormals
  v.si ^= (s.si ^ v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
  v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
  v.ui >>= shift;  // logical shift
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  return v.ui | sign;
}

float inline half2float(uint16_t value) {
  Bits v;
  v.ui = value;
  int32_t sign = v.si & signC;
  v.si ^= sign;
  sign <<= shiftSign;
  v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
  v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
  Bits s;
  s.si = mulC;
  s.f *= v.si;
  int32_t mask = -(norC > v.si);
  v.si <<= shift;
  v.si ^= (s.si ^ v.si) & mask;
  v.si |= sign;
  return v.f;
}

#if 0
/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ __half operator+(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) + static_cast<float>(rh);
}
__device__ __forceinline__ __half operator-(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) - static_cast<float>(rh);
}
__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) * static_cast<float>(rh);
}
__device__ __forceinline__ __half operator/(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) / static_cast<float>(rh);
}

__device__ __forceinline__ __half &operator+=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) + static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __half &operator-=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) - static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __half &operator*=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) * static_cast<float>(rh);
    return lh;
}
__device__ __forceinline__ __half &operator/=(__half &lh, const __half &rh) {
    lh = static_cast<float>(lh) / static_cast<float>(rh);
    return lh;
}

/* Note for increment and decrement we use the raw value 0x3C00U equating to half(1.0F), to avoid the extra conversion
 */
__device__ __forceinline__ __half &operator++(__half &h) {
    __half one=1.0F;
    //one.x = 0x3C00U;
    h += one;
    return h;
}
__device__ __forceinline__ __half &operator--(__half &h) {
    __half one=1.0F;
    //one.x = 0x3C00U;
    h -= one;
    return h;
}
__device__ __forceinline__ __half operator++(__half &h, const int ignored) {
    const __half ret = h;
    __half one=1.0F;
    //one.x = 0x3C00U;
    h += one;
    return ret;
}
__device__ __forceinline__ __half operator--(__half &h, const int ignored) {
    const __half ret = h;
    __half_raw one=1.0F;
    //one.x = 0x3C00U;
    h -= one;
    return ret;
}

/* Unary plus and inverse operators */
__device__ __forceinline__ __half operator+(const __half &h) { return h; }
__device__ __forceinline__ __half operator-(const __half &h) { return static_cast<float>(h); }

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) == static_cast<float>(rh);
}
__device__ __forceinline__ bool operator!=(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) != static_cast<float>(rh);
}
__device__ __forceinline__ bool operator>(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) > static_cast<float>(rh);
}
__device__ __forceinline__ bool operator<(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) < static_cast<float>(rh);
}
__device__ __forceinline__ bool operator>=(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) >= static_cast<float>(rh);
}
__device__ __forceinline__ bool operator<=(const __half &lh, const __half &rh) {
    return static_cast<float>(lh) <= static_cast<float>(rh);
}
//#endif /* !defined(rocm_HAS_HALF_MATH) || defined(__rocm_NO_HALF_OPERATORS__) */
#endif 
