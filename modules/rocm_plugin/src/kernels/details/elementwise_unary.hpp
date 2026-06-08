// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fmt/format.h>
#include <hip/hip_fp16.h>

#include <tuple>
#include <utility>

#include "rocm_type_traits.hpp"
#include "element_types_switch.hpp"
#include "tensor_helpers.hpp"
#include "type_validator.hpp"

#include "rocm/math.hpp"


namespace ov {
namespace rocm_gpu {
namespace kernel {

// ── Scalar kernel ─────────────────────────────────────────────────────────
template <typename T, typename OP, typename... Args>
__global__ void elementwise_unary(const T* in, size_t num_elements, T* out, Args... args) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in[i], args...);
    }
}

// ── Vectorized FP16 unary: 4 elements per thread ─────────────────────────
template <typename OP>
__global__ void elementwise_unary_fp16_vec4(const __half* __restrict__ in,
                                             size_t n,
                                             __half* __restrict__ out) {
    const size_t base = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        const __half2* in2 = reinterpret_cast<const __half2*>(in + base);
        __half2*       out2 = reinterpret_cast<__half2*>(out + base);
        __half2 a0 = in2[0], a1 = in2[1];
        __half2 r0, r1;
        r0.x = OP::op(a0.x); r0.y = OP::op(a0.y);
        r1.x = OP::op(a1.x); r1.y = OP::op(a1.y);
        out2[0] = r0; out2[1] = r1;
    } else {
        for (size_t i = base; i < n; ++i) out[i] = OP::op(in[i]);
    }
}

// ── Vectorized FP32 unary: 4 elements per thread ─────────────────────────
template <typename OP>
__global__ void elementwise_unary_fp32_vec4(const float* __restrict__ in,
                                             size_t n,
                                             float* __restrict__ out) {
    const size_t base = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        float4 a = *reinterpret_cast<const float4*>(in + base);
        float4 r;
        r.x = OP::op(a.x); r.y = OP::op(a.y);
        r.z = OP::op(a.z); r.w = OP::op(a.w);
        *reinterpret_cast<float4*>(out + base) = r;
    } else {
        for (size_t i = base; i < n; ++i) out[i] = OP::op(in[i]);
    }
}

template <typename ElementTypes, template <typename> typename OP>
class ElementwiseUnary {
public:
    ElementwiseUnary(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
        : element_type_{element_type}, num_elements_{num_elements} {
        TypeValidator<ElementTypes>::check(element_type);
        // Vectorized grid for f16/f32 (4 elements per thread)
        const bool use_vec = (element_type == Type_t::f16 || element_type == Type_t::f32);
        const size_t ept   = use_vec ? 4 : 1;
        const size_t vec_n = (num_elements + ept - 1) / ept;
        const size_t tpb   = std::min(max_threads_per_block, vec_n);
        threads_per_block_ = tpb == 0 ? 1 : tpb;
        num_blocks_        = (vec_n + threads_per_block_ - 1) / threads_per_block_;
    }

    template <typename... Args>
    void operator()(hipStream_t stream, const void* in, void* out, Args&&... args) const {
        ElementTypes::switch_(element_type_, *this, stream, in, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    constexpr void callKernel(hipStream_t stream, const void* in, void* out, Args&&... args) const noexcept {
        // Use vectorized kernel for f16/f32 with no extra args
        if constexpr (std::is_same_v<T, __half> && sizeof...(args) == 0) {
            elementwise_unary_fp16_vec4<OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const __half*>(in), num_elements_, static_cast<__half*>(out));
        } else if constexpr (std::is_same_v<T, float> && sizeof...(args) == 0) {
            elementwise_unary_fp32_vec4<OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const float*>(in), num_elements_, static_cast<float*>(out));
        } else {
            elementwise_unary<T, OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const T*>(in), num_elements_, static_cast<T*>(out),
                std::forward<Args>(args)...);
        }
    }

    template <typename T, typename... Args>
    constexpr void case_(hipStream_t stream, const void* in, void* out, Args&&... args) const noexcept {
        callKernel<T>(stream, in, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    void default_(T t, hipStream_t, const void*, void*, Args...) const noexcept {
        throwTypeNotSupported(t);
    }

private:
    Type_t element_type_;
    size_t num_elements_;
    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
