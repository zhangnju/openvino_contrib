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
#include "numpy_broadcast_mapper.hpp"
#include "tensor_helpers.hpp"
#include "type_validator.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

// ── Scalar fallback ───────────────────────────────────────────────────────
template <typename T, typename OP, typename... Args>
__global__ void elementwise_binary(const T* in0, const T* in1, T* out, size_t out_num_elements, Args... args) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_num_elements) {
        out[i] = OP::op(in0[i], in1[i], args...);
    }
}

// ── Vectorized FP16: 4 elements per thread via half2 pairs ───────────────
// Each thread processes 2 half2 = 4 __half elements with 128-bit loads.
// Reduces kernel launch overhead and improves memory bandwidth utilization.
template <typename OP>
__global__ void elementwise_binary_fp16_vec4(const __half* __restrict__ in0,
                                              const __half* __restrict__ in1,
                                              __half* __restrict__ out,
                                              size_t n) {
    const size_t base = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        // 128-bit load: 2 half2 values each
        const __half2* a2 = reinterpret_cast<const __half2*>(in0 + base);
        const __half2* b2 = reinterpret_cast<const __half2*>(in1 + base);
        __half2*        c2 = reinterpret_cast<__half2*>(out  + base);
        // Process pair 0
        __half2 a0 = a2[0], b0 = b2[0];
        __half r0_lo = OP::op(a0.x, b0.x);
        __half r0_hi = OP::op(a0.y, b0.y);
        c2[0] = __half2(r0_lo, r0_hi);
        // Process pair 1
        __half2 a1 = a2[1], b1 = b2[1];
        __half r1_lo = OP::op(a1.x, b1.x);
        __half r1_hi = OP::op(a1.y, b1.y);
        c2[1] = __half2(r1_lo, r1_hi);
    } else {
        // Tail elements
        for (size_t i = base; i < n; ++i) {
            out[i] = OP::op(in0[i], in1[i]);
        }
    }
}

// ── Vectorized FP32: 4 elements per thread via float4 ───────────────────
template <typename OP>
__global__ void elementwise_binary_fp32_vec4(const float* __restrict__ in0,
                                              const float* __restrict__ in1,
                                              float* __restrict__ out,
                                              size_t n) {
    const size_t base = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        const float4 a = *reinterpret_cast<const float4*>(in0 + base);
        const float4 b = *reinterpret_cast<const float4*>(in1 + base);
        float4 c;
        c.x = OP::op(a.x, b.x);
        c.y = OP::op(a.y, b.y);
        c.z = OP::op(a.z, b.z);
        c.w = OP::op(a.w, b.w);
        *reinterpret_cast<float4*>(out + base) = c;
    } else {
        for (size_t i = base; i < n; ++i) {
            out[i] = OP::op(in0[i], in1[i]);
        }
    }
}

template <typename T, typename OP, typename... Args>
__global__ void elementwise_binary_broadcasting(const T* in0,
                                                NumpyBroadcastMapper in0_mapper,
                                                const T* in1,
                                                NumpyBroadcastMapper in1_mapper,
                                                T* out,
                                                size_t out_num_elements,
                                                Args... args) {
    const unsigned out_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_index < out_num_elements) {
        const unsigned in0_index = in0_mapper.srcIndex(out_index);
        const unsigned in1_index = in1_mapper.srcIndex(out_index);
        out[out_index] = OP::op(in0[in0_index], in1[in1_index], args...);
    }
}

template <typename ElementTypes, template <typename> typename OP>
class ElementwiseBinary {
public:
    ElementwiseBinary(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
        : num_blocks_{}, threads_per_block_{}, element_type_{element_type}, out_num_elements_{out_num_elements} {
        TypeValidator<ElementTypes>::check(element_type);
        // Vectorized kernels handle 4 elements per thread for f16/f32 non-broadcast path.
        // The scalar grid is also computed for the broadcasting path.
        const bool use_vec = (element_type == Type_t::f16 || element_type == Type_t::f32);
        const size_t ept   = use_vec ? 4 : 1;
        const size_t vec_n = (out_num_elements + ept - 1) / ept;
        const size_t tpb   = std::min(max_threads_per_block, vec_n);
        threads_per_block_ = tpb == 0 ? 1 : tpb;
        num_blocks_        = (vec_n + threads_per_block_ - 1) / threads_per_block_;
        // Scalar grid for broadcasting path
        std::tie(scalar_num_blocks_, scalar_threads_per_block_) =
            calculateElementwiseGrid(out_num_elements, max_threads_per_block);
    }

    template <typename... Args>
    void operator()(hipStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out,
                    Args&&... args) const {
        if (in0_mapper.identity() && in1_mapper.identity()) {
            (*this)(stream, in0, in1, out, std::forward<Args>(args)...);
        } else {
            ElementTypes::switch_(
                element_type_, *this, stream, in0, in0_mapper, in1, in1_mapper, out, std::forward<Args>(args)...);
        }
    }

    /**
     * Simple variant of elementwise invocation for the case when all input and output shapes are the same.
     * It is expected to be more quick then generic variant which supports broadcasting.
     */
    template <typename... Args>
    void operator()(hipStream_t stream, const void* in0, const void* in1, void* out, Args&&... args) const {
        ElementTypes::switch_(element_type_, *this, stream, in0, in1, out, std::forward<Args>(args)...);
    }

private:
    friend ElementTypes;

    template <typename T, typename... Args>
    constexpr void case_(hipStream_t stream,
                         const void* in0,
                         const NumpyBroadcastMapper& in0_mapper,
                         const void* in1,
                         const NumpyBroadcastMapper& in1_mapper,
                         void* out,
                         Args&&... args) const noexcept {
        // Broadcasting path always uses scalar kernel (index remapping is per-element)
        elementwise_binary_broadcasting<T, OP<T>>
            <<<scalar_num_blocks_, scalar_threads_per_block_, 0, stream>>>(
                                                             static_cast<const T*>(in0),
                                                             in0_mapper,
                                                             static_cast<const T*>(in1),
                                                             in1_mapper,
                                                             static_cast<T*>(out),
                                                             out_num_elements_,
                                                             std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    void default_(T t,
                  hipStream_t,
                  const void*,
                  const NumpyBroadcastMapper&,
                  const void*,
                  const NumpyBroadcastMapper&,
                  void*,
                  Args...) const noexcept {
        throwTypeNotSupported(t);
    }

    template <typename T, typename... Args>
    constexpr void case_(hipStream_t stream, const void* in0, const void* in1, void* out, Args&&... args) const
        noexcept {
        // Non-broadcast fast path: use vectorized kernel for f16/f32, scalar for others.
        // Only vectorize when no extra args (simple binary ops without additional params).
        if constexpr (std::is_same_v<T, __half> && sizeof...(args) == 0) {
            elementwise_binary_fp16_vec4<OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const __half*>(in0),
                static_cast<const __half*>(in1),
                static_cast<__half*>(out),
                out_num_elements_);
        } else if constexpr (std::is_same_v<T, float> && sizeof...(args) == 0) {
            elementwise_binary_fp32_vec4<OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const float*>(in0),
                static_cast<const float*>(in1),
                static_cast<float*>(out),
                out_num_elements_);
        } else {
            elementwise_binary<T, OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(
                static_cast<const T*>(in0),
                static_cast<const T*>(in1),
                static_cast<T*>(out),
                out_num_elements_,
                std::forward<Args>(args)...);
        }
    }

    template <typename T, typename... Args>
    void default_(T t, hipStream_t, const void*, const void*, void*, Args...) const noexcept {
        throwTypeNotSupported(t);
    }

private:
    size_t num_blocks_;              // vectorized grid (4 elems/thread for f16/f32)
    size_t threads_per_block_;       // vectorized threads per block
    size_t scalar_num_blocks_;       // scalar grid (1 elem/thread, for broadcast path)
    size_t scalar_threads_per_block_;
    Type_t element_type_;
    size_t out_num_elements_;
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
