// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/float16.hpp>
#include <hip/hip_fp16.h>

#include "details/tensor_helpers.hpp"
#include "swish.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

// ── Vectorized FP16 Swish: 4 elements per thread using half2 SIMD ──────────
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Uses half2 to process 2 elements simultaneously with native RDNA instructions.
// Each thread handles 2 half2 pairs = 4 f16 elements for better latency hiding.
static __global__ void SwishOpImpl_fp16_vec(const __half* __restrict__ in,
                                             float beta_f,
                                             __half* __restrict__ out,
                                             size_t n) {
    const size_t base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        // Fast path: fully vectorized, 4 elements at once
        const __half2* in2  = reinterpret_cast<const __half2*>(in + base);
        __half2*        out2 = reinterpret_cast<__half2*>(out + base);

        // Load 2 pairs of f16
        __half2 x0 = in2[0];
        __half2 x1 = in2[1];

        // sigmoid(x) = 1 / (1 + exp(-x)) — computed in float for accuracy
        // then convert back to half2
        float2 x0f = __half22float2(x0);
        float2 x1f = __half22float2(x1);

        // SiLU: x * sigmoid(x) = x / (1 + exp(-x*beta))
        float2 r0, r1;
        r0.x = x0f.x / (1.0f + expf(-x0f.x * beta_f));
        r0.y = x0f.y / (1.0f + expf(-x0f.y * beta_f));
        r1.x = x1f.x / (1.0f + expf(-x1f.x * beta_f));
        r1.y = x1f.y / (1.0f + expf(-x1f.y * beta_f));

        out2[0] = __float22half2_rn(r0);
        out2[1] = __float22half2_rn(r1);
    } else {
        // Tail: handle remaining 0-3 elements
        for (size_t i = base; i < n; ++i) {
            float x = __half2float(in[i]);
            out[i] = __float2half(x / (1.0f + expf(-x * beta_f)));
        }
    }
}

// Scalar fallback for f16 (when n < 4 or unaligned)
static __global__ void SwishOpImpl_fp16(const __half* in, float beta, __half* out, size_t n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __half2float(in[i]);
        out[i] = __float2half(x / (1.0f + expf(-x * beta)));
    }
}

// Vectorized FP32 Swish: 4 elements per thread
static __global__ void SwishOpImpl_fp32_vec(const float* __restrict__ in,
                                             float beta,
                                             float* __restrict__ out,
                                             size_t n) {
    const size_t base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < n) {
        const float4* in4  = reinterpret_cast<const float4*>(in + base);
        float4*        out4 = reinterpret_cast<float4*>(out + base);
        float4 x = *in4;
        float4 r;
        r.x = x.x / (1.0f + expf(-x.x * beta));
        r.y = x.y / (1.0f + expf(-x.y * beta));
        r.z = x.z / (1.0f + expf(-x.z * beta));
        r.w = x.w / (1.0f + expf(-x.w * beta));
        *out4 = r;
    } else {
        for (size_t i = base; i < n; ++i) {
            out[i] = in[i] / (1.0f + expf(-in[i] * beta));
        }
    }
}

Swish::Swish(Type_t element_type, size_t max_threads_per_block, size_t num_elements, double beta)
    : beta_{beta}
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = num_elements;

    // Vectorized kernels process 4 elements per thread → divide grid by 4
    const size_t elems_per_thread = 4;
    const size_t vec_elems = (num_elements + elems_per_thread - 1) / elems_per_thread;
    threads_per_block_ = static_cast<unsigned>(std::min(max_threads_per_block, vec_elems));
    if (threads_per_block_ == 0) threads_per_block_ = 1;
    num_blocks_ = static_cast<unsigned>((vec_elems + threads_per_block_ - 1) / threads_per_block_);
}

void Swish::operator()(hipStream_t stream, const void* in, void* out) const
{
    const float beta_f = static_cast<float>(beta_);
    const size_t n = num_elements_;
    switch(element_type_)
    {
        case Type_t::f32:
            SwishOpImpl_fp32_vec<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const float*)in, beta_f, (float*)out, n);
            break;
        case Type_t::f16:
            SwishOpImpl_fp16_vec<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const __half*)in, beta_f, (__half*)out, n);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Swish operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
