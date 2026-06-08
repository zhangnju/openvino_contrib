// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Fused elementwise kernel: executes a chain of elementwise ops in a single
// GPU kernel launch, reading each input once and writing the output once.
//
// Design inspired by AITemplate's fuse_elementwise and LingLong's FusedElementwise.
// Eliminates N-1 VRAM round-trips for a chain of N elementwise ops.

#include "fused_elementwise.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// ─── Device functions per operation kind ─────────────────────────────────────

__device__ inline float fused_ew_apply(float acc, float aux, uint8_t op, float param) {
    switch(static_cast<FusedEwOp>(op)) {
        case FusedEwOp::Relu:       return acc > 0.f ? acc : 0.f;
        case FusedEwOp::Sigmoid:    return 1.f / (1.f + expf(-acc));
        case FusedEwOp::SiLU:       return acc * (1.f / (1.f + expf(-acc)));
        case FusedEwOp::Tanh:       return tanhf(acc);
        case FusedEwOp::Gelu:       return acc * 0.5f * (1.f + erff(acc * 0.7071067811865476f));
        case FusedEwOp::HardSwish:  return acc * fmaxf(0.f, fminf(1.f, acc / 6.f + 0.5f));
        case FusedEwOp::Abs:        return fabsf(acc);
        case FusedEwOp::Neg:        return -acc;
        case FusedEwOp::Sqrt:       return sqrtf(acc);
        case FusedEwOp::Exp:        return expf(acc);
        case FusedEwOp::Log:        return logf(acc);
        case FusedEwOp::Erf:        return erff(acc);
        case FusedEwOp::Add:        return acc + aux;
        case FusedEwOp::Sub:        return acc - aux;
        case FusedEwOp::Mul:        return acc * aux;
        case FusedEwOp::Div:        return acc / aux;
        case FusedEwOp::LeakyRelu:  return acc >= 0.f ? acc : acc * param;
        default:                    return acc;
    }
}

// ─── Main fused kernel (scalar fallback, works for all sizes) ────────────────

template<typename T>
__global__ void fused_elementwise_kernel(
        const T* __restrict__ primary_in,
        const void* const* __restrict__ aux_ptrs,  // per-step auxiliary inputs (nullptr if unary)
        T* __restrict__ out,
        int64_t n,
        const uint8_t* __restrict__ ops,
        const float*   __restrict__ params,
        int chain_len)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float acc = float(primary_in[i]);
    for (int s = 0; s < chain_len; ++s) {
        float aux = 0.f;
        if (aux_ptrs && aux_ptrs[s]) {
            aux = float(reinterpret_cast<const T*>(aux_ptrs[s])[i]);
        }
        acc = fused_ew_apply(acc, aux, ops[s], params[s]);
    }
    out[i] = T(acc);
}

// ─── Vectorized FP16 variant: 4 elements per thread ──────────────────────────

template<typename T>
__global__ void fused_elementwise_kernel_vec4(
        const T* __restrict__ primary_in,
        const void* const* __restrict__ aux_ptrs,
        T* __restrict__ out,
        int64_t n,
        const uint8_t* __restrict__ ops,
        const float*   __restrict__ params,
        int chain_len)
{
    int64_t base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= n) return;

    // Load up to 4 elements
    float v[4] = {};
    int count = (int)min((int64_t)4, n - base);
    for (int k = 0; k < count; ++k)
        v[k] = float(primary_in[base + k]);

    // Apply chain to each element
    for (int k = 0; k < count; ++k) {
        float acc = v[k];
        for (int s = 0; s < chain_len; ++s) {
            float aux = 0.f;
            if (aux_ptrs && aux_ptrs[s]) {
                aux = float(reinterpret_cast<const T*>(aux_ptrs[s])[base + k]);
            }
            acc = fused_ew_apply(acc, aux, ops[s], params[s]);
        }
        v[k] = acc;
    }

    // Store
    for (int k = 0; k < count; ++k)
        out[base + k] = T(v[k]);
}

// ─── Public launch function ───────────────────────────────────────────────────

void launch_fused_elementwise(
        const void* primary_in,
        const void* const* aux_ptrs_device,
        void* out,
        int64_t n,
        const uint8_t* ops_device,
        const float* params_device,
        int chain_len,
        bool is_fp16,
        hipStream_t stream)
{
    constexpr int BLOCK = 256;

    // Use vectorized kernel (4 elems/thread) for large tensors
    if (n >= 4096) {
        int64_t vec_n = (n + 3) / 4;
        int grid = (int)((vec_n + BLOCK - 1) / BLOCK);
        if (is_fp16) {
            hipLaunchKernelGGL(fused_elementwise_kernel_vec4<__half>,
                grid, BLOCK, 0, stream,
                reinterpret_cast<const __half*>(primary_in),
                aux_ptrs_device,
                reinterpret_cast<__half*>(out),
                n, ops_device, params_device, chain_len);
        } else {
            hipLaunchKernelGGL(fused_elementwise_kernel_vec4<float>,
                grid, BLOCK, 0, stream,
                reinterpret_cast<const float*>(primary_in),
                aux_ptrs_device,
                reinterpret_cast<float*>(out),
                n, ops_device, params_device, chain_len);
        }
    } else {
        int grid = (int)((n + BLOCK - 1) / BLOCK);
        if (is_fp16) {
            hipLaunchKernelGGL(fused_elementwise_kernel<__half>,
                grid, BLOCK, 0, stream,
                reinterpret_cast<const __half*>(primary_in),
                aux_ptrs_device,
                reinterpret_cast<__half*>(out),
                n, ops_device, params_device, chain_len);
        } else {
            hipLaunchKernelGGL(fused_elementwise_kernel<float>,
                grid, BLOCK, 0, stream,
                reinterpret_cast<const float*>(primary_in),
                aux_ptrs_device,
                reinterpret_cast<float*>(out),
                n, ops_device, params_device, chain_len);
        }
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
