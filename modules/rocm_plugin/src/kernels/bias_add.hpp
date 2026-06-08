// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <hip/hip_runtime.h>

namespace ov { namespace rocm_gpu { namespace kernel {

void launch_bias_add(void* output, const void* bias,
                      int N, int K, int H, int W,
                      bool fp16, hipStream_t stream);

// Vectorized pe 3-way add: out[i] += a[i] + b[k(i)]
// Uses __half2 to process 2 FP16 values per thread (2× memory throughput).
// a_flat: flat [K*HW] tensor (pe conv output),  b_k: broadcast bias [K]
// Both added to output (attn_out) in-place.
// total_elems must be even for __half2 vectorization.
void launch_pe_add_fp16(void* output,          // attn_out [K,H,W] flat, updated in-place
                        const void* a_flat,    // conv(V) output [K*H*W]
                        const void* b_k,       // bias [K]
                        int K, int HW,         // K channels, H*W spatial
                        hipStream_t stream);

// 3-way add (no extra D2D copy): dst[i] = src[i] + conv[i] + bias[k(i)]
void launch_pe_add3_fp16(void* dst,            // fgc_out [K*HW], output
                         const void* src,      // AV_out [K*HW], input (attn result)
                         const void* conv,     // conv(V) output [K*HW]
                         const void* bias_k,   // bias [K]
                         int K, int HW,
                         hipStream_t stream);

// Fused depthwise conv3×3 + bias + AV_out add in ONE pass (no intermediate pe_work buffer).
// fgc_out[k,h,w] = depthwise_conv3x3(V[k,h,w], filter[k]) + bias[k] + AV_out[k,h,w]
// Saves 1 memory traversal vs pe_conv (writes pe_work) + pe_add3 (reads pe_work).
void launch_fused_pe_conv_add(
    void* fgc_out,          // output [K*H*W]
    const void* V,          // input V [K*H*W]  (V from VariadicSplit)
    const void* filter,     // depthwise filter [K*9]
    const void* AV_out,     // attn result [K*H*W]
    const void* bias,       // bias [K]
    int K, int H, int W,
    hipStream_t stream);

} } }
