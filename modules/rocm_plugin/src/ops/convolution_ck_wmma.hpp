// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CK WMMA convolution backend for OpenVINO ROCm Plugin.
// Uses AMD Composable Kernel's WMMA conv forward for RDNA3/4 (gfx1100/gfx1201).
//
// Design: on first Execute, profiles both rocMLIR and CK WMMA kernels and
// permanently caches the faster choice. Subsequent executions pay zero profiling
// overhead (single mutex-guarded hash-map lookup per unique shape).
//
// CK WMMA requires NHWC layout; NCHW↔NHWC transposes are included in timing.
// The cache stores: 0 = use rocMLIR, 1 = use CK WMMA.

#pragma once

#ifdef ENABLE_CK_WMMA

#include <rocm_operation_base.hpp>
#include "convolution_components/convolution_components.hpp"
#include "rocm_creation_context.hpp"
#include "rocm/rocmlir_kernel_cache.hpp"
#include "rocm/rocmlir_compiler.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <mutex>
#include <unordered_map>
#include <string>
#include <cstring>
#include <cstdint>

// ── CK headers ────────────────────────────────────────────────────────────────
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/stream_config.hpp"

// StreamConfig is defined in global namespace (see /opt/rocm/include/ck/stream_config.hpp)

namespace ov {
namespace rocm_gpu {

// ── CK type aliases ──────────────────────────────────────────────────────────
using CkF16   = ck::half_t;
using CkIdx   = ck::index_t;
using GNHWC   = ck::tensor_layout::convolution::GNHWC;
using GKYXC   = ck::tensor_layout::convolution::GKYXC;
using GNHWK   = ck::tensor_layout::convolution::GNHWK;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using CkAdd   = ck::tensor_operation::element_wise::Add;
using CkAddSilu = ck::tensor_operation::element_wise::AddSilu;
using Spec    = ck::tensor_operation::device::ConvolutionForwardSpecialization;
namespace ck_inst = ck::tensor_operation::device::instance;

// ── NCHW↔NHWC transpose launchers (defined in convolution_ck_wmma.cpp) ──────
void launch_nchw_to_nhwc(const __half* src, __half* dst,
                          int N, int C, int H, int W, hipStream_t stream);
void launch_nhwc_to_nchw(const __half* src, __half* dst,
                          int N, int C, int H, int W, hipStream_t stream);

using CkAddSilu = ck::tensor_operation::element_wise::AddSilu;

// ── CK WMMA runner (Conv+Bias, NHWC layout) ─────────────────────────────────
// Returns true on success.
inline bool run_ck_wmma_conv_bias(
        const CkF16* X_nhwc, const CkF16* W_nhwc, const CkF16* bias, CkF16* Y_nhwc,
        int N, int G, int Cpg, int Hi, int Wi, int K,
        int kH, int kW, int padH, int padW, int sH, int sW,
        hipStream_t stream)
{
    using DsL = ck::Tuple<GNHWK>;
    using DsT = ck::Tuple<CkF16>;
    using Epi = CkAdd;
    using Spec = ck::tensor_operation::device::ConvolutionForwardSpecialization;

    int Ho = (Hi + 2 * padH - kH) / sH + 1;
    int Wo = (Wi + 2 * padW - kW) / sW + 1;

    // GNHWC strides (G is outer, C is inner)
    auto gnhwc_len = [](int G, int N, int H, int W, int C) {
        return std::array<CkIdx, 5>{G, N, H, W, C};
    };
    auto gnhwc_str = [](int G, int N, int H, int W, int C) {
        (void)G; (void)N; (void)H; (void)W;
        CkIdx sC = 1, sW = C, sH = C * W, sN = C * W * H, sG = C * W * H * N;
        return std::array<CkIdx, 5>{sG, sN, sH, sW, sC};
    };
    auto gkyxc_str = [](int G, int K, int kH, int kW, int C) {
        (void)G; (void)K; (void)kH; (void)kW;
        CkIdx sC = 1, sX = C, sY = C * kW, sK = C * kW * kH, sG = C * kW * kH * K;
        return std::array<CkIdx, 5>{sG, sK, sY, sX, sC};
    };

    // Bias: broadcast over N, H, W (stride 0 for those dims, stride 1 for K)
    std::array<const void*, 1> ds_ptrs = {bias};
    std::array<std::array<CkIdx, 5>, 1> ds_lens = {gnhwc_len(G, N, Ho, Wo, K)};
    std::array<std::array<CkIdx, 5>, 1> ds_strs = {
        std::array<CkIdx, 5>{(CkIdx)K, 0, 0, 0, 1}
    };

    // Select ConvSpec
    Spec convSpec = Spec::Default;
    if (kH == 1 && kW == 1 && padH == 0 && padW == 0 && sH == 1 && sW == 1)
        convSpec = Spec::Filter1x1Stride1Pad0;
    else if (kH == 1 && kW == 1 && padH == 0 && padW == 0)
        convSpec = Spec::Filter1x1Pad0;

    // Try each instance
    using Instances = typename ck_inst::device_grouped_conv_fwd_wmma_f16_instances<
        2, GNHWC, GKYXC, DsL, GNHWK, DsT, Epi, Spec::Default>;
    Instances instances{};
    bool found = false;

    auto try_one = [&](auto& dev) {
        if (found) return;
        auto arg = dev.MakeArgumentPointer(
            X_nhwc, W_nhwc, ds_ptrs, Y_nhwc,
            gnhwc_len(G, N, Hi, Wi, Cpg), gnhwc_str(G, N, Hi, Wi, Cpg),
            std::array<CkIdx,5>{G, K, kH, kW, Cpg}, gkyxc_str(G, K, kH, kW, Cpg),
            ds_lens, ds_strs,
            gnhwc_len(G, N, Ho, Wo, K), gnhwc_str(G, N, Ho, Wo, K),
            {sH, sW}, {1, 1}, {padH, padW}, {padH, padW},
            PassThrough{}, PassThrough{}, Epi{});
        if (dev.IsSupportedArgument(arg.get())) {
            dev.MakeInvokerPointer()->Run(arg.get(), StreamConfig{stream, false});
            found = true;
        }
    };
    std::apply([&](auto&... devs) { (try_one(devs), ...); }, instances);
    return found;
}

// ── CK WMMA runner (Conv+Bias+SiLU, NHWC layout) ─────────────────────────────
inline bool run_ck_wmma_conv_bias_silu(
        const CkF16* X_nhwc, const CkF16* W_nhwc, const CkF16* bias, CkF16* Y_nhwc,
        int N, int G, int Cpg, int Hi, int Wi, int K,
        int kH, int kW, int padH, int padW, int sH, int sW,
        hipStream_t stream)
{
    using DsL = ck::Tuple<GNHWK>;
    using DsT = ck::Tuple<CkF16>;
    using Epi = CkAddSilu;

    int Ho = (Hi + 2 * padH - kH) / sH + 1;
    int Wo = (Wi + 2 * padW - kW) / sW + 1;

    auto gnhwc_len = [](int G, int N, int H, int W, int C) {
        return std::array<CkIdx, 5>{G, N, H, W, C};
    };
    auto gnhwc_str = [](int G, int N, int H, int W, int C) {
        (void)G; (void)N; (void)H; (void)W;
        CkIdx sC=1, sW=C, sH=C*W, sN=C*W*H, sG=C*W*H*N;
        return std::array<CkIdx,5>{sG,sN,sH,sW,sC};
    };
    auto gkyxc_str = [](int G, int K, int kH, int kW, int C) {
        (void)G; (void)K; (void)kH; (void)kW;
        CkIdx sC=1, sX=C, sY=C*kW, sK=C*kW*kH, sG=C*kW*kH*K;
        return std::array<CkIdx,5>{sG,sK,sY,sX,sC};
    };

    std::array<const void*, 1> ds_ptrs = {bias};
    std::array<std::array<CkIdx,5>, 1> ds_lens = {gnhwc_len(G, N, Ho, Wo, K)};
    std::array<std::array<CkIdx,5>, 1> ds_strs = {
        std::array<CkIdx,5>{(CkIdx)K, 0, 0, 0, 1}
    };

    using Instances = typename ck_inst::device_grouped_conv_fwd_wmma_f16_instances<
        2, GNHWC, GKYXC, DsL, GNHWK, DsT, Epi, Spec::Default>;
    Instances instances{};
    bool found = false;

    auto try_one = [&](auto& dev) {
        if (found) return;
        auto arg = dev.MakeArgumentPointer(
            X_nhwc, W_nhwc, ds_ptrs, Y_nhwc,
            gnhwc_len(G,N,Hi,Wi,Cpg), gnhwc_str(G,N,Hi,Wi,Cpg),
            std::array<CkIdx,5>{G,K,kH,kW,Cpg}, gkyxc_str(G,K,kH,kW,Cpg),
            ds_lens, ds_strs,
            gnhwc_len(G,N,Ho,Wo,K), gnhwc_str(G,N,Ho,Wo,K),
            {sH,sW}, {1,1}, {padH,padW}, {padH,padW},
            PassThrough{}, PassThrough{}, Epi{});
        if (dev.IsSupportedArgument(arg.get())) {
            dev.MakeInvokerPointer()->Run(arg.get(), StreamConfig{stream, false});
            found = true;
        }
    };
    std::apply([&](auto&... devs){ (try_one(devs), ...); }, instances);
    return found;
}

// ── Per-shape backend selection cache ───────────────────────────────────────
struct CkWmmaKernelSelector {
    // Cache: shape hash → backend (0=rocMLIR, 1=CK WMMA, -1=not determined)
    static CkWmmaKernelSelector& global() {
        static CkWmmaKernelSelector s;
        return s;
    }

    int get(size_t shape_hash) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = cache_.find(shape_hash);
        if (it != cache_.end()) return it->second;
        return -1;
    }

    void set(size_t shape_hash, int backend) {
        std::lock_guard<std::mutex> lk(mu_);
        cache_[shape_hash] = backend;
    }

private:
    std::mutex mu_;
    std::unordered_map<size_t, int> cache_;
};

// ── ConvolutionCkWmma: Op class with rocMLIR vs CK profiling ─────────────────

class ConvolutionCkWmma : public OperationBase {
public:
    ConvolutionCkWmma(const CreationContext& ctx,
                       const ov::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds,
                       const Convolution::Details::ConvolutionParams& params);

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers& wbs) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        // Can't be in HIP graph during profiling phase; FULL after warmup
        return rocmGraphCompatibility::NONE;
    }

private:
    // rocMLIR params (always available as fallback)
    rocmlir::ConvParams rocmlir_params_;
    const rocmlir::KernelEntry* rocmlir_kernel_ = nullptr;

    // Shape descriptor for cache lookup
    size_t shape_hash_ = 0;

    // CK WMMA params
    int N_, G_, Cpg_, Hi_, Wi_, K_;
    int kH_, kW_, padH_, padW_, sH_, sW_;
    bool is_fp16_ = true;

    // Temporary NHWC buffers size
    size_t in_nhwc_elems_  = 0;
    size_t out_nhwc_elems_ = 0;
    size_t flt_nhwc_elems_ = 0;

    int OH_, OW_;

    void run_rocmlir(const InferenceRequestContext& ctx,
                     Inputs inputs, Outputs outputs,
                     const Workbuffers& wbs) const;

    void run_ck_wmma(const InferenceRequestContext& ctx,
                     Inputs inputs, Outputs outputs,
                     void* tmp_in_nhwc, void* tmp_flt_nhwc, void* tmp_out_nhwc) const;

    float time_kernel(const InferenceRequestContext& ctx,
                       Inputs inputs, Outputs outputs, const Workbuffers& wbs,
                       void* tmp_in_nhwc, void* tmp_flt_nhwc, void* tmp_out_nhwc,
                       int backend, int n_runs) const;
};

// ── FusedConvolutionCkWmma: fused conv+bias(+silu) via CK WMMA ─────────────
// Handles FusedConvolution nodes (Conv + bias + optional SiLU activation).
// On first Execute: profiles rocMLIR vs CK WMMA and caches the best choice.
// This is the hot path for yolo26x (all Conv nodes are fused with bias+SiLU).

class FusedConvolutionCkWmma : public OperationBase {
public:
    FusedConvolutionCkWmma(const CreationContext& ctx,
                            const ov::Node& node,
                            IndexCollection&& inputIds,
                            IndexCollection&& outputIds,
                            const Convolution::Details::FusedConvolutionParams& params);

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers& wbs) const override;

    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::NONE;  // profiling phase requires eager mode
    }

private:
    // rocMLIR fallback kernel (always compiled)
    rocmlir::ConvParams rocmlir_params_;
    const rocmlir::KernelEntry* rocmlir_kernel_ = nullptr;

    // Shape cache key
    size_t shape_hash_ = 0;

    // CK WMMA params
    int N_, G_, Cpg_, Hi_, Wi_, K_;
    int kH_, kW_, padH_, padW_, sH_, sW_;
    bool use_silu_ = false;   // true → Conv+Bias+SiLU; false → Conv+Bias

    // Buffer sizes for layout conversion
    size_t in_nhwc_elems_  = 0;
    size_t out_nhwc_elems_ = 0;
    size_t flt_nhwc_elems_ = 0;
    int    OH_ = 0, OW_ = 0;

    // Bias buffer size (K elements)
    size_t bias_elems_ = 0;
    bool has_add_ = false;

    void run_rocmlir(const InferenceRequestContext& ctx,
                     Inputs inputs, Outputs outputs,
                     const Workbuffers& wbs) const;

    void run_ck(const InferenceRequestContext& ctx,
                Inputs inputs, Outputs outputs,
                void* tmp_in, void* tmp_flt, void* tmp_out) const;

    float time_ms(const InferenceRequestContext& ctx,
                  Inputs inputs, Outputs outputs, const Workbuffers& wbs,
                  void* tmp_in, void* tmp_flt, void* tmp_out,
                  int backend, int runs) const;
};

}  // namespace rocm_gpu
}  // namespace ov

#endif  // ENABLE_CK_WMMA
