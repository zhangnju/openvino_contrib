// Winograd F(4,3) convolution op for 3×3 stride=1 dilation=1 groups=1 fp16.
// On first Execute, profiles Winograd vs rocMLIR and permanently caches the
// faster backend per unique shape. Subsequent executions dispatch with zero
// profiling overhead (single hash-map lookup).
#pragma once

#ifdef ENABLE_WINOGRAD

#include <rocm_operation_base.hpp>
#include <mutex>
#include <unordered_map>
#include "convolution_components/convolution_components.hpp"
#include "rocm/rocmlir_kernel_cache.hpp"
#include "rocm/rocmlir_compiler.hpp"
#include "kernels/winograd_conv.hpp"

namespace ov {
namespace rocm_gpu {

// ── Per-shape backend selection cache (thread-safe) ──────────────────────
struct WinogradSelector {
    static WinogradSelector& global() {
        static WinogradSelector s;
        return s;
    }
    // Returns: 0=rocMLIR, 1=Winograd, -1=not yet profiled
    int get(size_t shape_hash) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = cache_.find(shape_hash);
        return (it != cache_.end()) ? it->second : -1;
    }
    void set(size_t shape_hash, int backend) {
        std::lock_guard<std::mutex> lk(mu_);
        cache_[shape_hash] = backend;
    }
private:
    std::mutex mu_;
    std::unordered_map<size_t, int> cache_;
};

// ── WinogradConvOp ──────────────────────────────────────────────────────
class WinogradConvOp : public OperationBase {
public:
    // Plain Convolution constructor
    WinogradConvOp(const CreationContext& ctx, const ov::Node& node,
                   IndexCollection&& in, IndexCollection&& out,
                   const Convolution::Details::ConvolutionParams& params);

    // FusedConvolution constructor (conv + bias + optional relu)
    WinogradConvOp(const CreationContext& ctx, const ov::Node& node,
                   IndexCollection&& in, IndexCollection&& out,
                   const Convolution::Details::FusedConvolutionParams& params);

    void Execute(const InferenceRequestContext& ctx, Inputs inputs,
                 Outputs outputs, const Workbuffers& wb) const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::NONE;  // profiling on first run
    }

    static bool isEligible(const Convolution::Details::ConvolutionParams& p);
    static bool isEligible(const Convolution::Details::FusedConvolutionParams& p);

private:
    int N_, C_, H_, W_, K_;
    bool fused_;
    bool relu_;

    // Winograd buffers
    size_t filter_xform_bytes_;
    size_t winograd_ws_bytes_;

    // rocMLIR fallback
    rocmlir::ConvParams rocmlir_params_;
    const rocmlir::KernelEntry* rocmlir_kernel_ = nullptr;
    size_t rocmlir_ws_bytes_ = 0;

    // Backend selection
    size_t shape_hash_ = 0;

    mutable std::once_flag filter_init_flag_;

    void run_winograd(const InferenceRequestContext& ctx,
                      Inputs inputs, Outputs outputs,
                      const Workbuffers& wb) const;
    void run_rocmlir(const InferenceRequestContext& ctx,
                     Inputs inputs, Outputs outputs,
                     const Workbuffers& wb) const;
    float time_backend(const InferenceRequestContext& ctx,
                       Inputs inputs, Outputs outputs,
                       const Workbuffers& wb,
                       int backend, int runs) const;
};

}  // namespace rocm_gpu
}  // namespace ov

#endif  // ENABLE_WINOGRAD
