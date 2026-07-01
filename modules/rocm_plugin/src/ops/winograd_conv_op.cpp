// Winograd F(4,3) convolution op with profiling-based backend selection.
// On first Execute for each unique shape, profiles Winograd vs rocMLIR and
// permanently caches the faster backend. Falls back to rocMLIR when Winograd
// is slower (e.g. large-channel layers where workspace overhead dominates).
#ifdef ENABLE_WINOGRAD

#include "winograd_conv_op.hpp"
#include <openvino/op/convolution.hpp>
#include <openvino/core/except.hpp>
#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <iostream>

#include "transformer/nodes/fused_convolution.hpp"

namespace ov {
namespace rocm_gpu {

// ── Helper: build rocmlir::ConvParams from plugin ConvolutionParams ──────

static rocmlir::ConvParams to_rocmlir(
        const Convolution::Details::ConvolutionParams& p,
        const hipDeviceProp_t& props) {
    rocmlir::ConvParams rp;
    rp.N = (int)p.input_shape_[0];
    rp.C = (int)p.input_shape_[1];
    rp.H = (int)p.input_shape_[2];
    rp.W = (int)p.input_shape_[3];
    rp.K = (int)p.filter_shape_[0];
    rp.R = (int)p.filter_shape_[2];
    rp.S = (int)p.filter_shape_[3];
    rp.pad_h     = (int)p.padding_before_[0];
    rp.pad_w     = (int)p.padding_before_[1];
    rp.stride_h  = (int)p.strides_[0];
    rp.stride_w  = (int)p.strides_[1];
    rp.dilation_h = (int)p.dilations_[0];
    rp.dilation_w = (int)p.dilations_[1];
    rp.groups    = (int)p.groups_;
    rp.fp16      = (p.element_type_ == ov::element::Type_t::f16);
    std::string arch(props.gcnArchName);
    auto c = arch.find(':');
    if (c != std::string::npos) arch = arch.substr(0, c);
    rp.arch   = arch;
    rp.num_cu = props.multiProcessorCount;
    return rp;
}

// ── Eligibility check ────────────────────────────────────────────────────

bool WinogradConvOp::isEligible(const Convolution::Details::ConvolutionParams& p) {
    if (p.element_type_ != ov::element::f16) return false;
    if (p.NumberOfSpatialDims() != 2) return false;
    if (p.groups_ != 1) return false;
    if (p.filter_shape_[2] != 3 || p.filter_shape_[3] != 3) return false;
    if (p.strides_[0] != 1 || p.strides_[1] != 1) return false;
    if (p.dilations_[0] != 1 || p.dilations_[1] != 1) return false;
    if (p.padding_before_[0] != 1 || p.padding_before_[1] != 1) return false;
    if (p.padding_after_[0] != 1 || p.padding_after_[1] != 1) return false;
    return true;
}

bool WinogradConvOp::isEligible(const Convolution::Details::FusedConvolutionParams& p) {
    if (!isEligible(p.conv_)) return false;
    auto act = p.activation_;
    if (act != nodes::ActivationMode::NO_ACTIVATION &&
        act != nodes::ActivationMode::RELU)
        return false;
    if (p.add_shape_.has_value()) return false;
    return true;
}

// ── Plain Convolution constructor ────────────────────────────────────────

WinogradConvOp::WinogradConvOp(const CreationContext& ctx, const ov::Node& node,
                                 IndexCollection&& in, IndexCollection&& out,
                                 const Convolution::Details::ConvolutionParams& params)
    : OperationBase(ctx, node, std::move(in), std::move(out)),
      fused_(false), relu_(false)
{
    N_ = params.input_shape_[0];
    C_ = params.input_shape_[1];
    H_ = params.input_shape_[2];
    W_ = params.input_shape_[3];
    K_ = params.filter_shape_[0];

    filter_xform_bytes_ = kernel::WinogradConv::filterTransformBytes(K_, C_);
    winograd_ws_bytes_ = kernel::WinogradConv::workspaceBytes(N_, C_, H_, W_, K_);

    // Compile rocMLIR fallback
    auto props = ctx.device().props();
    rocmlir_params_ = to_rocmlir(params, props);
    rocmlir_kernel_ = &rocmlir::RocMLIRKernelCache::global().get_or_compile(rocmlir_params_);
    rocmlir_ws_bytes_ = rocmlir_kernel_->info.workspace_bytes;

    shape_hash_ = rocmlir_params_.hash() ^ static_cast<size_t>(0xA1E06CADULL);

    std::cerr << "[WinogradConv] N=" << N_ << " C=" << C_ << " H=" << H_
              << " W=" << W_ << " K=" << K_ << " (plain, profiled)"
              << std::endl;
}

// ── FusedConvolution constructor ─────────────────────────────────────────

WinogradConvOp::WinogradConvOp(const CreationContext& ctx, const ov::Node& node,
                                 IndexCollection&& in, IndexCollection&& out,
                                 const Convolution::Details::FusedConvolutionParams& params)
    : OperationBase(ctx, node, std::move(in), std::move(out)),
      fused_(true), relu_(false)
{
    N_ = params.conv_.input_shape_[0];
    C_ = params.conv_.input_shape_[1];
    H_ = params.conv_.input_shape_[2];
    W_ = params.conv_.input_shape_[3];
    K_ = params.conv_.filter_shape_[0];

    relu_ = (params.activation_ == nodes::ActivationMode::RELU);

    filter_xform_bytes_ = kernel::WinogradConv::filterTransformBytes(K_, C_);
    winograd_ws_bytes_ = kernel::WinogradConv::workspaceBytes(N_, C_, H_, W_, K_);

    // Compile rocMLIR fallback (fused conv+bias or conv+bias+relu)
    auto props = ctx.device().props();
    rocmlir_params_ = to_rocmlir(params.conv_, props);

    if (relu_) {
        rocmlir_kernel_ = &rocmlir::RocMLIRKernelCache::global()
            .get_or_compile_fused_bias_act(rocmlir_params_, rocmlir::Activation::ReLU);
    } else {
        rocmlir_kernel_ = &rocmlir::RocMLIRKernelCache::global()
            .get_or_compile_fused_bias(rocmlir_params_);
    }
    rocmlir_ws_bytes_ = rocmlir_kernel_->info.workspace_bytes;

    shape_hash_ = rocmlir_params_.hash() ^ static_cast<size_t>(0xB1E06FADULL);

    std::cerr << "[WinogradConv] N=" << N_ << " C=" << C_ << " H=" << H_
              << " W=" << W_ << " K=" << K_ << " bias relu=" << relu_
              << " (profiled)" << std::endl;
}

// ── Workbuffer request ───────────────────────────────────────────────────

WorkbufferRequest WinogradConvOp::GetWorkBufferRequest() const {
    // Immutable[0]: Winograd transformed filter [36, K, C] fp16
    // Mutable[0]:   max(winograd workspace, rocMLIR workspace)
    size_t mutable_bytes = std::max(winograd_ws_bytes_,
                                    rocmlir_ws_bytes_ > 0 ? rocmlir_ws_bytes_ : (size_t)1);
    return {{filter_xform_bytes_}, {mutable_bytes}};
}

void WinogradConvOp::InitSharedImmutableWorkbuffers(const Buffers&) {
    // Lazy init in Execute (need filter input pointer)
}

// ── Backend runners ──────────────────────────────────────────────────────

void WinogradConvOp::run_winograd(const InferenceRequestContext& ctx,
                                    Inputs inputs, Outputs outputs,
                                    const Workbuffers& wb) const {
    auto stream = ctx.getThreadContext().stream().get();
    auto blas_handle = ctx.getThreadContext().rocBlasHandle().get();

    const __half* d_input;
    const __half* d_filter;
    const __half* d_bias = nullptr;

    if (fused_) {
        using Idx = Convolution::Details::FusedConvolutionIndices;
        d_input  = reinterpret_cast<const __half*>(inputs[Idx::input].get());
        d_filter = reinterpret_cast<const __half*>(inputs[Idx::filter].get());
        d_bias   = reinterpret_cast<const __half*>(inputs[Idx::bias].get());
    } else {
        using Idx = Convolution::Details::ConvArgIndices;
        d_input  = reinterpret_cast<const __half*>(inputs[Idx::input].get());
        d_filter = reinterpret_cast<const __half*>(inputs[Idx::filter].get());
    }

    __half* d_output = reinterpret_cast<__half*>(outputs[0].get());
    __half* d_filter_xform = reinterpret_cast<__half*>(
        const_cast<void*>(wb.immutable_buffers[0].get()));
    __half* d_workspace = reinterpret_cast<__half*>(wb.mutable_buffers[0].get());

    // Lazy filter transform (runs exactly once, thread-safe)
    std::call_once(filter_init_flag_, [&]() {
        kernel::WinogradConv::filterTransform(d_filter, d_filter_xform, K_, C_, stream);
        hipStreamSynchronize(stream);
    });

    kernel::WinogradConv::forward(d_input, d_filter_xform, d_bias, d_output,
                                   d_workspace, N_, C_, H_, W_, K_, relu_,
                                   blas_handle, stream);
}

void WinogradConvOp::run_rocmlir(const InferenceRequestContext& ctx,
                                   Inputs inputs, Outputs outputs,
                                   const Workbuffers& wb) const {
    const auto& info = rocmlir_kernel_->info;
    hipStream_t stream = ctx.getThreadContext().stream().get();
    void* d_output = outputs[0].get();
    void* d_ws = wb.mutable_buffers[0].get();

    if (fused_) {
        using Idx = Convolution::Details::FusedConvolutionIndices;
        void* d_filter = const_cast<void*>(inputs[Idx::filter].get());
        void* d_input  = const_cast<void*>(inputs[Idx::input].get());
        void* d_bias   = const_cast<void*>(inputs[Idx::bias].get());
        // Arg order: legacy "mlir_convolution*" kernels use (input, filter, bias, output)
        // Patched kernels use (filter, input, bias, output)
        const bool legacy = (info.kernel_name.find("mlir_convolution") != std::string::npos);
        void* args_legacy[]  = { &d_input,  &d_filter, &d_bias, &d_output };
        void* args_patched[] = { &d_filter, &d_input,  &d_bias, &d_output };
        hipModuleLaunchKernel(
            rocmlir_kernel_->function,
            info.grid_x, 1, 1, info.block_x, 1, 1,
            0, stream,
            legacy ? args_legacy : args_patched, nullptr);
    } else {
        using Idx = Convolution::Details::ConvArgIndices;
        void* d_filter = const_cast<void*>(inputs[Idx::filter].get());
        void* d_input  = const_cast<void*>(inputs[Idx::input].get());
        void* args[] = { &d_filter, &d_input, &d_output, &d_ws };
        hipModuleLaunchKernel(
            rocmlir_kernel_->function,
            info.grid_x, 1, 1, info.block_x, 1, 1,
            0, stream,
            args, nullptr);
    }
}

// ── Profiling ────────────────────────────────────────────────────────────

float WinogradConvOp::time_backend(const InferenceRequestContext& ctx,
                                     Inputs inputs, Outputs outputs,
                                     const Workbuffers& wb,
                                     int backend, int runs) const {
    hipStream_t stream = ctx.getThreadContext().stream().get();
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        if (backend == 0) run_rocmlir(ctx, inputs, outputs, wb);
        else              run_winograd(ctx, inputs, outputs, wb);
    }
    hipStreamSynchronize(stream);

    // Timed runs
    hipEventRecord(ev0, stream);
    for (int i = 0; i < runs; ++i) {
        if (backend == 0) run_rocmlir(ctx, inputs, outputs, wb);
        else              run_winograd(ctx, inputs, outputs, wb);
    }
    hipEventRecord(ev1, stream);
    hipStreamSynchronize(stream);

    float ms = 0.f;
    hipEventElapsedTime(&ms, ev0, ev1);
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    return ms / runs;
}

// ── Execute ──────────────────────────────────────────────────────────────

void WinogradConvOp::Execute(const InferenceRequestContext& ctx,
                              Inputs inputs, Outputs outputs,
                              const Workbuffers& wb) const {
    int backend = WinogradSelector::global().get(shape_hash_);

    if (backend < 0) {
        // First call: profile both backends
        constexpr int PROFILE_RUNS = 10;

        // Must initialize Winograd filter transform before profiling
        {
            auto stream = ctx.getThreadContext().stream().get();
            const __half* d_filter;
            if (fused_) {
                using Idx = Convolution::Details::FusedConvolutionIndices;
                d_filter = reinterpret_cast<const __half*>(inputs[Idx::filter].get());
            } else {
                using Idx = Convolution::Details::ConvArgIndices;
                d_filter = reinterpret_cast<const __half*>(inputs[Idx::filter].get());
            }
            __half* d_filter_xform = reinterpret_cast<__half*>(
                const_cast<void*>(wb.immutable_buffers[0].get()));
            std::call_once(filter_init_flag_, [&]() {
                kernel::WinogradConv::filterTransform(d_filter, d_filter_xform, K_, C_, stream);
                hipStreamSynchronize(stream);
            });
        }

        float t_rocmlir = 1e9f;
        try { t_rocmlir = time_backend(ctx, inputs, outputs, wb, 0, PROFILE_RUNS); }
        catch (...) {}

        float t_winograd = 1e9f;
        try { t_winograd = time_backend(ctx, inputs, outputs, wb, 1, PROFILE_RUNS); }
        catch (...) {}

        backend = (t_winograd < t_rocmlir) ? 1 : 0;
        std::cerr << "[WinogradConv] " << GetName()
                  << " C=" << C_ << " H=" << H_ << " K=" << K_
                  << " rocMLIR=" << t_rocmlir << "ms Winograd=" << t_winograd << "ms"
                  << " → " << (backend == 1 ? "Winograd" : "rocMLIR")
                  << std::endl;
        WinogradSelector::global().set(shape_hash_, backend);
    }

    if (backend == 1) {
        run_winograd(ctx, inputs, outputs, wb);
    } else {
        run_rocmlir(ctx, inputs, outputs, wb);
    }
}

}  // namespace rocm_gpu
}  // namespace ov

#endif  // ENABLE_WINOGRAD
