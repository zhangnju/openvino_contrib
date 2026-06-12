// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fused_convolution_rocmlir.hpp"
#include "rocm_creation_context.hpp"
#include "error.hpp"
#include "ops/silu_tracking.hpp"
#include "transformer/nodes/fused_convolution_slice.hpp"
#include "transformer/nodes/fused_convolution_slice_out.hpp"
#include <mutex>
#include <sstream>
#include "ops/converters.hpp"
#include "ops/convolution_components/convolution_miopen_components.hpp"
#include "kernels/bias_add.hpp"

#include <fmt/format.h>
#include <openvino/core/except.hpp>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

namespace ov {
namespace rocm_gpu {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers to build rocmlir::ConvParams from OpenVINO ConvolutionParams
// (identical to convolution_rocmlir.cpp — shared code could be extracted later)
// ─────────────────────────────────────────────────────────────────────────────

static rocmlir::ConvParams to_rocmlir(
        const Convolution::Details::ConvolutionParams& p,
        const rocm::Device& dev) {
    OPENVINO_ASSERT(p.NumberOfSpatialDims() == 2,
                    "FusedConvolutionRocMLIR: only 2-D convolution supported");
    rocmlir::ConvParams rp;
    rp.N = static_cast<int>(p.input_shape_[0]);
    rp.C = static_cast<int>(p.input_shape_[1]);
    rp.H = static_cast<int>(p.input_shape_[2]);
    rp.W = static_cast<int>(p.input_shape_[3]);
    rp.K = static_cast<int>(p.filter_shape_[0]);
    rp.R = static_cast<int>(p.filter_shape_[2]);
    rp.S = static_cast<int>(p.filter_shape_[3]);
    rp.pad_h      = static_cast<int>(p.padding_before_[0]);
    rp.pad_w      = static_cast<int>(p.padding_before_[1]);
    rp.stride_h   = static_cast<int>(p.strides_[0]);
    rp.stride_w   = static_cast<int>(p.strides_[1]);
    rp.dilation_h = static_cast<int>(p.dilations_[0]);
    rp.dilation_w = static_cast<int>(p.dilations_[1]);
    rp.groups     = static_cast<int>(p.groups_);
    rp.fp16       = (p.element_type_ == ov::element::Type_t::f16);
    auto props    = dev.props();
    std::string arch_name(props.gcnArchName);
    auto colon = arch_name.find(':');
    if (colon != std::string::npos) arch_name = arch_name.substr(0, colon);
    rp.arch   = arch_name;
    rp.num_cu = props.multiProcessorCount;
    return rp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

FusedConvolutionRocMLIR::FusedConvolutionRocMLIR(
        const CreationContext& ctx,
        const ov::Node& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds,
        const Convolution::Details::FusedConvolutionParams& params)
    : OperationMIOPEN(ctx, node, std::move(inputIds), std::move(outputIds))
{
    conv_params_ = to_rocmlir(params.conv_, ctx.device());
    activation_  = params.activation_;
    has_add_     = params.add_shape_.has_value();

    // Check if this is a FusedConvolutionSlice node (input-side slice)
    if (const auto* slice_node = dynamic_cast<const nodes::FusedConvolutionSlice*>(&node)) {
        conv_params_.c_start = slice_node->get_c_start();
        const auto& in_shape = node.get_input_shape(0);
        conv_params_.C_full  = static_cast<int>(in_shape[1]);
    } else {
        const auto& rt_info = node.get_rt_info();
        auto it_cstart = rt_info.find("rocmlir_slice_c_start");
        auto it_cfull  = rt_info.find("rocmlir_slice_c_full");
        if (it_cstart != rt_info.end() && it_cfull != rt_info.end()) {
            conv_params_.c_start = std::stoi(it_cstart->second.as<std::string>());
            conv_params_.C_full  = std::stoi(it_cfull->second.as<std::string>());
        }
    }
    const bool has_input_slice = (conv_params_.C_full > 0 && conv_params_.C_full != conv_params_.C);

    // Check if this is a FusedConvolutionSliceOut node (output-side slice + SiLU + skip-add)
    int slice_out_k_full = 0, slice_out_c_start = 0, slice_out_c_end = 0;
    if (const auto* sout = dynamic_cast<const nodes::FusedConvolutionSliceOut*>(&node)) {
        slice_out_k_full  = static_cast<int>(node.get_input_shape(1)[0]); // K_full from filter
        slice_out_c_start = sout->get_c_out_start();
        slice_out_c_end   = sout->get_c_out_end();
    } else {
        const auto& rt_info = node.get_rt_info();
        auto it_kfull = rt_info.find("rocmlir_slice_out_k_full");
        auto it_cst   = rt_info.find("rocmlir_slice_out_c_start");
        auto it_cen   = rt_info.find("rocmlir_slice_out_c_end");
        if (it_kfull != rt_info.end() && it_cst != rt_info.end() && it_cen != rt_info.end()) {
            slice_out_k_full  = std::stoi(it_kfull->second.as<std::string>());
            slice_out_c_start = std::stoi(it_cst->second.as<std::string>());
            slice_out_c_end   = std::stoi(it_cen->second.as<std::string>());
        }
    }
    const bool has_output_slice = (slice_out_k_full > 0);

    // Detect 5-input FusedConvolution node: conv+bias+silu+skip → silu(out) → add(silu, aux)
    // Created by MarkSwishAddEpiloguePass when FC has single-consumer Swish.
    // Uses 6-arg migraphx kernel: mlir_convolution_broadcast_add_sigmoid_mul_add_sigmoid_mul_add
    const bool has_silu_add_epilogue = (node.get_input_size() == 5);
    has_silu_add_epilogue_ = has_silu_add_epilogue;

    // Detect conv+bias+reshape pattern (MIGraphX mlir_convolution_broadcast_add_reshape).
    // Tagged by MarkConvReshapeEpiloguePass when FC(bias) → Reshape (single consumer).
    // compile a 3-arg migraphx kernel that includes reshape as zero-cost epilogue.
    {
        const auto& rt = node.get_rt_info();
        auto it = rt.find("rocm_conv_reshape_dims");
        if (it != rt.end() && (std::getenv("ROCMLIR_CONV_RESHAPE_FUSION") || std::getenv("ROCMLIR_EPILOGUE_FUSION"))) {
            const std::string dims_str = it->second.as<std::string>();
            // Parse dims string e.g. "1,4,6400"
            std::vector<int> reshape_dims;
            std::istringstream ss(dims_str);
            std::string tok;
            while (std::getline(ss, tok, ','))
                reshape_dims.push_back(std::stoi(tok));

            static std::unordered_map<size_t, rocmlir::KernelEntry> reshape_cache_;
            static std::mutex reshape_mu_;
            size_t rkey = conv_params_.hash();
            for (int d : reshape_dims) rkey ^= std::hash<int>{}(d) + 0x9e3779b9 + (rkey << 6) + (rkey >> 2);
            {
                std::lock_guard<std::mutex> lk(reshape_mu_);
                auto it2 = reshape_cache_.find(rkey);
                if (it2 != reshape_cache_.end()) {
                    kernel_ = &it2->second;
                } else {
                    auto compiled = rocmlir::compile_conv_fused_reshape(conv_params_, reshape_dims);
                    hipModule_t mod; hipFunction_t fn;
                    hipModuleLoadData(&mod, compiled.hsaco.data());
                    hipModuleGetFunction(&fn, mod, compiled.kernel_name.c_str());
                    rocmlir::KernelEntry entry;
                    entry.module = mod; entry.function = fn;
                    entry.info = std::move(compiled); entry.params = conv_params_;
                    auto& stored = reshape_cache_[rkey] = std::move(entry);
                    kernel_ = &stored;
                    std::cerr << "[ConvReshapeKernel] grid=" << kernel_->info.grid_x
                              << " block=" << kernel_->info.block_x << "\n";
                }
            }
            // This kernel outputs reshaped data; no separate SiLU/bias needed
            conv_reshape_epilogue_ = true;
            return;  // kernel_ is set; skip standard selection below
        }
    }

    // Compile (or reuse cached) rocMLIR kernel based on activation type.
    // Priority:
    //   5-input epilogue  → 6-arg Conv+Bias+SiLU+Skip+SiLU+Aux kernel (migraphx dialect)
    //   output-slice      → Conv+Bias+SliceOut+SiLU+Add (5-arg)
    //   input-slice+SWISH → Slice+Conv+Bias+SiLU fused kernel (rocmlir-gen + v3 tuning)
    //   SWISH + skip-add  → 5-arg Conv+Bias+SiLU+SkipAdd fused kernel
    //   SWISH only        → 4-arg Conv+Bias+SiLU fused kernel
    //   no activation     → 4-arg Conv+Bias kernel
    if (has_silu_add_epilogue) {
        // 6-arg migraphx kernel: compile via ROCMLIR_EPILOGUE_FUSION pipeline
        static std::unordered_map<size_t, rocmlir::KernelEntry> silu_add_cache_;
        static std::mutex silu_add_mu_;
        const size_t key = conv_params_.hash() ^ static_cast<size_t>(0xEEFF12345678ULL);
        {
            std::lock_guard<std::mutex> lk(silu_add_mu_);
            auto it = silu_add_cache_.find(key);
            if (it != silu_add_cache_.end()) {
                kernel_ = &it->second;
            } else {
                // with_skip=true, with_silu_add=true → 6-arg kernel
                auto compiled = rocmlir::compile_conv_fused_epilogue(conv_params_, "",
                    /*with_skip=*/true, /*with_silu_add=*/true);
                compiled.bias_fused     = true;
                compiled.silu_fused     = true;
                compiled.skip_add_fused = true;
                hipModule_t mod; hipFunction_t fn;
                hipModuleLoadData(&mod, compiled.hsaco.data());
                hipModuleGetFunction(&fn, mod, compiled.kernel_name.c_str());
                rocmlir::KernelEntry entry;
                entry.module = mod; entry.function = fn;
                entry.info = std::move(compiled); entry.params = conv_params_;
                auto& stored = silu_add_cache_[key] = std::move(entry);
                kernel_ = &stored;
                std::cerr << "[SiLUAddEpilogue] Compiled 6-arg kernel '"
                          << kernel_->info.kernel_name << "' grid="
                          << kernel_->info.grid_x << " block=" << kernel_->info.block_x << "\n";
            }
        }
    } else if (has_output_slice) {
        // Use the slice-out-silu-add cache (need a custom cache entry)
        // For now compile directly and cache with unique key via conv_params modification
        // Store slice-out params in conv_params_ extra fields for the kernel cache
        // We reuse C_full/c_start to encode K_full/c_out_start; the cache key hash includes them
        rocmlir::ConvParams slice_out_params = conv_params_;
        slice_out_params.C_full  = slice_out_k_full;
        slice_out_params.c_start = slice_out_c_start;
        // The kernel will be compiled with output slice, but kernel cache uses this params hash
        static std::unordered_map<size_t, rocmlir::KernelEntry> slice_out_cache_;
        static std::mutex slice_out_mu_;
        const size_t key = slice_out_params.hash()
            ^ static_cast<size_t>(0xCCC7DDDEEE8FULL)
            ^ (static_cast<size_t>(slice_out_c_end) << 8);
        {
            std::lock_guard<std::mutex> lk(slice_out_mu_);
            auto it = slice_out_cache_.find(key);
            if (it != slice_out_cache_.end()) {
                kernel_ = &it->second;
            } else {
                auto compiled = rocmlir::compile_conv_slice_out_silu_add(
                    conv_params_, slice_out_k_full, slice_out_c_start, slice_out_c_end);
                hipModule_t mod; hipFunction_t fn;
                hipModuleLoadData(&mod, compiled.hsaco.data());
                hipModuleGetFunction(&fn, mod, compiled.kernel_name.c_str());
                rocmlir::KernelEntry entry;
                entry.module = mod; entry.function = fn;
                entry.info = std::move(compiled); entry.params = conv_params_;
                auto& stored = slice_out_cache_[key];
                stored = std::move(entry);
                kernel_ = &stored;
            }
        }
    } else if (has_input_slice && activation_ == nodes::ActivationMode::SWISH) {
        kernel_ = &rocmlir::RocMLIRKernelCache::global()
                       .get_or_compile_slice_conv_bias_silu(conv_params_);
    } else if (has_input_slice) {
        // Input slice but no SiLU - plain slice+bias kernel
        kernel_ = &rocmlir::RocMLIRKernelCache::global()
                       .get_or_compile_slice_conv_bias_silu(conv_params_);  // will use NO_ACTIVATION
    } else if (activation_ == nodes::ActivationMode::SWISH && has_add_) {
        // migraphx dialect: conv+bias+silu+skip_add in one kernel (5 args).
        // Default ON — uses MIGraphX-compatible MLIR pipeline for better GPU pipeline efficiency.
        // Set ROCMLIR_EPILOGUE_FUSION=0 to fall back to rock dialect (v3: perf_config path).
        static const bool use_epilogue_fusion = []{
            const char* e = std::getenv("ROCMLIR_EPILOGUE_FUSION");
            return !(e && std::string(e) == "0");
        }();
        if (use_epilogue_fusion) {
            // migraphx dialect: conv+bias+silu+skip_add in one kernel (5 args)
            auto compiled = rocmlir::compile_conv_fused_epilogue(conv_params_, "", /*with_skip=*/true);
            auto& stored = rocmlir::RocMLIRKernelCache::global()
                               .insert_fused_epilogue_silu_add(conv_params_, std::move(compiled));
            kernel_ = &stored;
        } else {
        kernel_ = &rocmlir::RocMLIRKernelCache::global()
                       .get_or_compile_fused_bias_silu_add(conv_params_);
        }
    } else if (activation_ == nodes::ActivationMode::SWISH) {
        // migraphx dialect: conv+bias+silu in one kernel (4 args). Default ON.
        // Set ROCMLIR_EPILOGUE_FUSION=0 to use rock dialect (v3: perf_config path).
        static const bool use_epilogue_fusion2 = []{
            const char* e = std::getenv("ROCMLIR_EPILOGUE_FUSION");
            return !(e && std::string(e) == "0");
        }();
        if (use_epilogue_fusion2) {
            // migraphx dialect: conv+bias+silu in one kernel (4 args)
            auto compiled = rocmlir::compile_conv_fused_epilogue(conv_params_, "", /*with_skip=*/false);
            auto& stored = rocmlir::RocMLIRKernelCache::global()
                               .insert_fused_epilogue_silu(conv_params_, std::move(compiled));
            kernel_ = &stored;
        } else {
        kernel_ = &rocmlir::RocMLIRKernelCache::global()
                       .get_or_compile_fused_bias_act(conv_params_, rocmlir::Activation::Sigmoid);
        }
    } else if (activation_ == nodes::ActivationMode::NO_ACTIVATION && has_add_) {
        // Conv+Bias+SkipAdd (no SiLU): matches MIGraphX mlir_convolution_broadcast_add_add.
        // Use migraphx kernel when ROCMLIR_EPILOGUE_FUSION=1 or ROCMLIR_CONV_SKIP_FUSION=1.
        static const bool fuse_skip = []{
            return std::getenv("ROCMLIR_EPILOGUE_FUSION") ||
                   std::getenv("ROCMLIR_CONV_SKIP_FUSION");
        }();
        if (fuse_skip) {
            static std::unordered_map<size_t, rocmlir::KernelEntry> skip_cache_;
            static std::mutex skip_mu_;
            const size_t key = conv_params_.hash() ^ static_cast<size_t>(0xABCD1234ABCDULL);
            {
                std::lock_guard<std::mutex> lk(skip_mu_);
                auto it = skip_cache_.find(key);
                if (it != skip_cache_.end()) {
                    kernel_ = &it->second;
                } else {
                    auto compiled = rocmlir::compile_conv_fused_skip(conv_params_);
                    hipModule_t mod; hipFunction_t fn;
                    hipModuleLoadData(&mod, compiled.hsaco.data());
                    hipModuleGetFunction(&fn, mod, compiled.kernel_name.c_str());
                    rocmlir::KernelEntry entry;
                    entry.module = mod; entry.function = fn;
                    entry.info = std::move(compiled); entry.params = conv_params_;
                    auto& stored = skip_cache_[key] = std::move(entry);
                    kernel_ = &stored;
                }
            }
        } else {
            kernel_ = &rocmlir::RocMLIRKernelCache::global()
                           .get_or_compile_fused_bias(conv_params_);
        }
    } else {
        kernel_ = &rocmlir::RocMLIRKernelCache::global()
                       .get_or_compile_fused_bias(conv_params_);
    }

    // Pre-initialize standalone Swish kernel for fallback (when SiLU not fused in kernel).
    // Not needed when skip_add_fused=true (5-arg kernel handles SiLU internally).
    if (activation_ == nodes::ActivationMode::SWISH && !has_add_) {
        const size_t n_out = (size_t)conv_params_.N * conv_params_.K *
                             conv_params_.out_h() * conv_params_.out_w();
        const auto el_type = (params.conv_.element_type_ == ov::element::Type_t::f16) ?
                             kernel::Type_t::f16 : kernel::Type_t::f32;
        swish_kernel_.emplace(el_type, ctx.device().props().maxThreadsPerBlock, n_out, 1.0);
    }

    // Build bias descriptor (shape: 1×K×1×1)
    bias_desc_ = *Convolution::Details::MakeFusedAddDescriptor(
                      params.bias_shape_, params.conv_.element_type_);

    // Build output descriptor (shape: N×K×OH×OW)
    output_desc_ = *Convolution::Details::MakeFusedAddDescriptor(
                       params.conv_.output_shape_, params.conv_.element_type_);

    // Optional: skip-connection add descriptor
    if (has_add_) {
        add_desc_ = *Convolution::Details::MakeFusedAddDescriptor(
                         *params.add_shape_, params.conv_.element_type_);
    }

    // Optional: activation descriptor (only for MIOpen-supported activations)
    // SWISH is handled separately via SwishOpImpl kernel (not MIOpen), so skip it here.
    if (activation_ != nodes::ActivationMode::NO_ACTIVATION &&
        activation_ != nodes::ActivationMode::SWISH) {
        activation_desc_ = rocm::DnnActivationDescriptor{};
        activation_desc_->set(convertActivationMode(activation_), 0.0, 0.0, 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute
// ─────────────────────────────────────────────────────────────────────────────

void FusedConvolutionRocMLIR::Execute(const InferenceRequestContext& ctx,
                                       Inputs inputs, Outputs outputs,
                                       const Workbuffers& wbs) const {
    using Idx = Convolution::Details::FusedConvolutionIndices;
    OPENVINO_ASSERT(outputs.size() == 1, GetName(), ": expected 1 output");

    const auto& dnn = ctx.getThreadContext().dnnHandle();

    void* d_input  = const_cast<void*>(inputs[Idx::input].get());
    void* d_filter = const_cast<void*>(inputs[Idx::filter].get());
    void* d_bias   = const_cast<void*>(inputs[Idx::bias].get());
    void* d_output = outputs[Idx::output].get();
    void* d_ws     = wbs.mutable_buffers.empty() ? nullptr : wbs.mutable_buffers[0].get();

    const auto& info = kernel_->info;

    // For input-slice kernels that do NOT contain a Slice op in the MLIR
    // (fallback v3 kernels), apply the channel offset manually.
    // The kernel was compiled for [N, C_slice, H, W] input, but d_input points
    // to the full [N, C_full, H, W] tensor. We compute the byte offset for
    // c_start channels and pass the adjusted pointer.
    // Condition: has_input_slice AND kernel uses standard (non-slice) MLIR.
    if (conv_params_.C_full > 0 && conv_params_.c_start > 0 &&
        info.kernel_name.find("slice_conv") == std::string::npos &&
        info.kernel_name.find("mlir_slice") == std::string::npos) {
        const size_t elem_size = conv_params_.fp16 ? 2 : 4;
        const size_t channel_stride = (size_t)conv_params_.H * conv_params_.W;
        const size_t byte_offset = (size_t)conv_params_.c_start * channel_stride * elem_size;
        d_input = static_cast<char*>(d_input) + byte_offset;
    }

    // ── Conv+Bias+Reshape kernel (3-arg migraphx): Q/K/V attention projection ──
    // mlir_convolution_broadcast_add_reshape: writes reshaped output directly from conv.
    // No separate SiLU or bias step; kernel writes to reshaped output layout.
    if (conv_reshape_epilogue_) {
        // 3-arg: (input, filter, bias) → reshaped_output
        // MIGraphX convention: input, filter, bias, output (legacy ordering)
        void* args3[] = { &d_input, &d_filter, &d_bias, &d_output };
        hipError_t err = hipModuleLaunchKernel(
            kernel_->function,
            info.grid_x, 1, 1, info.block_x, 1, 1,
            0, ctx.getThreadContext().stream().get(),
            args3, nullptr);
        if (err != hipSuccess)
            throw_ov_exception(fmt::format("FusedConvRocMLIR conv+bias+reshape launch failed: {}",
                                           hipGetErrorString(err)));
        return;
    }

    // ── 6-arg epilogue kernel: Conv+Bias+SiLU+Skip → SiLU(out) → Add(silu, aux) ──
    // Used for 5-input FusedConvolution created by MarkSwishAddEpiloguePass.
    // Arg order (migraphx convention): input, filter, bias, skip, aux_silu, output
    if (has_silu_add_epilogue_ && inputs.size() > Idx::aux_silu) {
        void* d_skip = const_cast<void*>(inputs[Idx::add].get());
        void* d_aux  = const_cast<void*>(inputs[Idx::aux_silu].get());
        // migraphx 6-arg: (input, filter, bias, skip, aux, output)
        void* args6[] = { &d_input, &d_filter, &d_bias, &d_skip, &d_aux, &d_output };
        hipError_t err = hipModuleLaunchKernel(
            kernel_->function,
            info.grid_x, 1, 1, info.block_x, 1, 1,
            0, ctx.getThreadContext().stream().get(),
            args6, nullptr);
        if (err != hipSuccess)
            throw_ov_exception(fmt::format("FusedConvRocMLIR 6arg-silu-add-silu-add launch failed: {}",
                                           hipGetErrorString(err)));
        // Output = final result after outer silu+add; no further SiLU needed
        return;
    }

    // For FusedConvolutionSliceOut: has_add_=true (set by factory via add_shape_) AND input[3]=skip
    if (info.bias_fused && info.skip_add_fused && inputs.size() > Idx::add) {
        // ── 5-arg fused kernel: Conv+Bias+(Slice)+SiLU+SkipAdd ───────────────
        // Arg order: filter, input, bias, skip, output (rocMLIR patched kernel convention)
        // Legacy format (mlir_convolution_broadcast_*): input, filter, bias, skip, output
        void* d_add = const_cast<void*>(inputs[Idx::add].get());
        const bool legacy_arg_order = info.kernel_name.find("mlir_convolution") != std::string::npos;
        void* args5_legacy[]  = { &d_input, &d_filter, &d_bias, &d_add, &d_output };
        void* args5_patched[] = { &d_filter, &d_input,  &d_bias, &d_add, &d_output };
        void** args5 = legacy_arg_order
            ? reinterpret_cast<void**>(args5_legacy)
            : reinterpret_cast<void**>(args5_patched);
        hipError_t err = hipModuleLaunchKernel(
            kernel_->function,
            info.grid_x, 1, 1, info.block_x, 1, 1,
            0, ctx.getThreadContext().stream().get(),
            args5, nullptr);
        if (err != hipSuccess)
            throw_ov_exception(fmt::format("FusedConvRocMLIR 5arg-silu-add launch failed: {}",
                                           hipGetErrorString(err)));
        // SiLU is fused; mark so downstream SwishOp becomes no-op
        mark_silu_applied(d_output);

    } else if (info.bias_fused) {
        // ── 4-arg fused kernel: Conv+Bias (+SiLU) ────────────────────────────
        // Arg order: filter, input, bias, output (rocMLIR patched kernel convention)
        // Legacy format (mlir_convolution_broadcast_*): input, filter, bias, output
        void* args_legacy[]  = { &d_input,  &d_filter, &d_bias, &d_output };
        void* args_patched[] = { &d_filter, &d_input,  &d_bias, &d_output };
        const bool legacy_arg_order2 =
            info.kernel_name.find("mlir_convolution") != std::string::npos;
        void** args = legacy_arg_order2
            ? reinterpret_cast<void**>(args_legacy)
            : reinterpret_cast<void**>(args_patched);
        hipError_t err = hipModuleLaunchKernel(
            kernel_->function,
            info.grid_x, 1, 1, info.block_x, 1, 1,
            0, ctx.getThreadContext().stream().get(),
            args, nullptr);
        if (err != hipSuccess)
            throw_ov_exception(fmt::format("FusedConvRocMLIR {}-fused launch failed: {}",
                                           info.kernel_name.substr(0, 20),
                                           hipGetErrorString(err)));
        // bias_add already done by kernel; SiLU too if silu_fused
        // Mark this output buffer so downstream SwishOp can skip its execution
        if (info.silu_fused) {
            mark_silu_applied(d_output);
        }
    } else {
        // ── Plain conv (3 args), then separate bias_add ────────────────────────
        {
            void* args[] = { &d_filter, &d_input, &d_output, &d_ws };
            (void)d_ws;
            hipError_t err = hipModuleLaunchKernel(
                kernel_->function,
                info.grid_x, 1, 1, info.block_x, 1, 1,
                0, ctx.getThreadContext().stream().get(),
                args, nullptr);
            if (err != hipSuccess)
                throw_ov_exception(fmt::format("FusedConvolutionRocMLIR conv launch failed: {}",
                                               hipGetErrorString(err)));
        }
        kernel::launch_bias_add(d_output, d_bias,
                                conv_params_.N, conv_params_.K,
                                conv_params_.out_h(), conv_params_.out_w(),
                                conv_params_.fp16,
                                ctx.getThreadContext().stream().get());
    }

    // ── Step 3: optional skip-connection add (only when NOT fused into kernel) ─
    if (has_add_ && !info.skip_add_fused && inputs.size() > Idx::add) {
        void* d_add = const_cast<void*>(inputs[Idx::add].get());
        const int total = conv_params_.N * conv_params_.K *
                          conv_params_.out_h() * conv_params_.out_w();
        kernel::launch_bias_add(d_output, d_add,
                                1, total, 1, 1,
                                conv_params_.fp16,
                                ctx.getThreadContext().stream().get());
    }

    // ── Step 4: activation ────────────────────────────────────────────────────
    // Swish is applied by the fused kernel (if silu_fused) or separately here.
    if (activation_ == nodes::ActivationMode::SWISH &&
        swish_kernel_.has_value() && !info.silu_fused) {
        (*swish_kernel_)(ctx.getThreadContext().stream().get(), d_output, d_output);
    }
}

WorkbufferRequest FusedConvolutionRocMLIR::GetWorkBufferRequest() const {
    const size_t ws = kernel_->info.workspace_bytes;
    if (ws > 0) return {{}, {ws}};
    return {{}, {}};
}

} // namespace rocm_gpu
} // namespace ov
