// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MIGraphX fuse_pointwise_reduce integration for OV ROCm Plugin.
//
// Architecture:
//   1. dlopen libmigraphx.so + libmigraphx_gpu.so at runtime (avoids static init conflicts)
//   2. Use MIGraphX C++ API via dlsym-resolved function pointers
//   3. Build LayerNorm subgraph, apply fuse_pointwise_reduce, compile to GPU
//   4. Extract HSACO from gpu::code_object_op via reflected value
//
// This approach avoids:
//   - Static initialization order conflicts with OV/HIP libraries
//   - ABI issues from different compiler flags
//   - Dependency hell at link time

#include "rocm_fused_kernel.hpp"
#include <openvino/core/except.hpp>
#include <dlfcn.h>
#include <cstdio>
#include <mutex>
#include <functional>
#include <sstream>

// ── MIGraphX headers (for type definitions only, not linking) ─────────────────
// We use these at compile time for type definitions, but resolve symbols at runtime.
#define __HIP_PLATFORM_AMD__ 1
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/fuse_pointwise_reduce.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/sub.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/div.hpp>
#include <migraphx/op/erf.hpp>
#include <migraphx/op/sqrt.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/softmax.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/multibroadcast.hpp>
#include <migraphx/op/rsqrt.hpp>

namespace ov {
namespace rocm_gpu {
namespace fused_kernel {

// ── MIGraphX library loader ───────────────────────────────────────────────────
namespace {

struct MxLibs {
    void* mx  = nullptr;
    void* gpu = nullptr;
    bool loaded = false;

    static MxLibs& instance() {
        static MxLibs libs;
        return libs;
    }

    bool load() {
        if (loaded) return true;
        // Load with RTLD_GLOBAL so symbol resolution works across libraries
        mx = dlopen("/opt/rocm/lib/migraphx/lib/libmigraphx.so",
                    RTLD_NOW | RTLD_GLOBAL);
        if (!mx) {
            fprintf(stderr, "[FusedKernel] dlopen libmigraphx.so failed: %s\n", dlerror());
            return false;
        }
        gpu = dlopen("/opt/rocm/lib/migraphx/lib/libmigraphx_gpu.so",
                     RTLD_NOW | RTLD_GLOBAL);
        if (!gpu) {
            fprintf(stderr, "[FusedKernel] dlopen libmigraphx_gpu.so failed: %s\n", dlerror());
            dlclose(mx); mx = nullptr;
            return false;
        }
        loaded = true;
        fprintf(stderr, "[FusedKernel] Loaded MIGraphX libraries\n");
        return true;
    }
};

}  // namespace

// ── LayerNorm program builder ─────────────────────────────────────────────────
static migraphx::program build_layernorm_program(
        int64_t seq, int64_t hidden, float eps) {

    migraphx::program prog;
    auto* mm = prog.get_main_module();

    using migraphx::shape;
    const auto half_t    = shape::half_type;
    const shape x_shape  {half_t, {(size_t)seq, (size_t)hidden}};
    const shape g_shape  {half_t, {(size_t)hidden}};

    auto x     = mm->add_parameter("x",     x_shape);
    auto gamma = mm->add_parameter("gamma", g_shape);
    auto beta  = mm->add_parameter("beta",  g_shape);

    // mean = reduce_mean(x, axes=[1])  → [seq, 1]
    auto mean = mm->add_instruction(migraphx::op::reduce_mean{{1}}, x);

    // diff = x - broadcast(mean)
    auto mean_bc = mm->add_instruction(
        migraphx::op::multibroadcast{{(size_t)seq, (size_t)hidden}}, mean);
    auto diff = mm->add_instruction(migraphx::op::sub{}, x, mean_bc);

    // var = reduce_mean(diff^2, axes=[1])
    auto diff_sq = mm->add_instruction(migraphx::op::mul{}, diff, diff);
    auto var     = mm->add_instruction(migraphx::op::reduce_mean{{1}}, diff_sq);

    // eps
    migraphx::half eps_h(eps);
    auto eps_lit = mm->add_literal(
        migraphx::literal{shape{half_t, {1}}, {eps_h}});
    auto eps_bc = mm->add_instruction(
        migraphx::op::multibroadcast{{(size_t)seq, 1}}, eps_lit);
    auto var_eps = mm->add_instruction(migraphx::op::add{}, var, eps_bc);

    // std_inv = rsqrt(var + eps)
    auto std_inv = mm->add_instruction(migraphx::op::rsqrt{}, var_eps);
    auto std_bc  = mm->add_instruction(
        migraphx::op::multibroadcast{{(size_t)seq, (size_t)hidden}}, std_inv);

    // norm = diff * std_inv
    auto norm = mm->add_instruction(migraphx::op::mul{}, diff, std_bc);

    // gamma and beta broadcast (axis=1 = column broadcast, stride_0 = 0)
    auto gamma_bc = mm->add_instruction(
        migraphx::op::broadcast{1, {(size_t)seq, (size_t)hidden}}, gamma);
    auto norm_g   = mm->add_instruction(migraphx::op::mul{}, norm, gamma_bc);

    auto beta_bc  = mm->add_instruction(
        migraphx::op::broadcast{1, {(size_t)seq, (size_t)hidden}}, beta);
    auto result   = mm->add_instruction(migraphx::op::add{}, norm_g, beta_bc);

    mm->add_return({result});
    return prog;
}

// ── Main compilation entry point ──────────────────────────────────────────────
std::optional<KernelInfo> compile_layernorm_residual(
        int64_t seq_len, int64_t hidden, bool /*with_residual*/,
        const std::string& /*arch*/, float epsilon) {

    // Ensure MIGraphX libraries are loaded
    if (!MxLibs::instance().load()) {
        fprintf(stderr, "[FusedKernel] MIGraphX not available\n");
        return std::nullopt;
    }

    try {
        // Build LayerNorm program
        auto prog = build_layernorm_program(seq_len, hidden, epsilon);
        auto* mm  = prog.get_main_module();

        size_t n_before = std::distance(mm->begin(), mm->end());
        fprintf(stderr, "[FusedKernel] Instructions before fusion: %zu\n", n_before);

        // Compile directly to GPU target.
        // GPU target's get_passes() internally applies:
        //   normalize_ops, fuse_reduce, fuse_pointwise, fuse_pointwise_reduce,
        //   gpu::lowering, split_reduce, dead_code_elimination, ...
        // This is the correct way to apply fuse_pointwise_reduce with proper GPU context.
        fprintf(stderr, "[FusedKernel] Compiling to GPU (applies fuse_pointwise_reduce)...\n");
        migraphx::gpu::target gpu_target;
        migraphx::compile_options opts;
        opts.offload_copy = false;
        prog.compile(gpu_target, opts);
        fprintf(stderr, "[FusedKernel] Compile done\n");

        fprintf(stderr, "[FusedKernel] Compiled successfully\n");

        // Extract HSACO from gpu::code_object instruction
        KernelInfo info;
        for (auto ins = mm->begin(); ins != mm->end(); ++ins) {
            if (ins->name() != "gpu::code_object") continue;

            const auto& op = ins->get_operator();
            const auto* co = migraphx::any_cast<migraphx::gpu::code_object_op>(&op);
            if (!co || co->code_object.empty()) continue;

            info.hsaco.assign(co->code_object.begin(), co->code_object.end());
            info.symbol_name = co->symbol_name;
            // grid_x = total_threads / threads_per_block
            info.global_x = (co->local > 0)
                ? static_cast<unsigned>(co->global / co->local)
                : static_cast<unsigned>(co->global);
            info.local_x   = static_cast<unsigned>(co->local);
            info.output_arg = static_cast<int>(co->output_arg);
            break;
        }

        if (info.hsaco.size() < 4 || info.symbol_name.empty()) {
            fprintf(stderr, "[FusedKernel] HSACO extraction failed\n");
            return std::nullopt;
        }

        fprintf(stderr, "[FusedKernel] Kernel=%s grid=%u block=%u HSACO=%zu B\n",
                info.symbol_name.c_str(), info.global_x, info.local_x,
                info.hsaco.size());
        return info;

    } catch (const std::exception& e) {
        fprintf(stderr, "[FusedKernel] Error: %s\n", e.what());
        return std::nullopt;
    }
}

// ── FC+GELU program builder ───────────────────────────────────────────────────
// Builds: x[seq,in] × W[in,out] + bias[out] → GELU → output[seq,out]
// MIGraphX's fuse_pointwise_reduce fuses MatMul+bias+GELU into a single kernel.
// GELU (erf form): y = x * 0.5 * (1 + erf(x / sqrt(2)))
static migraphx::program build_fc_gelu_program(
        int64_t seq, int64_t in_dim, int64_t out_dim) {
    migraphx::program prog;
    auto* mm = prog.get_main_module();

    using migraphx::shape;
    const auto half_t = shape::half_type;

    auto x    = mm->add_parameter("x",    shape{half_t, {(size_t)seq, (size_t)in_dim}});
    auto W    = mm->add_parameter("W",    shape{half_t, {(size_t)in_dim, (size_t)out_dim}});
    auto bias = mm->add_parameter("bias", shape{half_t, {(size_t)out_dim}});

    // FC = x × W + bias
    auto fc = mm->add_instruction(migraphx::op::dot{}, x, W);
    auto bias_bc = mm->add_instruction(
        migraphx::op::broadcast{1, {(size_t)seq, (size_t)out_dim}}, bias);
    auto fc_bias = mm->add_instruction(migraphx::op::add{}, fc, bias_bc);

    // GELU (erf approximation): y = x * 0.5 * (1 + erf(x / sqrt(2)))
    // Build as elementwise ops so fuse_pointwise_reduce can absorb them into the GEMM
    migraphx::half half_05{0.5f};
    migraphx::half sqrt2_recip{static_cast<float>(1.0 / std::sqrt(2.0))};
    migraphx::half one{1.0f};

    auto lit_05  = mm->add_literal(migraphx::literal{shape{half_t, {1}}, {half_05}});
    auto lit_r2  = mm->add_literal(migraphx::literal{shape{half_t, {1}}, {sqrt2_recip}});
    auto lit_one = mm->add_literal(migraphx::literal{shape{half_t, {1}}, {one}});

    auto bc_shape = std::vector<size_t>{(size_t)seq, (size_t)out_dim};
    auto bc_05  = mm->add_instruction(migraphx::op::multibroadcast{bc_shape}, lit_05);
    auto bc_r2  = mm->add_instruction(migraphx::op::multibroadcast{bc_shape}, lit_r2);
    auto bc_one = mm->add_instruction(migraphx::op::multibroadcast{bc_shape}, lit_one);

    // x * (1/sqrt(2))
    auto x_scaled = mm->add_instruction(migraphx::op::mul{}, fc_bias, bc_r2);
    // erf(x/sqrt(2))
    auto erf_val = mm->add_instruction(migraphx::op::erf{}, x_scaled);
    // 1 + erf(...)
    auto one_plus = mm->add_instruction(migraphx::op::add{}, bc_one, erf_val);
    // 0.5 * (1 + erf(...))
    auto half_coeff = mm->add_instruction(migraphx::op::mul{}, bc_05, one_plus);
    // x * 0.5 * (1 + erf(x/sqrt(2)))
    auto gelu = mm->add_instruction(migraphx::op::mul{}, fc_bias, half_coeff);

    mm->add_return({gelu});
    return prog;
}

// Compile a FC+GELU fused kernel.
// Returns: KernelInfo with HSACO and metadata, or nullopt on failure.
std::optional<KernelInfo> compile_fc_gelu(
        int64_t seq, int64_t in_dim, int64_t out_dim) {
    if (!MxLibs::instance().load()) {
        fprintf(stderr, "[FusedKernel] MIGraphX not available for FC+GELU\n");
        return std::nullopt;
    }
    try {
        auto prog = build_fc_gelu_program(seq, in_dim, out_dim);
        auto* mm  = prog.get_main_module();
        fprintf(stderr, "[FusedKernel] FC+GELU: %zu instructions before compile\n",
                (size_t)std::distance(mm->begin(), mm->end()));

        migraphx::gpu::target gpu_target;
        migraphx::compile_options opts;
        opts.offload_copy = false;
        prog.compile(gpu_target, opts);

        fprintf(stderr, "[FusedKernel] FC+GELU: %zu instructions after compile\n",
                (size_t)std::distance(mm->begin(), mm->end()));

        KernelInfo info;
        for (auto ins = mm->begin(); ins != mm->end(); ++ins) {
            if (ins->name() != "gpu::code_object") continue;
            const auto* co = migraphx::any_cast<migraphx::gpu::code_object_op>(
                &ins->get_operator());
            if (!co || co->code_object.empty()) continue;
            info.hsaco.assign(co->code_object.begin(), co->code_object.end());
            info.symbol_name = co->symbol_name;
            info.global_x = (co->local > 0)
                ? static_cast<unsigned>(co->global / co->local)
                : static_cast<unsigned>(co->global);
            info.local_x   = static_cast<unsigned>(co->local);
            info.output_arg = static_cast<int>(co->output_arg);
            break;
        }
        if (info.hsaco.size() < 4 || info.symbol_name.empty()) {
            fprintf(stderr, "[FusedKernel] FC+GELU: HSACO extraction failed\n");
            return std::nullopt;
        }
        fprintf(stderr, "[FusedKernel] FC+GELU kernel: %s grid=%u block=%u HSACO=%zu B\n",
                info.symbol_name.c_str(), info.global_x, info.local_x, info.hsaco.size());
        return info;
    } catch (const std::exception& e) {
        fprintf(stderr, "[FusedKernel] FC+GELU error: %s\n", e.what());
        return std::nullopt;
    }
}

// ── Masked Softmax program builder ───────────────────────────────────────────
// Builds: scores[batch,heads,sq,sk] × scale + mask[1,1,sq,sk] → Softmax → [batch,heads,sq,sk]
// MIGraphX's fuse_pointwise_reduce fuses scale+mask+softmax into a single kernel.
static migraphx::program build_masked_softmax_program(
        int64_t heads, int64_t sq, int64_t sk) {
    migraphx::program prog;
    auto* mm = prog.get_main_module();

    using migraphx::shape;
    const auto half_t = shape::half_type;

    // Input shapes
    const shape scores_shape{half_t, {1, (size_t)heads, (size_t)sq, (size_t)sk}};
    // Mask is [sq, sk] (2D, already squeezed in BERT OV graph)
    const shape mask_shape{half_t, {(size_t)sq, (size_t)sk}};

    auto scores = mm->add_parameter("scores", scores_shape);
    auto mask   = mm->add_parameter("mask",   mask_shape);

    // Broadcast 2D mask [sq,sk] to 4D [1,heads,sq,sk]
    auto mask_bc = mm->add_instruction(
        migraphx::op::multibroadcast{{1, (size_t)heads, (size_t)sq, (size_t)sk}}, mask);

    // scores + mask
    auto masked = mm->add_instruction(migraphx::op::add{}, scores, mask_bc);

    // Softmax over last axis (sk)
    auto soft = mm->add_instruction(migraphx::op::softmax{3}, masked);

    mm->add_return({soft});
    return prog;
}

// Compile a masked softmax fused kernel.
std::optional<KernelInfo> compile_masked_softmax(
        int64_t heads, int64_t sq, int64_t sk) {
    if (!MxLibs::instance().load()) return std::nullopt;
    try {
        auto prog = build_masked_softmax_program(heads, sq, sk);
        auto* mm  = prog.get_main_module();
        fprintf(stderr, "[FusedKernel] MaskedSoftmax: %zu instructions before compile\n",
                (size_t)std::distance(mm->begin(), mm->end()));

        migraphx::gpu::target gpu_target;
        migraphx::compile_options opts;
        opts.offload_copy = false;
        prog.compile(gpu_target, opts);

        KernelInfo info;
        for (auto ins = mm->begin(); ins != mm->end(); ++ins) {
            if (ins->name() != "gpu::code_object") continue;
            const auto* co = migraphx::any_cast<migraphx::gpu::code_object_op>(
                &ins->get_operator());
            if (!co || co->code_object.empty()) continue;
            info.hsaco.assign(co->code_object.begin(), co->code_object.end());
            info.symbol_name = co->symbol_name;
            info.global_x = (co->local > 0)
                ? static_cast<unsigned>(co->global / co->local)
                : static_cast<unsigned>(co->global);
            info.local_x   = static_cast<unsigned>(co->local);
            info.output_arg = static_cast<int>(co->output_arg);
            break;
        }
        if (info.hsaco.size() < 4 || info.symbol_name.empty()) {
            fprintf(stderr, "[FusedKernel] MaskedSoftmax: HSACO extraction failed\n");
            return std::nullopt;
        }
        fprintf(stderr, "[FusedKernel] MaskedSoftmax kernel: %s grid=%u block=%u HSACO=%zu B\n",
                info.symbol_name.c_str(), info.global_x, info.local_x, info.hsaco.size());
        return info;
    } catch (const std::exception& e) {
        fprintf(stderr, "[FusedKernel] MaskedSoftmax error: %s\n", e.what());
        return std::nullopt;
    }
}

// ── Cache ─────────────────────────────────────────────────────────────────────
KernelCache& KernelCache::instance() {
    static KernelCache c;
    return c;
}

std::optional<KernelInfo> KernelCache::get(const std::string& key) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_.find(key);
    if (it == cache_.end()) return std::nullopt;
    return it->second;
}

void KernelCache::put(const std::string& key, KernelInfo info) {
    std::lock_guard<std::mutex> lk(mu_);
    cache_[key] = std::move(info);
}

std::string KernelCache::make_key(const std::string& arch, int64_t seq,
                                   int64_t hidden, bool residual) {
    return arch + "_" + std::to_string(seq) + "_" + std::to_string(hidden) +
           (residual ? "_r" : "");
}

// ── HIP kernel loader ─────────────────────────────────────────────────────────
HipKernel::~HipKernel() {
    if (module) hipModuleUnload(module);
}

HipKernel load_kernel(const KernelInfo& info) {
    HipKernel hk;
    hk.symbol  = info.symbol_name;
    hk.grid_x  = info.global_x;
    hk.block_x = info.local_x;

    auto err = hipModuleLoadData(&hk.module, info.hsaco.data());
    OPENVINO_ASSERT(err == hipSuccess,
        "[FusedKernel] hipModuleLoadData: ", hipGetErrorString(err));

    err = hipModuleGetFunction(&hk.func, hk.module, info.symbol_name.c_str());
    OPENVINO_ASSERT(err == hipSuccess,
        "[FusedKernel] hipModuleGetFunction('", info.symbol_name, "'): ",
        hipGetErrorString(err));

    return hk;
}

}  // namespace fused_kernel
}  // namespace rocm_gpu
}  // namespace ov
