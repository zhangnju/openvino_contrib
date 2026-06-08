// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rocmlir_kernel_cache.hpp"
#include <openvino/core/except.hpp>

namespace ov {
namespace rocm_gpu {
namespace rocmlir {

// ─────────────────────────────────────────────────────────────────────────────
// Global singleton
// ─────────────────────────────────────────────────────────────────────────────

RocMLIRKernelCache& RocMLIRKernelCache::global() {
    static RocMLIRKernelCache instance;
    return instance;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal: load compiled HSACO into HIP module
// ─────────────────────────────────────────────────────────────────────────────

KernelEntry RocMLIRKernelCache::load_kernel(CompiledConv&& compiled) {
    KernelEntry entry;
    entry.info = std::move(compiled);

    hipError_t err = hipModuleLoadData(&entry.module, entry.info.hsaco.data());
    if (err != hipSuccess)
        OPENVINO_THROW("rocMLIR: hipModuleLoadData failed: ", hipGetErrorString(err));

    err = hipModuleGetFunction(&entry.function,
                               entry.module,
                               entry.info.kernel_name.c_str());
    if (err != hipSuccess) {
        hipModuleUnload(entry.module);
        OPENVINO_THROW("rocMLIR: hipModuleGetFunction('", entry.info.kernel_name,
                       "') failed: ", hipGetErrorString(err));
    }
    return entry;
}

// ─────────────────────────────────────────────────────────────────────────────
// Plain conv
// ─────────────────────────────────────────────────────────────────────────────

const KernelEntry& RocMLIRKernelCache::get_or_compile(const ConvParams& p,
                                                       const std::string& driver) {
    const size_t key = p.hash();
    std::lock_guard<std::mutex> lk(mu_);
    auto it = cache_.find(key);
    // Return cached entry only if valid AND matches params (guards against hash collision)
    if (it != cache_.end() && it->second && it->second->function != nullptr
            && it->second->params == p)
        return *it->second;

    // Compile. May throw — unique_ptr prevents leaving a broken entry in the map.
    auto entry = std::make_unique<KernelEntry>(load_kernel(compile_conv(p, driver)));
    entry->params = p;
    KernelEntry* ptr = entry.get();
    cache_[key] = std::move(entry);  // pointer to heap object stays stable after rehash
    return *ptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused conv + bias
// ─────────────────────────────────────────────────────────────────────────────

const KernelEntry& RocMLIRKernelCache::get_or_compile_fused_bias(
        const ConvParams& p, const std::string& driver) {
    const size_t key = p.hash() ^ static_cast<size_t>(0xB1a51ULL);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = fused_bias_cache_.find(key);
    if (it != fused_bias_cache_.end() && it->second && it->second->function != nullptr
            && it->second->params == p)
        return *it->second;

    auto entry = std::make_unique<KernelEntry>(load_kernel(compile_fused_conv_bias(p, driver)));
    entry->params = p;
    KernelEntry* ptr = entry.get();
    fused_bias_cache_[key] = std::move(entry);
    return *ptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused conv + bias + activation
// ─────────────────────────────────────────────────────────────────────────────

const KernelEntry& RocMLIRKernelCache::get_or_compile_fused_bias_act(
        const ConvParams& p, Activation act, const std::string& driver) {
    auto* target_cache = (act == Activation::ReLU)
                         ? &fused_bias_relu_cache_
                         : &fused_bias_sigmoid_cache_;
    const size_t key = p.hash();
    std::lock_guard<std::mutex> lk(mu_);
    auto it = target_cache->find(key);
    if (it != target_cache->end() && it->second && it->second->function != nullptr
            && it->second->params == p)
        return *it->second;

    auto entry = std::make_unique<KernelEntry>(load_kernel(compile_fused_conv_bias_act(p, act, driver)));
    entry->params = p;
    KernelEntry* ptr = entry.get();
    (*target_cache)[key] = std::move(entry);
    return *ptr;
}

const KernelEntry& RocMLIRKernelCache::get_or_compile_fused_bias_silu_add(
        const ConvParams& p, const std::string& driver) {
    const size_t key = p.hash() ^ static_cast<size_t>(0xA7D3C2B1E0F5ULL);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = fused_bias_silu_add_cache_.find(key);
    if (it != fused_bias_silu_add_cache_.end() && it->second && it->second->function != nullptr
            && it->second->params == p)
        return *it->second;

    auto entry = std::make_unique<KernelEntry>(
        load_kernel(compile_fused_conv_bias_silu_add(p, driver)));
    entry->params = p;
    KernelEntry* ptr = entry.get();
    fused_bias_silu_add_cache_[key] = std::move(entry);
    return *ptr;
}

const KernelEntry& RocMLIRKernelCache::get_or_compile_slice_conv_bias_silu(
        const ConvParams& p, const std::string& driver) {
    const size_t key = p.hash() ^ static_cast<size_t>(0xB3C5D7E9F1A2ULL);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = slice_conv_bias_silu_cache_.find(key);
    if (it != slice_conv_bias_silu_cache_.end() && it->second && it->second->function != nullptr
            && it->second->params == p)
        return *it->second;

    std::unique_ptr<KernelEntry> entry;
    try {
        // Try MIGraphX-guided slice-conv compilation (produces mlir_slice_convolution_*).
        entry = std::make_unique<KernelEntry>(load_kernel(compile_slice_conv_bias_silu(p, driver)));
    } catch (const std::exception& ex) {
        // MIGraphX doesn't emit MLIR for this shape (e.g., 3×3 conv uses binary code objects).
        // Fall back to standard rocMLIR v3 bias+SiLU compilation. The execute path will
        // offset the input pointer by c_start channels to implement the slice.
        std::cerr << "[SliceConv] Falling back to v3 rocmlir for "
                  << "C_full=" << p.C_full << " C=" << p.C << " c_start=" << p.c_start
                  << " K=" << p.K << " R=" << p.R << "x" << p.S << "\n";
        ConvParams p_sliced = p;
        p_sliced.C_full = 0;  // no slice in the kernel itself; Execute applies pointer offset
        p_sliced.c_start = 0;
        entry = std::make_unique<KernelEntry>(
            load_kernel(compile_fused_conv_bias_act(p_sliced, Activation::Sigmoid)));
        entry->info.bias_fused = true;
        entry->info.silu_fused = true;
    }
    entry->params = p;
    KernelEntry* ptr = entry.get();
    slice_conv_bias_silu_cache_[key] = std::move(entry);
    return *ptr;
}

const KernelEntry& RocMLIRKernelCache::insert_migraphx_silu(const ConvParams& p, CompiledConv&& compiled) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t key = p.hash();
    auto it = migraphx_silu_cache_.find(key);
    if (it != migraphx_silu_cache_.end()) return *it->second;
    // migraphx conv+bias+silu: bias and silu are fused in the kernel
    compiled.bias_fused = true;
    compiled.silu_fused = true;
    compiled.skip_add_fused = false;
    auto entry = std::make_unique<KernelEntry>(load_kernel(std::move(compiled)));
    entry->params = p;
    auto* ptr = entry.get();
    migraphx_silu_cache_[key] = std::move(entry);
    return *ptr;
}

const KernelEntry& RocMLIRKernelCache::insert_migraphx_silu_add(const ConvParams& p, CompiledConv&& compiled) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t key = p.hash();
    auto it = migraphx_silu_add_cache_.find(key);
    if (it != migraphx_silu_add_cache_.end()) return *it->second;
    // migraphx conv+bias+silu+add: bias, silu, and skip-add are all fused in the kernel
    compiled.bias_fused = true;
    compiled.silu_fused = true;
    compiled.skip_add_fused = true;
    auto entry = std::make_unique<KernelEntry>(load_kernel(std::move(compiled)));
    entry->params = p;
    auto* ptr = entry.get();
    migraphx_silu_add_cache_[key] = std::move(entry);
    return *ptr;
}

void RocMLIRKernelCache::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    auto unload = [](auto& m) {
        for (auto& [k, e] : m) if (e && e->module) hipModuleUnload(e->module);
        m.clear();
    };
    unload(cache_);
    unload(fused_bias_cache_);
    unload(fused_bias_relu_cache_);
    unload(fused_bias_sigmoid_cache_);
    unload(fused_bias_silu_add_cache_);
    unload(slice_conv_bias_silu_cache_);
    unload(migraphx_silu_cache_);
    unload(migraphx_silu_add_cache_);
}

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
