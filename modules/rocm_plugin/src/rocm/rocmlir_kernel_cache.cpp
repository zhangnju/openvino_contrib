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
    if (it != cache_.end()) return it->second;

    auto& entry = cache_[key];
    entry = load_kernel(compile_conv(p, driver));
    return entry;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused conv + bias
// ─────────────────────────────────────────────────────────────────────────────

const KernelEntry& RocMLIRKernelCache::get_or_compile_fused_bias(
        const ConvParams& p, const std::string& driver) {
    const size_t key = p.hash() ^ static_cast<size_t>(0xB1a51ULL);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = fused_bias_cache_.find(key);
    if (it != fused_bias_cache_.end()) return it->second;

    auto& entry = fused_bias_cache_[key];
    entry = load_kernel(compile_fused_conv_bias(p, driver));
    return entry;
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
    if (it != target_cache->end()) return it->second;

    auto& entry = (*target_cache)[key];
    entry = load_kernel(compile_fused_conv_bias_act(p, act, driver));
    return entry;
}

void RocMLIRKernelCache::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& [k, e] : cache_)               if (e.module) hipModuleUnload(e.module);
    for (auto& [k, e] : fused_bias_cache_)    if (e.module) hipModuleUnload(e.module);
    for (auto& [k, e] : fused_bias_relu_cache_)    if (e.module) hipModuleUnload(e.module);
    for (auto& [k, e] : fused_bias_sigmoid_cache_) if (e.module) hipModuleUnload(e.module);
    cache_.clear();
    fused_bias_cache_.clear();
    fused_bias_relu_cache_.clear();
    fused_bias_sigmoid_cache_.clear();
}

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
