// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Thread-safe cache: ConvParams → loaded hipModule_t + hipFunction_t.
// The first compile for a given shape runs rocmlir-driver (slow).
// All subsequent executions use the cached module (fast).

#pragma once

#include "rocmlir_compiler.hpp"

#include <hip/hip_runtime.h>
#include <mutex>
#include <unordered_map>
#include <string>

namespace ov {
namespace rocm_gpu {
namespace rocmlir {

struct KernelEntry {
    hipModule_t   module   = nullptr;
    hipFunction_t function = nullptr;
    CompiledConv  info;
};

class RocMLIRKernelCache {
public:
    // Returns the kernel entry for the given params, compiling on first access.
    // Throws if compilation or HIP module loading fails.
    const KernelEntry& get_or_compile(const ConvParams& p,
                                      const std::string& driver_path = "");

    // For fused conv+bias
    const KernelEntry& get_or_compile_fused_bias(const ConvParams& p,
                                                  const std::string& driver_path = "");

    // For fused conv+bias+activation
    const KernelEntry& get_or_compile_fused_bias_act(const ConvParams& p,
                                                      Activation act,
                                                      const std::string& driver_path = "");

    void clear();

    static RocMLIRKernelCache& global();

private:
    KernelEntry load_kernel(CompiledConv&& compiled);

    std::mutex mu_;
    // Key: hash(ConvParams), Value: KernelEntry
    std::unordered_map<size_t, KernelEntry> cache_;
    std::unordered_map<size_t, KernelEntry> fused_bias_cache_;
    std::unordered_map<size_t, KernelEntry> fused_bias_relu_cache_;
    std::unordered_map<size_t, KernelEntry> fused_bias_sigmoid_cache_;
};

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
