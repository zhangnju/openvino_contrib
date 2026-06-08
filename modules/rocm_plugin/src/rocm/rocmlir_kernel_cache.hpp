// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Thread-safe cache: ConvParams → loaded hipModule_t + hipFunction_t.
// The first compile for a given shape runs rocmlir-driver (slow).
// All subsequent executions use the cached module (fast).

#pragma once

#include "rocmlir_compiler.hpp"

#include <hip/hip_runtime.h>
#include <memory>
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
    ConvParams    params;  // stored for hash-collision detection
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

    // For fused conv+bias+SiLU+skip-Add (5-arg kernel)
    const KernelEntry& get_or_compile_fused_bias_silu_add(const ConvParams& p,
                                                            const std::string& driver_path = "");

    // For Slice+Conv+Bias+SiLU (4-arg, kernel reads full input and slices internally)
    const KernelEntry& get_or_compile_slice_conv_bias_silu(const ConvParams& p,
                                                             const std::string& driver_path = "");

    // Insert a pre-compiled migraphx kernel (conv+bias+silu or conv+bias+silu+add).
    // Used when ROCMLIR_EPILOGUE_FUSION=1 bypasses the standard compile path.
    const KernelEntry& insert_migraphx_silu(const ConvParams& p, CompiledConv&& compiled);
    const KernelEntry& insert_migraphx_silu_add(const ConvParams& p, CompiledConv&& compiled);

    void clear();

    static RocMLIRKernelCache& global();

private:
    KernelEntry load_kernel(CompiledConv&& compiled);

    std::mutex mu_;
    // Use unique_ptr so that pointers to KernelEntry remain stable across map rehash.
    // Key: hash(ConvParams), Value: heap-allocated KernelEntry
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> cache_;
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> fused_bias_cache_;
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> fused_bias_relu_cache_;
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> fused_bias_sigmoid_cache_;
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> fused_bias_silu_add_cache_;
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> slice_conv_bias_silu_cache_;
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> migraphx_silu_cache_;
    std::unordered_map<size_t, std::unique_ptr<KernelEntry>> migraphx_silu_add_cache_;
};

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
