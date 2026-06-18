// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MIGraphX-based fused kernel infrastructure.
// Directly uses MIGraphX C++ API (libmigraphx_gpu.so) to build programs,
// apply fuse_pointwise_reduce pass, compile to GPU, and extract HSACO binary.
// No shell commands, no temp files, no ONNX intermediate format.

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <mutex>
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace fused_kernel {

struct KernelInfo {
    std::vector<uint8_t> hsaco;       // Raw ELF binary
    std::string          symbol_name; // Kernel function name
    unsigned             global_x{0}; // Grid size (blocks)
    unsigned             local_x{0};  // Block size (threads per block)
    int                  output_arg{-1};
};

// Compile a masked softmax fused kernel: scores + mask → Softmax.
// MIGraphX fuse_pointwise_reduce fuses add+softmax into a single kernel.
// Kernel inputs: (scores, mask, output) — 3 device pointer args.
std::optional<KernelInfo> compile_masked_softmax(
    int64_t heads, int64_t sq, int64_t sk);

// Compile a FC+GELU fused kernel: x[seq,in] × W[in,out] + bias → GELU → [seq,out].
// MIGraphX fuse_pointwise_reduce merges the GELU epilogue into the GEMM kernel.
// Kernel inputs: (x, W, bias, output) — 4 device pointer args.
std::optional<KernelInfo> compile_fc_gelu(
    int64_t seq,
    int64_t in_dim,
    int64_t out_dim);

// Compile a LayerNorm fused kernel using MIGraphX's fuse_pointwise_reduce.
// Builds a MIGraphX program directly (no ONNX), applies fusion passes,
// compiles with migraphx::gpu::target, and extracts HSACO from gpu::code_object_op.
std::optional<KernelInfo> compile_layernorm_residual(
    int64_t seq_len,
    int64_t hidden,
    bool    with_residual,
    const std::string& arch,
    float   epsilon = 1e-12f);

// Global cache keyed by (arch, seq_len, hidden, with_residual).
class KernelCache {
public:
    static KernelCache& instance();
    std::optional<KernelInfo> get(const std::string& key) const;
    void put(const std::string& key, KernelInfo info);
    static std::string make_key(const std::string& arch, int64_t seq,
                                 int64_t hidden, bool residual);
private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, KernelInfo> cache_;
};

// Load KernelInfo into HIP (hipModuleLoadData + hipModuleGetFunction).
struct HipKernel {
    hipModule_t   module{nullptr};
    hipFunction_t func{nullptr};
    std::string   symbol;
    unsigned      grid_x{0};
    unsigned      block_x{0};
    ~HipKernel();
};
HipKernel load_kernel(const KernelInfo& info);

}  // namespace fused_kernel
}  // namespace rocm_gpu
}  // namespace ov
