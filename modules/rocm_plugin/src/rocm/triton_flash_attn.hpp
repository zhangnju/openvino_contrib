// Triton Flash Attention C++ wrapper for OV ROCm Plugin.
// Loads AOT-compiled Triton HSACO and launches via hipModuleLaunchKernel.
#pragma once
#include <hip/hip_runtime.h>
#include <string>
#include <memory>
#include "hiprtc_compiler.hpp"

namespace ov { namespace rocm_gpu { namespace triton_fa {

struct KernelMeta {
    int block_m = 128, block_n = 128, num_warps = 4, shared_mem = 8192;
    std::string kernel_name = "_fwd_kernel";
};

struct TritonFAKernel {
    std::shared_ptr<hiprtc::CompiledKernel> kernel;
    KernelMeta meta;
};

// AOT-compile flash attention for given shapes. Calls Python subprocess.
// Returns nullptr on failure.
std::shared_ptr<TritonFAKernel> compile(
    int seqlen_q, int seqlen_k, int headdim,
    const std::string& arch,
    const std::string& bias_type = "none",
    bool causal = false);

// Launch the compiled flash attention kernel.
// Q,K,V: [batch, seq, heads, headdim] fp16 contiguous
// O:     [batch, seqlen_q, heads, headdim] fp16, pre-allocated
// Lse:   [batch, heads, seqlen_q_rounded] fp32, workspace
// TMP:   [batch, heads, seqlen_q_rounded] fp32, workspace
// Size of the kernel args struct (for workbuffer allocation)
constexpr size_t kFAArgsSize = 148;

// Launch flash attention kernel.
// args_buf: caller-provided persistent buffer (>= kFAArgsSize bytes, GPU-accessible or pinned).
//           Must remain valid until the kernel completes on the given stream.
//           Use a per-stream mutable workbuffer to avoid stack-lifetime issues.
void launch(const TritonFAKernel& kernel, hipStream_t stream,
            void* args_buf,
            const void* Q, const void* K, const void* V, const void* Bias, void* O,
            void* Lse, void* TMP,
            int batch, int heads, int seqlen_q, int seqlen_k, int headdim,
            float softmax_scale,
            int stride_qb, int stride_qh, int stride_qm,
            int stride_kb, int stride_kh, int stride_kn,
            int stride_vb, int stride_vh, int stride_vn,
            int stride_ob, int stride_oh, int stride_om);

}}} // namespace
