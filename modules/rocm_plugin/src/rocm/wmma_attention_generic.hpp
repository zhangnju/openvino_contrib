// Generic WMMA Fused Attention for small sequences (sq,sk <= 256, hd <= 64).
// Uses rocWMMA/builtin WMMA for QK and PV matrix multiplications.
// Designed for CV models (DETR, BEVFormer, PETR-small) with small-medium attention.
//
// Input:  Q [B, Sq, H, D], K [B, Sk, H, D], V [B, Sk, H, D] — fp16, contiguous
// Output: O [B, Sq, H, D] — fp16
//
// Launch: grid = (B * H), block = 32 (1 warp)
// Constraints: Sq % 16 == 0, Sk % 16 == 0, D % 16 == 0, D <= 64, Sk <= 256
#pragma once
#include <hip/hip_runtime.h>
#include <string>
#include <memory>
#include "hiprtc_compiler.hpp"

namespace ov { namespace rocm_gpu { namespace wmma_attn {

struct WMMAAttnKernel {
    std::shared_ptr<hiprtc::CompiledKernel> kernel;
    int sq, sk, hd;
};

// Compile a WMMA attention kernel for given shapes via hipRTC.
std::shared_ptr<WMMAAttnKernel> compile(int sq, int sk, int hd, const std::string& arch);

// Launch the kernel.
void launch(const WMMAAttnKernel& kernel, hipStream_t stream,
            const void* Q, const void* K, const void* V, void* O,
            int batch, int heads, float scale);

}}} // namespace
