// Swin WMMA fused attention: QK^T + [bias] + softmax + PV in one kernel.
// One warp per (window, head). sq,sk <= 64, hd <= 64, wave32.
#pragma once
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Q,K,V: [nW, H, sq/sk, hd] fp16
// bias:  [H, sq, sk] fp16 or nullptr
// out:   [nW, H, sq, hd] fp16
void launchSwinFlashAttention(
    hipStream_t stream,
    const void* Q, const void* K, const void* V,
    const void* bias, void* out,
    int nW, int H, int sq, int sk, int hd,
    float scale);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
