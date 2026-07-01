// Winograd F(4,3) convolution for 3×3 stride=1 fp16
// Reduces arithmetic by 2.25× vs standard im2col GEMM.
// 3-phase: input transform → batched GEMM (rocBLAS) → output transform + bias + activation.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstddef>

namespace ov {
namespace rocm_gpu {
namespace kernel {

class WinogradConv {
public:
    // Filter transform: [K, C, 3, 3] → [36, K, C] (fp16)
    // Called once; result cached as immutable workbuffer.
    static void filterTransform(const __half* filter,  // [K, C, 3, 3]
                                __half* transformed,   // [36, K, C]
                                int K, int C,
                                hipStream_t stream);

    // Fused Winograd forward: input_transform + batched GEMM + output_transform
    // Input:  [N, C, H, W] fp16, same-padding (pad=1) assumed
    // Output: [N, K, H, W] fp16
    // filter_xform: pre-transformed [36, K, C] from filterTransform()
    // workspace: mutable buffer for V [36, T, C] + M [36, T, K]
    static void forward(const __half* input,
                        const __half* filter_xform,
                        const __half* bias,           // [K] or nullptr
                        __half* output,
                        __half* workspace,
                        int N, int C, int H, int W, int K,
                        bool relu,
                        rocblas_handle blas_handle,
                        hipStream_t stream);

    // Workspace bytes: V [36, T, C] + M [36, T, K]
    static size_t workspaceBytes(int N, int C, int H, int W, int K);

    // Filter transform buffer size in bytes
    static size_t filterTransformBytes(int K, int C);
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
