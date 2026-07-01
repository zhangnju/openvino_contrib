// L2 Normalize kernel: y[i] = x[i] / max(||x_row||_2, eps)
// Each warp handles one row of `cols` elements.
// Fuses: Power(2) → ReduceSum → Sqrt → Maximum(eps) → Divide into single pass.

#include "l2_normalize.hpp"
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

static __global__ void l2_normalize_warp_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int rows, int cols, float eps)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= rows) return;

    const __half* row_in  = x + (size_t)warp_id * cols;
    __half*       row_out = y + (size_t)warp_id * cols;

    // Pass 1: compute sum of squares
    float sq_sum = 0.0f;
    for (int i = lane; i < cols; i += 32) {
        float v = __half2float(row_in[i]);
        sq_sum += v * v;
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_xor(sq_sum, offset);

    float inv_norm = rsqrtf(fmaxf(sq_sum, eps));

    // Pass 2: normalize and write
    for (int i = lane; i < cols; i += 32)
        row_out[i] = __float2half(__half2float(row_in[i]) * inv_norm);
}

void launchL2Normalize(hipStream_t stream,
                       const void* x, void* y,
                       int rows, int cols, float eps) {
    const int warps_per_block = 4;
    const int block = warps_per_block * 32;
    const int grid = (rows + warps_per_block - 1) / warps_per_block;
    hipLaunchKernelGGL(l2_normalize_warp_kernel,
                       dim3(grid), dim3(block), 0, stream,
                       (const __half*)x, (__half*)y, rows, cols, eps);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
