// Fast MVN kernel: warp-per-row, f32 accumulate, replaces MIOpen gridwise_generic_reduce.
// Each warp handles one row of `cols` elements:
//   1. Compute mean (warp reduce sum / cols)
//   2. Compute variance (warp reduce sum of (x-mean)^2 / cols)
//   3. Normalize: y = (x - mean) * rsqrt(var + eps)
// Optional epilogue: scale, bias, residual add.

#include "mvn_fast.hpp"
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

static __global__ void mvn_fast_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int rows, int cols, float eps)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= rows) return;

    const __half* row_in  = x + (size_t)warp_id * cols;
    __half*       row_out = y + (size_t)warp_id * cols;

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32)
        sum += __half2float(row_in[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor(sum, offset);
    float mean = sum / cols;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int i = lane; i < cols; i += 32) {
        float v = __half2float(row_in[i]) - mean;
        var_sum += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_xor(var_sum, offset);
    float inv_std = rsqrtf(var_sum / cols + eps);

    // Pass 3: normalize and write
    for (int i = lane; i < cols; i += 32)
        row_out[i] = __float2half((__half2float(row_in[i]) - mean) * inv_std);
}

static __global__ void mvn_fast_epi_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const __half* __restrict__ scale,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    int rows, int cols, float eps)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= rows) return;

    const __half* row_in  = x + (size_t)warp_id * cols;
    __half*       row_out = y + (size_t)warp_id * cols;
    const __half* row_res = residual ? (residual + (size_t)warp_id * cols) : nullptr;

    // Pass 1: mean
    float sum = 0.0f;
    for (int i = lane; i < cols; i += 32)
        sum += __half2float(row_in[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor(sum, offset);
    float mean = sum / cols;

    // Pass 2: variance
    float var_sum = 0.0f;
    for (int i = lane; i < cols; i += 32) {
        float v = __half2float(row_in[i]) - mean;
        var_sum += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        var_sum += __shfl_xor(var_sum, offset);
    float inv_std = rsqrtf(var_sum / cols + eps);

    // Pass 3: normalize + scale + bias + residual
    for (int i = lane; i < cols; i += 32) {
        float val = (__half2float(row_in[i]) - mean) * inv_std;
        if (scale) val *= __half2float(scale[i]);
        if (bias)  val += __half2float(bias[i]);
        if (row_res) val += __half2float(row_res[i]);
        row_out[i] = __float2half(val);
    }
}

void launchMvnFast(hipStream_t stream,
                   const void* x, void* y,
                   int rows, int cols, float eps) {
    const int warps_per_block = 4;
    const int block = warps_per_block * 32;
    const int grid = (rows + warps_per_block - 1) / warps_per_block;
    hipLaunchKernelGGL(mvn_fast_kernel,
                       dim3(grid), dim3(block), 0, stream,
                       (const __half*)x, (__half*)y, rows, cols, eps);
}

void launchMvnFastEpi(hipStream_t stream,
                      const void* x, void* y,
                      const void* scale, const void* bias,
                      const void* residual,
                      int rows, int cols, float eps) {
    const int warps_per_block = 4;
    const int block = warps_per_block * 32;
    const int grid = (rows + warps_per_block - 1) / warps_per_block;
    hipLaunchKernelGGL(mvn_fast_epi_kernel,
                       dim3(grid), dim3(block), 0, stream,
                       (const __half*)x, (__half*)y,
                       (const __half*)scale, (const __half*)bias,
                       (const __half*)residual,
                       rows, cols, eps);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
