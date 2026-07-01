// Fast MVN (Mean-Variance Normalization) kernel: warp-per-row, f32 accumulate.
// Replaces MIOpen's miopenLayerNormForward/gridwise_generic_reduce for MVN ops.
// y = (x - mean(x)) / sqrt(var(x) + eps)
#pragma once
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Pure MVN: normalize only (no scale/bias).
// x [rows, cols] fp16 → y [rows, cols] fp16
void launchMvnFast(hipStream_t stream,
                   const void* x, void* y,
                   int rows, int cols, float eps);

// MVN + scale + bias + optional residual (full LayerNorm tail).
// y = (x - mean) / sqrt(var + eps) * scale[cols] + bias[cols] [+ residual[rows,cols]]
void launchMvnFastEpi(hipStream_t stream,
                      const void* x, void* y,
                      const void* scale, const void* bias,
                      const void* residual,
                      int rows, int cols, float eps);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
