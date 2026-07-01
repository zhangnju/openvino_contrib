#pragma once
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// L2 normalize along last dim: y[i] = x[i] / max(||x||_2, eps)
// x,y: [rows, cols] fp16, cols = normalize dim
void launchL2Normalize(hipStream_t stream,
                       const void* x, void* y,
                       int rows, int cols, float eps);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
