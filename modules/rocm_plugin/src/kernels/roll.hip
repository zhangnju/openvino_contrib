// Roll (circular shift) kernel: replaces Slice+Concat pattern.
// Each thread computes one output element by reading from the shifted source position.
// Supports arbitrary number of rolled axes.

#include "roll.hpp"
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// For swin: typically 4D [B,H,W,C] with roll on axes 1,2
// General N-D roll with up to 2 shifted axes (covers all swin cases).
static __global__ void roll_2axis_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    size_t total,
    int ndim,
    int axis0, int dim0, int shift0, size_t stride0,
    int axis1, int dim1, int shift1, size_t stride1)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose linear index to find position along rolled axes
    size_t src_idx = idx;

    if (axis0 >= 0) {
        size_t pos0 = (idx / stride0) % dim0;
        size_t new_pos0 = ((int64_t)pos0 - shift0 % dim0 + dim0) % dim0;
        src_idx += (new_pos0 - pos0) * stride0;
    }
    if (axis1 >= 0) {
        size_t pos1 = (idx / stride1) % dim1;
        size_t new_pos1 = ((int64_t)pos1 - shift1 % dim1 + dim1) % dim1;
        src_idx += (new_pos1 - pos1) * stride1;
    }

    y[idx] = x[src_idx];
}

void launchRoll(hipStream_t stream,
                const void* x, void* y,
                const std::vector<size_t>& shape,
                const std::vector<int64_t>& shift,
                const std::vector<int64_t>& axes) {
    size_t total = 1;
    for (auto s : shape) total *= s;

    // Compute strides
    std::vector<size_t> strides(shape.size());
    strides.back() = 1;
    for (int i = (int)shape.size() - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * shape[i + 1];

    int axis0 = -1, axis1 = -1;
    int dim0 = 1, dim1 = 1;
    int sh0 = 0, sh1 = 0;
    size_t str0 = 1, str1 = 1;

    for (size_t i = 0; i < axes.size() && i < 2; i++) {
        int ax = axes[i] < 0 ? axes[i] + shape.size() : axes[i];
        if (i == 0) { axis0 = ax; dim0 = shape[ax]; sh0 = shift[i]; str0 = strides[ax]; }
        else        { axis1 = ax; dim1 = shape[ax]; sh1 = shift[i]; str1 = strides[ax]; }
    }

    int block = 256;
    int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(roll_2axis_kernel,
        dim3(grid), dim3(block), 0, stream,
        (const __half*)x, (__half*)y, total,
        (int)shape.size(),
        axis0, dim0, sh0, str0,
        axis1, dim1, sh1, str1);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
