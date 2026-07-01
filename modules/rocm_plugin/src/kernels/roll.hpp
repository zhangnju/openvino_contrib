#pragma once
#include <hip/hip_runtime.h>
#include <vector>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Circular shift (roll) along specified axes.
// x,y: same shape, fp16. shift[i] applied to axes[i].
void launchRoll(hipStream_t stream,
                const void* x, void* y,
                const std::vector<size_t>& shape,
                const std::vector<int64_t>& shift,
                const std::vector<int64_t>& axes);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
