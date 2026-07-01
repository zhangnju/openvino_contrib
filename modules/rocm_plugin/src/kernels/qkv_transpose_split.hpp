// QKV transpose+split kernel: [nW, sq, 3*H*hd] → Q,K,V [nW, H, sq, hd]
#pragma once
#include <hip/hip_runtime.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

void launchQKVTransposeSplit(hipStream_t stream,
                             const void* x, void* q, void* k, void* v,
                             int nW, int sq, int H, int head, bool is_fp16,
                             bool norm_q = false, bool norm_k = false, float eps = 1e-12f);

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
