// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "abs.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

//namespace cumath = rocm::math;

template <typename T>
struct AbsOpImpl {
    __device__ static inline T op(T x) {
        return x < T{0} ? -x : x;
    }
};
/*
static __global__ void AbsImpl_fp32(const float* in0, float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    out[i] = rocm::math::abs(in0[i]);
}

static __global__ void AbsImpl_u8(const unsigned char* in0, unsigned char* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp=(float)in0[i];
    out[i] = (unsigned char) (rocm::math::abs(temp));
}
*/
Abs::Abs(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
 : impl_{element_type, max_threads_per_block, num_elements} {}

void Abs::operator()(hipStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
