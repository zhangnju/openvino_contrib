// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <rocm/float16.hpp>

#include "multiply.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

template <typename T>
struct MultiplyOpImpl {
    __device__ static inline T op(T in0, T in1) { return in0 * in1; }
};

/*
static __global__ void MultiplyOpImpl_fp32(const float* in0, const float* in1,float* out)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    float x = in0[i];
    float y = in1[i];

    out[i] = x * y;
}
*/

// Fast periodic broadcast multiply: out[i] = in0[i] * in1[i % period]
// Covers swin per-channel *[1,1,1,C] (period=C, repeat=1) and per-head *[1,H,1,1]
static __global__ void multiply_bcast_periodic_h2(
    const __half2* __restrict__ in0,
    const __half* __restrict__ in1,
    __half2* __restrict__ out,
    unsigned n2, unsigned period, unsigned repeat)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;
    __half2 a = in0[i];
    if (repeat == 1) {
        // Last-dim broadcast: in1 repeats every period elements
        unsigned j = (i * 2) % period;
        __half2 b = __halves2half2(in1[j], in1[j + 1]);
        out[i] = __hmul2(a, b);
    } else {
        // Per-head broadcast: in1[head_idx] scalar broadcast
        unsigned elem_idx = i * 2;
        unsigned head_idx = (elem_idx / repeat) % period;
        __half b = in1[head_idx];
        __half2 b2 = __half2half2(b);
        out[i] = __hmul2(a, b2);
    }
}

Multiply::Multiply(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}
/*
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = out_num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);
}
*/
void Multiply::launch_bcast_periodic_f16(hipStream_t stream,
                                        const void* in0, const void* in1, void* out,
                                        size_t total, size_t period, size_t repeat) const {
    unsigned n2 = total / 2;
    unsigned blk = 256;
    unsigned grd = (n2 + blk - 1) / blk;
    hipLaunchKernelGGL(multiply_bcast_periodic_h2,
        dim3(grd), dim3(blk), 0, stream,
        reinterpret_cast<const __half2*>(in0),
        reinterpret_cast<const __half*>(in1),
        reinterpret_cast<__half2*>(out),
        n2, (unsigned)period, (unsigned)repeat);
}

void Multiply::operator()(hipStream_t stream,
                          const void* in0,
                          const NumpyBroadcastMapper& in0_mapper,
                          const void* in1,
                          const NumpyBroadcastMapper& in1_mapper,
                          void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
    /*
    const float* in0_fp32=nullptr;
    const float* in1_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            in1_fp32 = (const float*) in1;
            out_fp32 = (float*) out;
            MultiplyOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(in0_fp32, in1_fp32,out_fp32);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Multiply operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
    */
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
