#include "prelu.hpp"
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {
/*
template <typename T>
struct PReluOpImpl {
    __device__ static inline T op(T in0, T in1) { return in0 > T{} ? in0 : in0 * in1; }
};
*/

static __global__ void PReluOpImpl_fp32(const float* in0, NumpyBroadcastMapper m0,
                                        const float* in1, NumpyBroadcastMapper m1,
                                        float* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = in0[m0.srcIndex(i)];
    out[i] = a > 0.0f ? a : a * in1[m1.srcIndex(i)];
}

static __global__ void PReluOpImpl_fp16(const __half* in0, NumpyBroadcastMapper m0,
                                     const __half* in1, NumpyBroadcastMapper m1,
                                     __half* out, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = __half2float(in0[m0.srcIndex(i)]);
    out[i] = __float2half(a > 0.0f ? a : a * __half2float(in1[m1.srcIndex(i)]));
}
PRelu::PRelu(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
{
    element_type_ = element_type;
    max_threads_per_block_ = max_threads_per_block;
    num_elements_ = out_num_elements;

    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);
}

void PRelu::operator()(hipStream_t stream,
                       const void* in0,
                       const NumpyBroadcastMapper& in0_mapper,
                       const void* in1,
                       const NumpyBroadcastMapper& in1_mapper,
                       void* out) const {
    const float* in0_fp32=nullptr;
    const float* in1_fp32=nullptr;
    float* out_fp32=nullptr;
    switch(element_type_)
    {
        case Type_t::f32:
            in0_fp32 = (const float*) in0;
            in1_fp32 = (const float*) in1;
            out_fp32 = (float*) out;
            PReluOpImpl_fp32<<<num_blocks_, threads_per_block_, 0, stream>>>(
                in0_fp32, in0_mapper, in1_fp32, in1_mapper, out_fp32, (unsigned)num_elements_);
            break;
        case Type_t::f16:
            PReluOpImpl_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>(
                (const __half*)in0, in0_mapper, (const __half*)in1, in1_mapper, (__half*)out, (unsigned)num_elements_);
            break;
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by PRelu operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
