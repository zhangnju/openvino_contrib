#include "hip/hip_runtime.h"
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>
#include <hip/hip_fp16.h>

#include <rocm/float16.hpp>

#include "details/error.hpp"
//#include "details/type_validator.hpp"
#include "split.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

static __global__ void split(
    const size_t numSplitChunks, const size_t splitStepSize, const size_t numSplits, const float *x, float **y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned chunkIdx = i / splitStepSize;
    const unsigned chunkOffset = i % splitStepSize;
    if (chunkIdx < numSplitChunks) {
        const unsigned splitIdx = chunkIdx % numSplits;
        const unsigned splitStepIdx = chunkIdx / numSplits;
        auto src = &x[chunkIdx * splitStepSize];
        auto dest = &y[splitIdx][splitStepIdx * splitStepSize];
        dest[chunkOffset] = src[chunkOffset];
    }
}

// Separate __half kernel — float* reinterpret causes wrong stride for 2-byte elements
static __global__ void split_fp16(
    const size_t numSplitChunks, const size_t splitStepSize, const size_t numSplits, const __half *x, __half **y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned chunkIdx = i / splitStepSize;
    const unsigned chunkOffset = i % splitStepSize;
    if (chunkIdx < numSplitChunks) {
        const unsigned splitIdx = chunkIdx % numSplits;
        const unsigned splitStepIdx = chunkIdx / numSplits;
        auto src = &x[chunkIdx * splitStepSize];
        auto dest = &y[splitIdx][splitStepIdx * splitStepSize];
        dest[chunkOffset] = src[chunkOffset];
    }
}

Split::Split(Type_t element_type,
             size_t num_splits,
             size_t num_split_chunks,
             size_t split_step_size,
             unsigned num_blocks,
             unsigned threads_per_block)
    : element_type_{element_type},
      num_splits_{num_splits},
      num_split_chunks_{num_split_chunks},
      split_step_size_{split_step_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    //TypeValidator<AllElementTypesSwitch>::check(element_type_);
}

void Split::operator()(hipStream_t stream, const void *src, void **dst) const {
    const float* src_fp32=nullptr;
    float** dst_fp32=nullptr;
    switch (element_type_) {
        case Type_t::f16:
            // Use __half kernel to correctly handle 2-byte element stride
            return Call_fp16(stream,
                             reinterpret_cast<const __half*>(src),
                             reinterpret_cast<__half**>(dst));
        case Type_t::f32:
            src_fp32 = (const float*) src;
            dst_fp32 = (float**) dst;
            return Call(stream, src_fp32, dst_fp32);
        /*
        case Type_t::f64:
            return Call<double>(stream, src, dst);
        case Type_t::i8:
            return Call<int8_t>(stream, src, dst);
        case Type_t::i16:
            return Call<int16_t>(stream, src, dst);
        case Type_t::i32:
            return Call<int32_t>(stream, src, dst);
        case Type_t::i64:
            return Call<int64_t>(stream, src, dst);
        case Type_t::u8:
            return Call<uint8_t>(stream, src, dst);
        case Type_t::u16:
            return Call<uint16_t>(stream, src, dst);
        case Type_t::u32:
            return Call<uint32_t>(stream, src, dst);
        case Type_t::u64:
            return Call<uint64_t>(stream, src, dst);
        */
        default:
            throw_ov_exception(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<Type_t>(element_type_)));
    }
}


void Split::Call(hipStream_t stream, const float *src, float **dst) const {
    split<<<num_blocks_, threads_per_block_, 0, stream>>>(
        num_split_chunks_, split_step_size_, num_splits_, src, dst);
}

void Split::Call_fp16(hipStream_t stream, const __half *src, __half **dst) const {
    split_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>(
        num_split_chunks_, split_step_size_, num_splits_, src, dst);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
