#include "hip/hip_runtime.h"
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <rocm/float16.hpp>

#include "concat.hpp"
#include "details/error.hpp"
//#include "details/type_validator.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {


static __global__ void concat(const Concat::Chunk* chunks,
                              const size_t allChunkSize,
                              const size_t numInputChunks,
                              const size_t chunkSize,
                              const float* const* src,
                              float* dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= allChunkSize) {
        return;
    }
    const unsigned chunkIdx = (i / chunkSize) % numInputChunks;
    const unsigned dataIdx = i % chunkSize;
    const auto& chunk = chunks[chunkIdx];
    dst[chunkIdx * chunkSize + dataIdx] = (src[chunk.input] + chunk.offset)[dataIdx];
}

Concat::Concat(Type_t element_type,
               size_t num_inputs,
               std::vector<Chunk>&& chunks,
               size_t chunk_size,
               size_t all_chunk_size,
               size_t num_blocks,
               size_t threadsPerBlock)
    : element_type_{element_type},
      num_inputs_{num_inputs},
      chunks_{std::move(chunks)},
      chunk_size_{chunk_size},
      all_chunk_size_{all_chunk_size},
      num_blocks_{num_blocks},
      threads_per_block_{threadsPerBlock} {
    //TypeValidator<AllElementTypesSwitch>::check(element_type_);
}

void Concat::operator()(const hipStream_t stream, const void* chunks, const void* const* src, void* dst) const {
    const float* const* src_fp32=nullptr;
    float* dst_fp32=nullptr;
    switch (element_type_) {
        case Type_t::f16:
            // Reinterpret __half* as float* for the kernel — concat/split just copies bytes
            return Call(stream, chunks,
                        reinterpret_cast<const float* const*>(src),
                        reinterpret_cast<float*>(dst));
        case Type_t::f32:
            src_fp32 = (const float* const*) src;
            dst_fp32 = (float*) dst;
            return Call(stream, chunks, src_fp32, dst_fp32);
        /*
        case Type_t::f64:
            return Call<double>(stream, chunks, src, dst);
        case Type_t::i8:
            return Call<int8_t>(stream, chunks, src, dst);
        case Type_t::i16:
            return Call<int16_t>(stream, chunks, src, dst);
        case Type_t::i32:
            return Call<int32_t>(stream, chunks, src, dst);
        case Type_t::i64:
            return Call<int64_t>(stream, chunks, src, dst);
        case Type_t::u8:
            return Call<uint8_t>(stream, chunks, src, dst);
        case Type_t::u16:
            return Call<uint16_t>(stream, chunks, src, dst);
        case Type_t::u32:
            return Call<uint32_t>(stream, chunks, src, dst);
        case Type_t::u64:
            return Call<uint64_t>(stream, chunks, src, dst);
        */
        default:
            throw_ov_exception(fmt::format("Input element type = {} is not supported by Split operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}


void Concat::Call(const hipStream_t stream, const void* chunks, const float* const* src, float* dst) const {
    concat<<<num_blocks_, threads_per_block_, 0, stream>>>(reinterpret_cast<const Chunk*>(chunks),
                                                              all_chunk_size_,
                                                              chunks_.size(),
                                                              chunk_size_,
                                                              reinterpret_cast<const float* const*>(src),
                                                              reinterpret_cast<float*>(dst));
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
