#include "hip/hip_runtime.h"
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>
#include <hip/hip_fp16.h>

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
    if (i >= allChunkSize) { return; }
    const unsigned chunkIdx = (i / chunkSize) % numInputChunks;
    const unsigned dataIdx = i % chunkSize;
    const auto& chunk = chunks[chunkIdx];
    dst[chunkIdx * chunkSize + dataIdx] = (src[chunk.input] + chunk.offset)[dataIdx];
}

// Separate __half kernel — necessary because float* reinterpret of __half* uses wrong stride
static __global__ void concat_fp16(const Concat::Chunk* chunks,
                                   const size_t allChunkSize,
                                   const size_t numInputChunks,
                                   const size_t chunkSize,
                                   const __half* const* src,
                                   __half* dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= allChunkSize) { return; }
    const unsigned chunkIdx = (i / chunkSize) % numInputChunks;
    const unsigned dataIdx = i % chunkSize;
    const auto& chunk = chunks[chunkIdx];
    dst[chunkIdx * chunkSize + dataIdx] = (src[chunk.input] + chunk.offset)[dataIdx];
}

// half2-vectorized concat: 2 contiguous elements/thread. Requires chunkSize even and every
// chunk.offset even (half2 alignment). dst chunk base (chunkIdx*chunkSize) is then also even.
static __global__ void concat_fp16_h2(const Concat::Chunk* chunks,
                                      const size_t allChunkSize2,   // total elems / 2
                                      const size_t numInputChunks,
                                      const size_t chunkSize2,      // chunkSize / 2
                                      const __half* const* src,
                                      __half* dst) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // half2 index
    if (j >= allChunkSize2) return;
    const unsigned chunkIdx = (unsigned)((j / chunkSize2) % numInputChunks);
    const size_t dataIdx2 = j % chunkSize2;
    const auto& chunk = chunks[chunkIdx];
    const __half2* s2 = reinterpret_cast<const __half2*>(src[chunk.input] + chunk.offset);
    __half2* d2 = reinterpret_cast<__half2*>(dst + (size_t)chunkIdx * (chunkSize2 * 2));
    d2[dataIdx2] = s2[dataIdx2];
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
    // When chunk table is too large (>64MB), build compact representation.
    // Instead of per-chunk {input, offset}, store per-input {cumulative_chunks, input, offset}.
    // Kernel uses binary search to find which input a chunk belongs to.
    constexpr size_t COMPACT_THRESHOLD = 64 * 1024 * 1024;
    if (sizeof(Chunk) * chunks_.size() > COMPACT_THRESHOLD && !chunks_.empty()) {
        compact_entries_.push_back({0, chunks_[0].input, chunks_[0].offset});
        for (size_t i = 1; i < chunks_.size(); ++i) {
            if (chunks_[i].input != chunks_[i-1].input) {
                compact_entries_.push_back({i, chunks_[i].input, chunks_[i].offset});
            }
        }
        use_compact_ = true;
        fprintf(stderr, "[Concat] compact mode: %zu chunks -> %zu entries (%.1fMB -> %.1fKB)\n",
                chunks_.size(), compact_entries_.size(),
                sizeof(Chunk) * chunks_.size() / 1048576.0,
                sizeof(CompactEntry) * compact_entries_.size() / 1024.0);
        chunks_.clear();
        chunks_.shrink_to_fit();
    }
}

// Compact concat kernel: uses binary search on a small entries table instead of
// per-chunk lookup. Entries table has N entries (one per input boundary).
static __global__ void concat_compact_fp16(const Concat::CompactEntry* entries,
                                            size_t n_entries,
                                            size_t total_chunks,
                                            size_t chunk_size,
                                            const __half* const* src,
                                            __half* dst) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_chunks * chunk_size) return;
    const size_t chunkIdx = i / chunk_size;
    const size_t dataIdx = i % chunk_size;
    // Binary search for which input this chunk belongs to
    size_t lo = 0, hi = n_entries - 1;
    while (lo < hi) {
        size_t mid = (lo + hi + 1) / 2;
        if (entries[mid].cumulative_chunks <= chunkIdx) lo = mid;
        else hi = mid - 1;
    }
    const auto& e = entries[lo];
    size_t local_chunk = chunkIdx - e.cumulative_chunks;
    dst[chunkIdx * chunk_size + dataIdx] = (src[e.input] + e.offset)[local_chunk * chunk_size + dataIdx];
}

void Concat::operator()(const hipStream_t stream, const void* chunks, const void* const* src, void* dst) const {
    if (use_compact_ && element_type_ == Type_t::f16) {
        return CallCompact_fp16(stream, chunks, compact_entries_.size(),
                                reinterpret_cast<const __half* const*>(src),
                                reinterpret_cast<__half*>(dst));
    }
    const float* const* src_fp32=nullptr;
    float* dst_fp32=nullptr;
    switch (element_type_) {
        case Type_t::f16:
            return Call_fp16(stream, chunks,
                             reinterpret_cast<const __half* const*>(src),
                             reinterpret_cast<__half*>(dst));
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
    concat<<<num_blocks_, threads_per_block_, 0, stream>>>(
        reinterpret_cast<const Chunk*>(chunks),
        all_chunk_size_, chunks_.size(), chunk_size_, src, dst);
}

void Concat::Call_fp16(const hipStream_t stream, const void* chunks, const __half* const* src, __half* dst) const {
    // half2 fast path: chunkSize even AND every chunk offset even AND total even.
    bool h2_ok = (chunk_size_ % 2 == 0) && (all_chunk_size_ % 2 == 0);
    if (h2_ok) for (const auto& c : chunks_) if (c.offset % 2 != 0) { h2_ok = false; break; }
    if (h2_ok) {
        const size_t n2 = all_chunk_size_ / 2;
        const size_t tpb = threads_per_block_ ? threads_per_block_ : 256;
        const size_t blk = (n2 + tpb - 1) / tpb;
        concat_fp16_h2<<<blk, tpb, 0, stream>>>(
            reinterpret_cast<const Chunk*>(chunks),
            n2, chunks_.size(), chunk_size_ / 2, src, dst);
        return;
    }
    concat_fp16<<<num_blocks_, threads_per_block_, 0, stream>>>(
        reinterpret_cast<const Chunk*>(chunks),
        all_chunk_size_, chunks_.size(), chunk_size_, src, dst);
}

void Concat::CallCompact_fp16(hipStream_t stream, const void* entries, size_t n_entries,
                               const __half* const* src, __half* dst) const {
    const size_t total_elems = all_chunk_size_;
    constexpr size_t tpb = 256;
    const size_t blk = (total_elems + tpb - 1) / tpb;
    concat_compact_fp16<<<blk, tpb, 0, stream>>>(
        reinterpret_cast<const CompactEntry*>(entries),
        n_entries, all_chunk_size_ / chunk_size_, chunk_size_, src, dst);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
