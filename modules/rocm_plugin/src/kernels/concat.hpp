// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include "rocm/float16.hpp"
#include "details/rocm_type_traits.hpp"
#include "details/tensor_helpers.hpp"
#include "details/error.hpp"
#include "details/numpy_broadcast_mapper.hpp" 

#include "details/rocm_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace rocm_gpu {
namespace kernel {

class Concat {
public:
    struct Chunk {
        size_t input;
        size_t offset;
    };

    Concat(Type_t element_type,
           size_t numInputs,
           std::vector<Chunk>&& chunks,
           size_t chunkSize,
           size_t allChunkSize,
           size_t numBlocks,
           size_t threadsPerBlock);
    Concat(Concat&&) = default;
    Concat& operator=(Concat&&) = default;

    void operator()(hipStream_t stream, const void* chunks, const void* const* src, void* dst) const;

    [[nodiscard]] size_t immutableWbSize() const {
        if (use_compact_) return sizeof(CompactEntry) * compact_entries_.size();
        return sizeof(Chunk) * chunks_.size();
    }
    [[nodiscard]] size_t mutableWbSize() const { return sizeof(float*) * num_inputs_; }
    [[nodiscard]] const void* immutableWbData() const {
        if (use_compact_) return compact_entries_.data();
        return chunks_.data();
    }
    [[nodiscard]] bool useCompact() const { return use_compact_; }

    struct CompactEntry {
        size_t cumulative_chunks;
        size_t input;
        size_t offset;
    };

private:
    void Call(hipStream_t stream, const void* chunks, const float* const* src, float* dst) const;
    void Call_fp16(hipStream_t stream, const void* chunks, const __half* const* src, __half* dst) const;
    void CallCompact_fp16(hipStream_t stream, const void* entries, size_t n_entries,
                          const __half* const* src, __half* dst) const;

    Type_t element_type_{};
    size_t num_inputs_{};
    std::vector<Chunk> chunks_;
    std::vector<CompactEntry> compact_entries_;
    bool use_compact_{false};
    size_t chunk_size_{};
    size_t all_chunk_size_{};
    size_t num_blocks_{};
    size_t threads_per_block_{};
};

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
