// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>
#include <rocm/device_pointers.hpp>
#include <vector>

#include "tensor_types.hpp"

namespace ov {
namespace rocm_gpu {

/**
 * @brief WorkbufferRequest - a POD structure describing operator's memory demands
 *
 * pinned_sizes: host pinned memory (hipHostMalloc) per-infer-request.
 * Used by ops that perform H2D copies inside Execute/Capture; having a stable
 * pinned address allows hipGraph to record the memcpy and replay it correctly.
 */
struct WorkbufferRequest {
    using size_in_bytes_t = size_t;
    std::vector<size_in_bytes_t> immutable_sizes;
    std::vector<size_in_bytes_t> mutable_sizes;
    std::vector<size_in_bytes_t> pinned_sizes;  // per-request host-pinned memory
};

/**
 * @brief Workbuffers - structure holding preallocated memory buffers
 */
struct Workbuffers {
    using immutable_buffer = rocm::DevicePointer<const void*>;
    using mutable_buffer = rocm::DevicePointer<void*>;

    std::vector<immutable_buffer> immutable_buffers;
    std::vector<mutable_buffer> mutable_buffers;
    std::vector<void*> pinned_buffers;  // host-pinned pointers (per-request)

    template <std::size_t Index>
    rocm::DeviceBuffer<std::uint8_t> createMutableSpanFrom(size_t workspaceSize) const {
        if (!workspaceSize) return {};
        return {mutable_buffers.at(Index).cast<std::uint8_t*>().get(), workspaceSize};
    }

    template <std::size_t Index>
    rocm::DeviceBuffer<const std::uint8_t> createImmutableSpanFrom(size_t workspaceSize) const {
        if (!workspaceSize) return {};
        return {immutable_buffers.at(Index).cast<const std::uint8_t*>().get(), workspaceSize};
    }
};

/**
 * @brief WorkbufferIds - structure holding the memory buffers' indices
 *
 * pinnedOffsets: byte offsets into the per-request DeviceMemBlock::pinnedPool().
 * Each entry maps to one entry in WorkbufferRequest::pinned_sizes.
 */
struct WorkbufferIds {
    using vector_of_ids = std::vector<BufferID>;
    vector_of_ids immutableIds;
    vector_of_ids mutableIds;
    std::vector<size_t> pinnedOffsets;  // offsets into DeviceMemBlock pinned pool
};

}  // namespace rocm_gpu
}  // namespace ov
