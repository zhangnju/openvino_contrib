// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm/runtime.hpp>
#include <rocm_graph_context.hpp>
#include <gsl/pointers>

#include "memory_manager/model/rocm_memory_model.hpp"

namespace ov {
namespace rocm_gpu {

/**
 * @brief Allocates and owns continuous memory blob on rocm device.
 * Uses MemoryModel to determine a size of memory to allocate and
 * to map tensor offsets to device side pointers.
 */
class DeviceMemBlock {
public:
    using Ptr = std::shared_ptr<DeviceMemBlock>;

    /**
     * @throws ov::Exception if device memory block allocation
     * failed.
     */
    DeviceMemBlock(MemoryModel::Ptr model);

    /**
     * Provides buffer memory address if any.
     *
     * @param [in] id Buffer identifier.
     * @returns device memory pointer if buffer is located within the blob
     * or nullptr otherwise.
     */
    void* deviceBufferPtr(const BufferID& id) const;

    /**
     * Provides tensor memory address if any.
     *
     * @param [in] id Tensor identifier.
     * @returns device memory pointer if tensor is located within the blob
     * or nullptr otherwise.
     */
    void* deviceTensorPtr(const TensorID& id) const;

    rocm::DeviceBuffer<uint8_t> view() const {
        return {static_cast<uint8_t*>(device_mem_ptr_.get()), model_->deviceMemoryBlockSize()};
    }

    const std::vector<BufferID>& bufferIds() const { return model_->bufferIds(); }

    MemoryModel::Ptr memoryModel() const { return model_; }

    ::ov::rocm_gpu::rocmGraphContext& rocmGraphContext() { return rocm_graph_context_; }

    // Per-request pinned host memory pool (hipHostMalloc).
    // Ops with pinned workbuffer requests receive offsets into this pool.
    // The pool holds H2D source data (aux ptrs, output ptrs etc.) so hipGraph
    // can record stable H2D addresses and replay them after ExecuteGraph() updates.
    void* pinnedPool() const { return pinned_pool_; }
    void  allocPinnedPool(size_t bytes);
    size_t pinnedPoolSize() const { return pinned_pool_size_; }

    ~DeviceMemBlock();

private:
    MemoryModel::Ptr model_;
    rocm::DefaultAllocation device_mem_ptr_ = rocm::DefaultStream::stream().malloc(model_->deviceMemoryBlockSize());
    ::ov::rocm_gpu::rocmGraphContext rocm_graph_context_;
    void*  pinned_pool_      = nullptr;
    size_t pinned_pool_size_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
