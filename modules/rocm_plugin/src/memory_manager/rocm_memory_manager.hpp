// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gsl/span>
#include <memory>
#include <vector>

#include "rocm/device_pointers.hpp"
#include "rocm_device_mem_block.hpp"
#include "rocm_workbuffers.hpp"
#include "memory_manager/model/rocm_memory_model.hpp"

namespace ov {
namespace rocm_gpu {

class MemoryModel;
class IOperationMeta;
class IOperationExec;

/**
 * @brief MemoryManager provides device side tensor pointers by
 * combining together tensor locations from shared immutable memory
 * blob and infer request specific mutable memory blob.
 *
 * Shared immutable tensors are allocated when executable network is loaded
 * and then shared between multiple infer requests. Additionally, MemoryManager
 * allocates mutable memory blob which is used by only one infer request at a time.
 */
class MemoryManager {
public:
    using InputTensors = std::vector<rocm::DevicePointer<const void*>>;
    using OutputTensors = std::vector<rocm::DevicePointer<void*>>;

    /**
     * @param[in] immutableTensors Immutable memory blob which stores constant tensors
     * which are used by multiple infer requests at the same time.
     * @param[in] mutableMemoryModel Infer request specific mutable memory model. It is
     * used to allocate a memory which is used by a single infer request at a time.
     * @param[in] immutableWorkbufferMemory Blob for immutable workbuffers
     */
    MemoryManager(DeviceMemBlock::Ptr immutableTensors,
                  MemoryModel::Ptr mutableMemoryModel,
                  DeviceMemBlock::Ptr immutableWorkbufferMemory = nullptr);

    /**
     * Maps input tensor identifiers into device side tensor pointers.
     * @param[in] operation An operation which defines input tensors.
     * @param[in] mutableBufferPtr A memory block based on which mapping is performed.
     * @returns An array of corresponding input tensor pointers.
     * @throws ov::Exception if any of tensor pointers is not found
     */
    InputTensors inputTensorPointers(const IOperationMeta& operation,
                                     rocm::DevicePointer<void*> mutableBufferPtr) const;

    /**
     * Maps output tensor identifiers into device side tensor pointers.
     * @param[in] operation An operation which defines output tensors.
     * @param[in] mutableBufferPtr A memory block based on which mapping is performed.
     * @returns An array of corresponding output tensor pointers.
     * @throws ov::Exception if any of tensor pointers is not found
     */
    OutputTensors outputTensorPointers(const IOperationMeta& operation,
                                       rocm::DevicePointer<void*> mutableBufferPtr) const;

    /**
     * Maps operation onto device side work work buffer pointers.
     * @param[in] operation An operation
     * @param[in] mutableBufferPtr A memory block based on which mapping is performed.
     * @returns Work buffer pointers
     * @throws ov::Exception if any of tensor pointers is not found
     */
    Workbuffers workBuffers(const IOperationExec& operation,
                            rocm::DevicePointer<void*> mutableBufferPtr,
                            void* pinnedPool = nullptr) const;

    /**
     * Returns immutable tensors
     * @return DeviceMemBlock
     */
    [[nodiscard]] const DeviceMemBlock& immutableTensors() const { return *immutable_tensors_; }

    /**
     * Returns mutable tensors model
     * @return MemoryModel
     */
    [[nodiscard]] MemoryModel::Ptr mutableTensorsMemoryModel() const { return mutable_tensors_model_; }

    /**
     * Returns immutable workbuffers
     * @return DeviceMemBlock
     */
    [[nodiscard]] const DeviceMemBlock& immutableWorkbuffers() const { return *immutable_workbuffers_; }

    // Total pinned host memory required per infer-request (computed by OperationBuffersExtractor).
    size_t pinnedPoolBytesPerRequest() const { return pinned_pool_bytes_; }
    void setPinnedPoolBytesPerRequest(size_t bytes) { pinned_pool_bytes_ = bytes; }

private:
    DeviceMemBlock::Ptr immutable_tensors_;
    MemoryModel::Ptr mutable_tensors_model_;
    DeviceMemBlock::Ptr immutable_workbuffers_;
    size_t pinned_pool_bytes_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
