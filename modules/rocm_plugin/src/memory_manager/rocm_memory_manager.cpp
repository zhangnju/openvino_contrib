// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_memory_manager.hpp"
#include "rocm_operation_base.hpp"

namespace ov {
namespace rocm_gpu {

MemoryManager::MemoryManager(DeviceMemBlock::Ptr immutableTensors,
                             MemoryModel::Ptr mutableMemoryModel,
                             DeviceMemBlock::Ptr immutableWorkbufferMemory)
    : immutable_tensors_{immutableTensors},
      mutable_tensors_model_{mutableMemoryModel},
      immutable_workbuffers_{immutableWorkbufferMemory} {}

MemoryManager::InputTensors MemoryManager::inputTensorPointers(const IOperationMeta& operation,
                                                               rocm::DevicePointer<void*> mutableBufferPtr) const {
    InputTensors result;
    for (auto id : operation.GetInputIds()) {
        const void* ptr = immutable_tensors_->deviceTensorPtr(id);
        if (ptr == nullptr) ptr = mutable_tensors_model_->deviceTensorPtr(mutableBufferPtr.cast<uint8_t*>(), id);
        OPENVINO_ASSERT(ptr != nullptr, "Tensor not found. ID is " + to_string(id));
        result.emplace_back(ptr);
    }
    return result;
}

MemoryManager::OutputTensors MemoryManager::outputTensorPointers(const IOperationMeta& operation,
                                                                 rocm::DevicePointer<void*> mutableBufferPtr) const {
    OutputTensors result;
    for (auto id : operation.GetOutputIds()) {
        void* ptr = mutable_tensors_model_->deviceTensorPtr(mutableBufferPtr.cast<uint8_t*>(), id);

        OPENVINO_ASSERT(ptr != nullptr, "Tensor not found. ID is " + to_string(id));
        result.emplace_back(ptr);
    }
    return result;
}

Workbuffers MemoryManager::workBuffers(const IOperationExec& operation,
                                       rocm::DevicePointer<void*> mutableBufferPtr,
                                       void* pinnedPool) const {
    Workbuffers result{};
    const auto& indices = operation.GetWorkbufferIds();
    for (const auto immutable_id : indices.immutableIds) {
        void* ptr = immutable_workbuffers_->deviceBufferPtr(immutable_id);
        OPENVINO_ASSERT(ptr != nullptr, "Workbuffer not found. ID is " + std::to_string(immutable_id));
        result.immutable_buffers.emplace_back(ptr);
    }
    for (const auto mutable_id : indices.mutableIds) {
        void* ptr = mutable_tensors_model_->deviceBufferPtr(mutableBufferPtr.cast<uint8_t*>(), mutable_id);
        OPENVINO_ASSERT(ptr != nullptr, "Workbuffer not found. ID is " + std::to_string(mutable_id));
        result.mutable_buffers.emplace_back(ptr);
    }
    // Pinned host buffers: slice the per-request pinned pool by recorded offsets.
    if (pinnedPool) {
        for (const auto offset : indices.pinnedOffsets) {
            result.pinned_buffers.push_back(static_cast<uint8_t*>(pinnedPool) + offset);
        }
    }
    return result;
}

}  // namespace rocm_gpu
}  // namespace ov
