// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_device_mem_block.hpp"

#include <hip/hip_runtime.h>

#include <rocm/runtime.hpp>
#include <iostream>

namespace ov {
namespace rocm_gpu {

DeviceMemBlock::DeviceMemBlock(MemoryModel::Ptr model) : model_{std::move(model)} {}

void* DeviceMemBlock::deviceBufferPtr(const BufferID& id) const {
    if (ptrdiff_t offset = 0; model_->offsetForBuffer(id, offset))
        return reinterpret_cast<uint8_t*>(device_mem_ptr_.get()) + offset;
    return nullptr;
}

void* DeviceMemBlock::deviceTensorPtr(const TensorID& id) const {
    if (auto bufferPtr = deviceBufferPtr(id.GetBuffer().GetId()); bufferPtr) {
        return reinterpret_cast<uint8_t*>(bufferPtr) + id.GetOffset();
    }
    return nullptr;
}

}  // namespace rocm_gpu
}  // namespace ov
