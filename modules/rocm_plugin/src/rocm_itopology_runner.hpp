// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_inference_request_context.hpp>
#include <memory_manager/rocm_workbuffers.hpp>

namespace ov {
namespace rocm_gpu {

class SubGraph;

struct ITopologyRunner {
    virtual ~ITopologyRunner() = default;

    virtual void Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const = 0;
    virtual void Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const = 0;

    virtual void Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const = 0;
    virtual void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const = 0;

    virtual const SubGraph& GetSubGraph() const = 0;
    virtual std::size_t GetrocmGraphsCount() const = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
