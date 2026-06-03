// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/subgraph.hpp>

#include "rocm_itopology_runner.hpp"

namespace ov {
namespace rocm_gpu {

class EagerTopologyRunner final : public SubGraph, public ITopologyRunner {
public:
    EagerTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model) : SubGraph(context, model) {}
    ~EagerTopologyRunner() override = default;

    void Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override {
        Workbuffers workbuffers{};
        workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
        SubGraph::Execute(context, {}, {}, workbuffers);
    }

    void Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const override{};

    void Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const override{};

    void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override{};

    const SubGraph& GetSubGraph() const override { return *this; }

    std::size_t GetrocmGraphsCount() const override { return 0; }
};

}  // namespace rocm_gpu
}  // namespace ov
