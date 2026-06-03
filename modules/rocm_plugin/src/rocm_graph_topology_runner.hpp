// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_creation_context.hpp>
#include <rocm_itopology_runner.hpp>
#include <ops/subgraph.hpp>

namespace ov {
namespace rocm_gpu {

class rocmGraphTopologyRunner final : public ITopologyRunner {
public:
    rocmGraphTopologyRunner(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);

    rocmGraphTopologyRunner(const CreationContext& context,
                            const std::shared_ptr<const ov::Model>& model,
                            const SubGraph::ExecSequence& sequence,
                            const std::shared_ptr<MemoryManager>& memoryManager);

    ~rocmGraphTopologyRunner() override = default;

    void Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const override;
    void Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;

    void Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const override;
    void UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const override;

    const SubGraph& GetSubGraph() const override;
    std::size_t GetrocmGraphsCount() const override;

    bool hasNestedRunners() const;

private:
    explicit rocmGraphTopologyRunner(const CreationContext& context, const SubGraph& subgraph);

    void Capture(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const;
    void UpdateCapture(InferenceRequestContext& context) const;

    std::vector<SubGraph> subgraphs_;
    SubGraph orig_subgraph_;
    std::size_t rocm_graphs_count_;
};

}  // namespace rocm_gpu
}  // namespace ov
