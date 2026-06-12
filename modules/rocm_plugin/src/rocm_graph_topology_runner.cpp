// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rocm_graph_topology_runner.hpp"

#include "rocm/event.hpp"
#include "rocm/runtime.hpp"
#include "ops/tensor_iterator.hpp"

namespace ov {
namespace rocm_gpu {

rocmGraphTopologyRunner::rocmGraphTopologyRunner(const CreationContext& context, const SubGraph& subgraph)
    : orig_subgraph_(subgraph), rocm_graphs_count_{0} {
    std::vector<SubGraph::ExecSequence> sequences;
    SubGraph::ExecSequence currentSequence;
    const auto& origSequence = orig_subgraph_.getExecSequence();
    const auto totalSize = origSequence.size();
    OPENVINO_ASSERT(totalSize != 0, "ExecSequence size is 0");

    rocmGraphCompatibility lastOpCompatibility = origSequence[0]->GetrocmGraphCompatibility();
    currentSequence.push_back(origSequence[0]);
    for (std::size_t i = 1; i < totalSize; ++i) {
        const auto& op = origSequence[i];
        auto comp = op->GetrocmGraphCompatibility();
        if (comp != lastOpCompatibility || comp == rocmGraphCompatibility::SPECIAL) {
            lastOpCompatibility = comp;
            sequences.emplace_back(std::move(currentSequence));
            currentSequence.clear();
        }
        if (comp == rocmGraphCompatibility::SPECIAL) {
            auto sg = std::dynamic_pointer_cast<SubGraph>(op);
            sg->initializeRunner();
            rocm_graphs_count_ += sg->GetrocmGraphsCount();
        }
        currentSequence.push_back(op);
    }
    sequences.emplace_back(std::move(currentSequence));

    const auto& model = orig_subgraph_.getModel();
    const auto& memoryManager = orig_subgraph_.memoryManager();
    for (const auto& sequence : sequences) {
        subgraphs_.emplace_back(context, model, sequence, memoryManager);
        if (subgraphs_.back().GetrocmGraphCompatibility() == rocmGraphCompatibility::FULL) {
            ++rocm_graphs_count_;
        }
    }
}

rocmGraphTopologyRunner::rocmGraphTopologyRunner(const CreationContext& context,
                                                 const std::shared_ptr<const ov::Model>& model)
    : rocmGraphTopologyRunner(context, {context, model}) {}

rocmGraphTopologyRunner::rocmGraphTopologyRunner(const CreationContext& context,
                                                 const std::shared_ptr<const ov::Model>& model,
                                                 const SubGraph::ExecSequence& sequence,
                                                 const std::shared_ptr<MemoryManager>& memoryManager)
    : rocmGraphTopologyRunner(context, {context, model, sequence, memoryManager}) {}

void rocmGraphTopologyRunner::Run(InferenceRequestContext& context, const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    auto& graphPack = context.getCurrentrocmGraphInfo();
    std::size_t graphIndex = 0;
    for (auto& subgraph : subgraphs_) {
        auto compatibility = subgraph.GetrocmGraphCompatibility();
        if (compatibility == rocmGraphCompatibility::FULL) {
            graphPack.select_current_graph(graphIndex);
            // Allow ops to refresh CPU-side data (pinned pool slots) before replay.
            // ExecuteGraph() is a no-op for most ops; for FusedElementwise/Split it
            // updates the pinned aux/output pointer arrays so the captured H2D nodes
            // copy the current inference's GPU addresses into the device workbuffers.
            subgraph.ExecuteGraph(context, {}, {}, workbuffers);
            graphPack.launch(stream);
            graphIndex++;
        } else if (compatibility == rocmGraphCompatibility::SPECIAL) {
            graphPack.select_current_graph(graphIndex);
            context.setCurrentrocmGraphInfo(graphPack.get_current_graph());
            subgraph.ExecuteGraph(context, {}, {}, workbuffers);
            graphIndex++;
        } else {
            subgraph.Execute(context, {}, {}, workbuffers);
        }
    }
}

void rocmGraphTopologyRunner::Run(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    context.setCurrentrocmGraphInfo(context.getrocmGraphContext());
    Run(context, workbuffers);
}

void rocmGraphTopologyRunner::Capture(InferenceRequestContext& context, const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    auto& graphPack = context.getCurrentrocmGraphInfo();
    graphPack.reset();
    for (const auto& subgraph : subgraphs_) {
        auto compatibility = subgraph.GetrocmGraphCompatibility();
        if (compatibility == rocmGraphCompatibility::FULL) {
            graphPack.add(rocmGraphInfo::create());
            rocm::GraphCapture capture{stream};
            {
                auto scope = capture.getScope();
                subgraph.Capture(context, {}, {}, workbuffers);
            }
            graphPack.set_current_graph(capture.getGraph());
        } else if (compatibility == rocmGraphCompatibility::SPECIAL) {
            auto& currentGraph =
                hasNestedRunners() ? graphPack.add(rocmGraphContext::create()) : graphPack.add(rocmGraphInfo::create());
            context.setCurrentrocmGraphInfo(currentGraph);
            subgraph.Capture(context, {}, {}, workbuffers);
        }
    }
    OPENVINO_ASSERT(rocm_graphs_count_ == graphPack.get_graphs_count());
}

void rocmGraphTopologyRunner::Capture(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    Workbuffers workbuffers{};
    workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
    context.setCurrentrocmGraphInfo(context.getrocmGraphContext());
    Capture(context, workbuffers);
}

const SubGraph& rocmGraphTopologyRunner::GetSubGraph() const {
    return orig_subgraph_;
}

std::size_t rocmGraphTopologyRunner::GetrocmGraphsCount() const { return rocm_graphs_count_; }

bool rocmGraphTopologyRunner::hasNestedRunners() const {
    return std::any_of(
        subgraphs_.begin(), subgraphs_.end(), [](const SubGraph& sg) { return sg.hasTopologyRunners(); });
}

void rocmGraphTopologyRunner::UpdateContext(InferenceRequestContext& context, const DeviceMemBlock& memoryBlock) const {
    if (context.getrocmGraphContext().is_initialized()) {
        UpdateCapture(context);
    } else {
        // Warm-up run: execute all ops eagerly first so all hipModuleLoadData calls
        // (which internally call hipMalloc) complete before hipStreamBeginCapture.
        // hipMalloc is not allowed during stream capture (returns hipErrorStreamCaptureUnsupported)
        // and would invalidate the capture, so kernels must be pre-loaded.
        {
            Workbuffers workbuffers{};
            workbuffers.mutable_buffers.emplace_back(memoryBlock.view().data());
            for (const auto& subgraph : subgraphs_) {
                subgraph.Execute(context, {}, {}, workbuffers);
            }
            context.getThreadContext().stream().synchronize();
        }
        Capture(context, memoryBlock);
    }
}

void rocmGraphTopologyRunner::UpdateCapture(InferenceRequestContext& context) const {
    context.getrocmGraphContext().update_capture(context.getTensorMappingContext());
}

}  // namespace rocm_gpu
}  // namespace ov
