// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/parameter.hpp>
#include <ops/result.hpp>
#include <ops/tensor_iterator.hpp>

#include "rocm_iexecution_delegator.hpp"

namespace ov {
namespace rocm_gpu {

/**
 * Basic implementaion for IExecutionDelegator interface
 */
class SimpleExecutionDelegator : public IExecutionDelegator {
public:
    /**
     * Constructor of SimpleExecutionDelegator class
     */
    SimpleExecutionDelegator() = default;

    /**
     * Dummy set_stream implementation
     */
    void set_stream(const rocm::Stream& stream) override{};

    /**
     * Dummy start_stage implementation
     */
    void start_stage() override {}

    /**
     * Dummy stop_stage implementation
     */
    virtual void stop_stage(PerfStages stage) override{};

    /**
     * Execute sequence from SubGraph/TensorIterator class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    virtual void execute_sequence(const SubGraph* subGraphPtr,
                                  const MemoryManager& memoryManager,
                                  const Workbuffers::mutable_buffer& buffer,
                                  const InferenceRequestContext& context) override {
        for (auto& op : subGraphPtr->getExecSequence()) {
            const auto& inputTensors = memoryManager.inputTensorPointers(*op, buffer);
            const auto& outputTensors = memoryManager.outputTensorPointers(*op, buffer);
            const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
            op->Execute(context, inputTensors, outputTensors, workBuffers);
        }
    };

    /**
     * Capture sequence from SubGraph/TensorIterator class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    virtual void capture_sequence(const SubGraph* subGraphPtr,
                                  const MemoryManager& memoryManager,
                                  const Workbuffers::mutable_buffer& buffer,
                                  InferenceRequestContext& context) override {
        for (auto& op : subGraphPtr->getExecSequence()) {
            const auto& inputTensors = memoryManager.inputTensorPointers(*op, buffer);
            const auto& outputTensors = memoryManager.outputTensorPointers(*op, buffer);
            const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
            op->Capture(context, inputTensors, outputTensors, workBuffers);
        }
    };

    /**
     * Call ExecuteGraph for all operations from SubGraph class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    virtual void execute_graph_sequence(const SubGraph* subGraphPtr,
                                        const MemoryManager& memoryManager,
                                        const Workbuffers::mutable_buffer& buffer,
                                        InferenceRequestContext& context) override {
        for (auto& op : subGraphPtr->getExecSequence()) {
            const auto& inputTensors = memoryManager.inputTensorPointers(*op, buffer);
            const auto& outputTensors = memoryManager.outputTensorPointers(*op, buffer);
            const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
            op->ExecuteGraph(context, inputTensors, outputTensors, workBuffers);
        }
    };

    /**
     * Dummy get_performance_counts implementation
     */
    virtual const std::vector<ov::ProfilingInfo> get_performance_counts() const override {
        return std::vector<ov::ProfilingInfo>{};
    };

    /**
     * Dummy process_events implementation
     */
    virtual void process_events() override{};

    /**
     * Dummy set_rocm_event_record_mode implementation
     */
    //virtual void set_rocm_event_record_mode(rocm::Event::RecordMode mode) override{};
};

}  // namespace rocm_gpu
}  // namespace ov
