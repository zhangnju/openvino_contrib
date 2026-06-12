// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory_manager/rocm_device_mem_block.hpp>

#include "cancellation_token.hpp"
#include "rocm_graph_context.hpp"
#include "rocm_tensor_mapping_context.hpp"
#include "rocm_thread_context.hpp"

namespace ov {
namespace rocm_gpu {

class IExecutionDelegator;

class InferenceRequestContext {
public:
    /**
     * @brief A smart pointer to the InferenceRequestContext object
     */
    using Ptr = std::shared_ptr<InferenceRequestContext>;
    using WeakPtr = std::weak_ptr<InferenceRequestContext>;

    InferenceRequestContext(const std::vector<std::shared_ptr<ov::Tensor>>& inputs,
                            const std::map<std::string, std::size_t>& inputMapping,
                            const std::vector<std::shared_ptr<ov::Tensor>>& outputs,
                            const std::map<std::string, std::size_t>& outputMapping,
                            const ThreadContext& threadContext,
                            CancellationToken& token,
                            IExecutionDelegator& executionDelegator,
                            rocmGraphContext& rocmGraphContext,
                            bool isBenchmarkMode = false)
        : threadContext{threadContext},
          token{token},
          executionDelegator{executionDelegator},
          tensor_mapping_context_{inputs, inputMapping, outputs, outputMapping},
          rocm_graph_context_{rocmGraphContext},
          is_benchmark_mode_{isBenchmarkMode} {}

    // don't allow storing references to temporary
    template <typename... Args>
    InferenceRequestContext(std::vector<std::shared_ptr<ov::Tensor>>&& inputs,
                            std::map<std::string, std::size_t>&& inputMapping,
                            std::vector<std::shared_ptr<ov::Tensor>>&& outputs,
                            std::map<std::string, std::size_t>&& outputMapping,
                            Args... args) = delete;

    InferenceRequestContext(std::vector<std::shared_ptr<ov::Tensor>>&& inputs,
                            std::map<std::string, std::size_t>&& inputMapping,
                            std::vector<std::shared_ptr<ov::Tensor>>&& outputs,
                            std::map<std::string, std::size_t>&& outputMapping,
                            const ThreadContext& threadContext) = delete;

    const ThreadContext& getThreadContext() const noexcept { return threadContext; }
    [[nodiscard]] ov::rocm_gpu::CancellationToken& getCancellationToken() const noexcept { return token; }
    [[nodiscard]] IExecutionDelegator& getExecutionDelegator() const noexcept { return executionDelegator; }
    [[nodiscard]] bool isBenchmarkMode() const noexcept { return is_benchmark_mode_; }
    [[nodiscard]] const TensorMappingContext& getTensorMappingContext() const { return tensor_mapping_context_; }
    [[nodiscard]] const rocmGraphContext& getrocmGraphContext() const { return rocm_graph_context_; }
    [[nodiscard]] rocmGraphContext& getrocmGraphContext() { return rocm_graph_context_; }

    // Per-request pinned host memory pool for hipGraph-safe H2D transfers.
    // Set from DeviceMemBlock::pinnedPool() before each Execute/Capture sequence.
    void setPinnedPool(void* pool) noexcept { pinned_pool_ = pool; }
    [[nodiscard]] void* getPinnedPool() const noexcept { return pinned_pool_; }

    void setCurrentrocmGraphInfo(IrocmGraphInfo& info) { current_rocm_graph_info_ = &info; }

    IrocmGraphInfo& getCurrentrocmGraphInfo() {
        OPENVINO_ASSERT(current_rocm_graph_info_, "current_rocm_graph_info_ is nullptr");
        return *current_rocm_graph_info_;
    }

private:
    const ThreadContext& threadContext;
    CancellationToken& token;
    IExecutionDelegator& executionDelegator;
    const TensorMappingContext tensor_mapping_context_;
    rocmGraphContext& rocm_graph_context_;
    bool is_benchmark_mode_;
    IrocmGraphInfo* current_rocm_graph_info_ = nullptr;
    void* pinned_pool_ = nullptr;
};

}  // namespace rocm_gpu
}  // namespace ov
