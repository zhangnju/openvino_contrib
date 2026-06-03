// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <memory_manager/model/details/rocm_memory_utils.hpp>
#include <vector>

#include "dnn_be.hpp"
#include "event.hpp"

namespace rocm {

inline std::vector<std::shared_ptr<DnnBEExecutionPlan>> getAllExecutionPlansFromHeuristics(
    const std::shared_ptr<DnnBEOperationGraphDescriptor>& graph, const rocm::DnnHandle& dnnHandle) {
    std::vector<std::shared_ptr<DnnBEExecutionPlan>> plans;
    {
        auto heuristics =
            rocm::DnnBEEngineHeuristicsDescriptorBuilder().setOpGraph(graph).setMode(MIOPEN_HEUR_MODE_INSTANT).build();

        std::vector<std::shared_ptr<rocm::DnnBEEngineConfigDescriptor>> configs = heuristics->getEngineConfigs();
        for (const auto& config : configs) {
            try {
                auto plan = rocm::DnnBEExecutionPlanBuilder().setDnnHandle(dnnHandle).setEngineConfig(config).build();
                plans.push_back(std::move(plan));
            } catch (const ov::Exception&) {
                continue;
            }
        }
    }

    return std::move(plans);
}

template <size_t NumBenchmarks>
std::shared_ptr<rocm::DnnBEExecutionPlan> performBenchmarks(
    const rocm::DnnHandle& dnnHandle,
    const std::vector<std::shared_ptr<rocm::DnnBEExecutionPlan>>& plans,
    rocm::DnnBEVariantPackBuilder& variantPackBuilder) {
    auto getDescendSortedWorkspaceSizes = [](const std::vector<std::shared_ptr<rocm::DnnBEExecutionPlan>>& plans) {
        std::vector<size_t> workspace_sizes{};
        std::transform(plans.begin(), plans.end(), std::back_inserter(workspace_sizes), [](const auto& p) {
            return p->getWorkspaceSize();
        });
        std::sort(workspace_sizes.begin(), workspace_sizes.end(), std::greater<size_t>{});
        return workspace_sizes;
    };

    auto tryAllocateMaxWorkspace =
        [](const std::vector<size_t>& workspace_sizes) -> std::optional<std::pair<rocm::DefaultAllocation, size_t>> {
        for (const auto workspace_size : workspace_sizes) {
            try {
                const auto aligned_workspace_size = ov::rocm_gpu::applyAllignment(workspace_size);
                rocm::DefaultAllocation workspace = rocm::DefaultStream::stream().malloc(aligned_workspace_size);
                return std::optional<std::pair<rocm::DefaultAllocation, size_t>>{
                    {std::move(workspace), workspace_size}};
            } catch (...) {
                // NOTE: If not enough memory try another workspace size
            }
        }
        return std::nullopt;
    };

    auto filterPlansByWorkspaceSize = [](const std::vector<std::shared_ptr<rocm::DnnBEExecutionPlan>>& plans,
                                         const size_t max_workspace_size) {
        auto filtered_plans = plans;
        auto erased =
            std::remove_if(filtered_plans.begin(), filtered_plans.end(), [max_workspace_size](const auto& plan) {
                return plan->getWorkspaceSize() > max_workspace_size;
            });
        filtered_plans.erase(erased, filtered_plans.end());
        return filtered_plans;
    };

    const auto& workspace_sizes = getDescendSortedWorkspaceSizes(plans);
    auto max_workspace = tryAllocateMaxWorkspace(workspace_sizes);
    auto [workspace, max_workspace_size] = max_workspace.value();
    auto filtered_plans = filterPlansByWorkspaceSize(plans, max_workspace_size);

    if (max_workspace) {
        variantPackBuilder.setWorkspase(workspace.get());
    }
    auto variantPack = variantPackBuilder.build();

    auto executeBenchmarkStep = [&](auto& plan) {
        throwIfError(::miopenBackendExecute(dnnHandle.get(), plan->get(), variantPack->get()));
    };

    rocm::Event start, stop;
    rocm::Device{}.synchronize();

    auto stream = dnnHandle.getStream();

    std::vector<float> time;
    for (auto& plan : filtered_plans) {
        // Warmup
        executeBenchmarkStep(plan);

        start.record(stream);
        for (size_t i = 0; i < NumBenchmarks; ++i) {
            executeBenchmarkStep(plan);
        }
        stop.record(stream);
        stop.synchronize();
        const float time_ms = stop.elapsedSince(start) / NumBenchmarks;
        time.push_back(time_ms);
    }
    const auto min_time = std::min_element(time.begin(), time.end());
    const auto best_time_index = std::distance(time.begin(), min_time);

    return filtered_plans[best_time_index];
}

inline std::vector<size_t> generateStrides(gsl::span<const size_t> dim, miopenTensorLayout_t filterFormat) {
    std::vector<size_t> strides(dim.size());
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the MIOPEN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == miopenTensorNCHW) {
        strides[strides.size() - 1] = 1;
        for (int64_t d = strides.size() - 2; d >= 0; d--) {
            strides[d] = strides[d + 1] * dim[d + 1];
        }
    } else {
        // Here we assume that the format is MIOPEN_TENSOR_NHWC
        strides[1] = 1;
        strides[strides.size() - 1] = strides[1] * dim[1];
        for (int64_t d = strides.size() - 2; d >= 2; d--) {
            strides[d] = strides[d + 1] * dim[d + 1];
        }
        strides[0] = strides[2] * dim[2];
    }
    return strides;
}

}  // namespace rocm
