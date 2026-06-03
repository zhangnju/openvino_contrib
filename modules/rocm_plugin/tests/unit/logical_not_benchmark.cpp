// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <rocm/device_pointers.hpp>
#include <rocm_config.hpp>
#include <rocm_graph_context.hpp>
#include <rocm_operation_registry.hpp>
#include <rocm_simple_execution_delegator.hpp>
#include <iomanip>
#include <openvino/op/logical_not.hpp>
#include <openvino/op/parameter.hpp>
#include <ops/parameter.hpp>
#include <random>
#include <typeinfo>
#include <vector>

namespace {

using devptr_t = rocm::DevicePointer<void*>;
using cdevptr_t = rocm::DevicePointer<const void*>;

struct LogicalNotBenchmark : testing::Test {
    using TensorID = ov::rocm_gpu::TensorID;
    using ElementType = std::uint8_t;
    static constexpr int length = 10 * 1024;
    static constexpr size_t size = length * sizeof(ElementType);
    ov::rocm_gpu::ThreadContext threadContext{{}};
    rocm::Allocation in_alloc = threadContext.stream().malloc(size);
    rocm::Allocation out_alloc = threadContext.stream().malloc(size);
    std::vector<cdevptr_t> inputs{in_alloc};
    std::vector<devptr_t> outputs{out_alloc};
    std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    ov::rocm_gpu::OperationBase::Ptr operation = [this] {
        const bool optimizeOption = false;
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{length});
        auto node = std::make_shared<ov::op::v1::LogicalNot>(param->output(0));
        auto& registry = ov::rocm_gpu::OperationRegistry::getInstance();
        auto op = registry.createOperation(ov::rocm_gpu::CreationContext{threadContext.device(), optimizeOption},
                                           node,
                                           std::vector<TensorID>{TensorID{0u}},
                                           std::vector<TensorID>{TensorID{0u}});
        return op;
    }();
};

TEST_F(LogicalNotBenchmark, DISABLED_benchmark) {
    constexpr int kNumAttempts = 20;
    ov::rocm_gpu::CancellationToken token{};
    ov::rocm_gpu::SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::rocm_gpu::rocmGraphContext rocmGraphContext{};
    ov::rocm_gpu::InferenceRequestContext context{emptyTensor,
                                                    emptyMapping,
                                                    emptyTensor,
                                                    emptyMapping,
                                                    threadContext,
                                                    token,
                                                    simpleExecutionDelegator,
                                                    rocmGraphContext};
    auto& stream = context.getThreadContext().stream();
    std::vector<ElementType> in(length);
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<> dist{std::numeric_limits<std::uint8_t>::min(),
                                         std::numeric_limits<std::uint8_t>::max()};
    auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
    std::generate(in.begin(), in.end(), gen);
    stream.upload(in_alloc, in.data(), size);
    ov::rocm_gpu::Workbuffers workbuffers{};
    hipEvent_t start;
    hipEvent_t stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float totalExecTime{};
    for (int i = 0; i < kNumAttempts; i++) {
        hipEventRecord(start, stream.get());
        operation->Execute(context, inputs, outputs, workbuffers);
        hipEventRecord(stop, stream.get());
        hipEventSynchronize(stop);
        float execTime{};
        hipEventElapsedTime(&execTime, start, stop);
        totalExecTime += execTime;
    }
    stream.synchronize();
    std::cout << std::fixed << std::setfill('0') << "LogicalNot execution time: " << totalExecTime * 1000 / kNumAttempts
              << " microseconds\n";
}

}  // namespace
