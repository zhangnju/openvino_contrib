// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <rocm_config.hpp>
#include <rocm_graph_context.hpp>
#include <rocm_operation_registry.hpp>
#include <rocm_simple_execution_delegator.hpp>
#include <iomanip>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/select.hpp>
#include <ops/parameter.hpp>
#include <random>

namespace {

using devptr_t = rocm::DevicePointer<void*>;
using cdevptr_t = rocm::DevicePointer<const void*>;

struct SelectTest : testing::Test {
    static constexpr auto kNumOfDim = 5u;
    using OffsetType = size_t;
    static constexpr auto kOffsetBufferSize = kNumOfDim * sizeof(OffsetType);

    const ov::Shape tensorShape{32, 256, 256};
    // const ov::Shape tensorShape{1, 8, 129};
    const size_t bufferLength = ov::shape_size(tensorShape);

    const size_t conditionBufferSize = bufferLength * sizeof(uint8_t);
    const size_t thenBufferSize = bufferLength * sizeof(float);
    const size_t elseBufferSize = bufferLength * sizeof(float);
    const size_t outputBufferSize = bufferLength * sizeof(float);

    ov::rocm_gpu::ThreadContext threadContext{rocm::Device{}};
    rocm::Allocation conditionAlloc = threadContext.stream().malloc(conditionBufferSize);
    rocm::Allocation thenAlloc = threadContext.stream().malloc(thenBufferSize);
    rocm::Allocation elseAlloc = threadContext.stream().malloc(elseBufferSize);
    rocm::Allocation outputAlloc = threadContext.stream().malloc(outputBufferSize);
    std::vector<cdevptr_t> inputs{conditionAlloc, conditionAlloc, elseAlloc};
    std::vector<devptr_t> outputs{outputAlloc};
    std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    std::function<std::shared_ptr<ov::op::v1::Select>()> create_node = [this]() {
        auto condition = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{tensorShape});
        auto then_flow = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{tensorShape});
        auto else_flow = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{tensorShape});

        auto node =
            std::make_shared<ov::op::v1::Select>(condition->output(0), then_flow->output(0), else_flow->output(0));
        return node;
    };

    ov::rocm_gpu::OperationBase::Ptr operation = [this] {
        const bool optimizeOption = false;
        auto& registry = ov::rocm_gpu::OperationRegistry::getInstance();
        return registry.createOperation(ov::rocm_gpu::CreationContext{threadContext.device(), optimizeOption},
                                        create_node(),
                                        std::vector<ov::rocm_gpu::TensorID>{ov::rocm_gpu::TensorID{0u}},
                                        std::vector<ov::rocm_gpu::TensorID>{ov::rocm_gpu::TensorID{0u}});
    }();
};

namespace {
template <typename T>
void fillArrayWithRandomData(std::vector<T>& v) {
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<int> dist;
    std::function<T()> generator;
    if constexpr (std::is_same<T, uint8_t>::value) {
        dist = std::uniform_int_distribution<int>{0, 1};
        generator = [&dist, &mersenne_engine]() { return static_cast<T>(dist(mersenne_engine)); };
    } else {
        dist = std::uniform_int_distribution<int>{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
        generator = [&dist, &mersenne_engine]() {
            return static_cast<T>(10) * dist(mersenne_engine) / std::numeric_limits<int>::max();
        };
    }
    std::generate(v.begin(), v.end(), generator);
}
}  // namespace

TEST_F(SelectTest, DISABLED_benchmark) {
    using microseconds = std::chrono::duration<double, std::micro>;
    constexpr int kNumAttempts = 20000;
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

    std::vector<uint8_t> conditions(bufferLength);
    fillArrayWithRandomData(conditions);
    stream.upload(conditionAlloc, conditions.data(), conditionBufferSize);

    std::vector<float> then_flow(bufferLength);
    fillArrayWithRandomData(then_flow);
    stream.upload(thenAlloc, then_flow.data(), thenBufferSize);

    std::vector<float> else_flow(bufferLength);
    fillArrayWithRandomData(else_flow);
    stream.upload(elseAlloc, else_flow.data(), elseBufferSize);
    stream.synchronize();

    rocm::Allocation condOffsetAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    rocm::Allocation thenOffsetAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    rocm::Allocation elseOffsetAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    rocm::Allocation outputSizesAlloc = threadContext.stream().malloc(kOffsetBufferSize);
    ov::rocm_gpu::Workbuffers workbuffers{};
    workbuffers.immutable_buffers = {condOffsetAlloc, thenOffsetAlloc, elseOffsetAlloc, outputSizesAlloc};
    operation->InitSharedImmutableWorkbuffers({condOffsetAlloc, thenOffsetAlloc, elseOffsetAlloc, outputSizesAlloc});

    float elapsedTime = 0;
    rocmEvent_t start, stop;
    rocmEventCreate(&start);
    rocmEventCreate(&stop);
    for (int i = 0; i < kNumAttempts; i++) {
        rocmEventRecord(start, stream.get());
        operation->Execute(context, inputs, outputs, workbuffers);
        rocmEventRecord(stop, stream.get());
        stream.synchronize();
        rocmEventSynchronize(stop);
        float milliseconds = 0;
        rocmEventElapsedTime(&milliseconds, start, stop);
        elapsedTime += milliseconds;
    }
    std::cout << std::fixed << std::setfill('0') << "Sigmoid execution time: " << elapsedTime * 1000 / kNumAttempts
              << " microseconds\n";
}

}  // namespace
