// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <rocm_config.hpp>
#include <rocm_graph_context.hpp>
#include <rocm_op_buffers_extractor.hpp>
#include <rocm_operation_registry.hpp>
#include <rocm_simple_execution_delegator.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/relu.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>
namespace {

using devptr_t = rocm::DevicePointer<void*>;
using cdevptr_t = rocm::DevicePointer<const void*>;
template <typename F>
auto assertToThrow(F&& f,
                   const std::experimental::source_location& loc = std::experimental::source_location::current()) {
    bool success = false;
    std::forward<F>(f)(success);
    if (!success) ov::rocm_gpu::throw_ov_exception("pathetic google test failed in non-void function", loc);
}

#define TASSERT_TRUE(condition)                \
    assertToThrow([&](bool& tassert_success) { \
        ASSERT_TRUE(condition);                \
        tassert_success = true;                \
    })

struct ReluTest : testing::Test {
    using TensorID = ov::rocm_gpu::TensorID;
    using ElementType = float;
    static constexpr int length = 5;
    static constexpr size_t size = length * sizeof(ElementType);
    ov::rocm_gpu::ThreadContext threadContext{{}};
    rocm::Allocation inAlloc = threadContext.stream().malloc(size);
    rocm::Allocation outAlloc = threadContext.stream().malloc(size);
    std::vector<cdevptr_t> inputs{inAlloc};
    std::vector<devptr_t> outputs{outAlloc};
    std::vector<std::shared_ptr<ov::Tensor>> empty_tensor;
    std::map<std::string, std::size_t> empty_mapping;
    ov::rocm_gpu::OperationBase::Ptr operation = [this] {
        rocm::Device device{};
        const bool optimizeOption = false;
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{length});
        auto node = std::make_shared<ov::op::v0::Relu>(param->output(0));
        auto& registry = ov::rocm_gpu::OperationRegistry::getInstance();
        TASSERT_TRUE(registry.hasOperation(node));
        auto op = registry.createOperation(ov::rocm_gpu::CreationContext{device, optimizeOption},
                                           node,
                                           std::vector<TensorID>{TensorID{0u}},
                                           std::vector<TensorID>{TensorID{0u}});
        TASSERT_TRUE(op);
        return op;
    }();
};

TEST_F(ReluTest, canExecuteSync) {
    ov::rocm_gpu::CancellationToken token{};
    ov::rocm_gpu::SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::rocm_gpu::rocmGraphContext rocmGraphContext{};
    ov::rocm_gpu::InferenceRequestContext context{empty_tensor,
                                                    empty_mapping,
                                                    empty_tensor,
                                                    empty_mapping,
                                                    threadContext,
                                                    token,
                                                    simpleExecutionDelegator,
                                                    rocmGraphContext};
    auto& stream = context.getThreadContext().stream();
    std::array<ElementType, length> in{-1, 1, -5, 5, 0};
    std::array<ElementType, length> correct;
    for (std::size_t i = 0; i < in.size(); i++)
        if (auto c = in[i]; c < 0)
            correct[i] = 0;
        else
            correct[i] = c;
    stream.upload(inAlloc, in.data(), size);
    operation->Execute(context, inputs, outputs, {});
    std::array<ElementType, length> out;
    out.fill(-1);
    stream.download(out.data(), outputs[0], size);
    stream.synchronize();
    for (std::size_t i = 0; i < out.size(); i++) EXPECT_EQ(out[i], correct[i]) << "at i == " << i;
    ASSERT_EQ(std::memcmp(out.data(), correct.data(), sizeof out), 0);
}

}  // namespace
