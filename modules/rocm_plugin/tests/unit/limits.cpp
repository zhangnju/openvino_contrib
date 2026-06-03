// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cancellation_token.hpp>
#include <cmath>
#include <rocm/device_pointers.hpp>
#include "rocm/math.hpp"
#include <rocm/runtime.hpp>
#include <rocm_config.hpp>
#include <rocm_creation_context.hpp>
#include <rocm_graph_context.hpp>
#include <rocm_inference_request_context.hpp>
#include <rocm_operation_base.hpp>
#include <rocm_operation_registry.hpp>
#include <rocm_simple_execution_delegator.hpp>
#include <rocm_thread_context.hpp>
#include <limits>
#include <memory_manager/rocm_workbuffers.hpp>
#include <memory_manager/tensor_types.hpp>
#include <type_traits>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/parameter.hpp"

namespace {

struct LimitsTest : testing::Test {};

TEST_F(LimitsTest, Limits_Equal) {
    ASSERT_EQ(std::numeric_limits<int64_t>::max(), rocm::math::limit_max<int64_t>());
    ASSERT_EQ(std::numeric_limits<int64_t>::min(), rocm::math::limit_min<int64_t>());

    ASSERT_EQ(std::numeric_limits<int32_t>::max(), rocm::math::limit_max<int32_t>());
    ASSERT_EQ(std::numeric_limits<int32_t>::min(), rocm::math::limit_min<int32_t>());

    ASSERT_EQ(std::numeric_limits<int16_t>::max(), rocm::math::limit_max<int16_t>());
    ASSERT_EQ(std::numeric_limits<int16_t>::min(), rocm::math::limit_min<int16_t>());

    ASSERT_EQ(std::numeric_limits<int8_t>::max(), rocm::math::limit_max<int8_t>());
    ASSERT_EQ(std::numeric_limits<int8_t>::min(), rocm::math::limit_min<int8_t>());

    ASSERT_EQ(std::numeric_limits<uint64_t>::max(), rocm::math::limit_max<uint64_t>());
    ASSERT_EQ(std::numeric_limits<uint64_t>::min(), rocm::math::limit_min<uint64_t>());

    ASSERT_EQ(std::numeric_limits<uint32_t>::max(), rocm::math::limit_max<uint32_t>());
    ASSERT_EQ(std::numeric_limits<uint32_t>::min(), rocm::math::limit_min<uint32_t>());

    ASSERT_EQ(std::numeric_limits<uint16_t>::max(), rocm::math::limit_max<uint16_t>());
    ASSERT_EQ(std::numeric_limits<uint16_t>::min(), rocm::math::limit_min<uint16_t>());

    ASSERT_EQ(std::numeric_limits<uint8_t>::max(), rocm::math::limit_max<uint8_t>());
    ASSERT_EQ(std::numeric_limits<uint8_t>::min(), rocm::math::limit_min<uint8_t>());
}

template <typename T, typename U>
T finite_cast(U val) {
    if constexpr (std::is_integral_v<T>) {
        if (std::isinf(val)) {
            return val > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        } else if (std::isnan(val)) {
            return std::numeric_limits<T>::min();
        }
    }
    return static_cast<T>(val);
}

template <typename T>
bool is_inf(T val) {
    if constexpr (std::is_integral_v<T>) {
        return false;
    } else {
        return std::isinf(val);
    }
}

template <typename T>
bool is_nan(T val) {
    if constexpr (std::is_integral_v<T>) {
        return false;
    } else {
        return std::isnan(val);
    }
}

template <typename T>
void run_zero_div_test() {
    using devptr_t = rocm::DevicePointer<void*>;
    using cdevptr_t = rocm::DevicePointer<const void*>;

    static constexpr int length = 3;
    static constexpr size_t size_bytes = length * sizeof(T);

    ov::rocm_gpu::ThreadContext threadContext{{}};
    rocm::Allocation in1_alloc = threadContext.stream().malloc(size_bytes);
    rocm::Allocation in2_alloc = threadContext.stream().malloc(sizeof(T));
    rocm::Allocation out_alloc = threadContext.stream().malloc(size_bytes);
    std::vector<cdevptr_t> inputs{in1_alloc, in2_alloc};
    std::vector<devptr_t> outputs{out_alloc};

    ov::rocm_gpu::OperationBase::Ptr operation = [&] {
        rocm::Device device{};
        const bool optimizeOption = false;
        const ov::element::Type ng_type = ov::element::from<T>();
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ng_type, ov::PartialShape{length});
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ng_type, ov::PartialShape{1});
        auto node = std::make_shared<ov::op::v1::Divide>(param1->output(0), param2->output(0));
        auto& registry = ov::rocm_gpu::OperationRegistry::getInstance();
        auto op = registry.createOperation(ov::rocm_gpu::CreationContext{device, optimizeOption},
                                           node,
                                           std::vector<ov::rocm_gpu::TensorID>{ov::rocm_gpu::TensorID{0u}},
                                           std::vector<ov::rocm_gpu::TensorID>{ov::rocm_gpu::TensorID{0u}});
        return op;
    }();
    ASSERT_TRUE(operation);

    std::vector<rocm::DefaultAllocation> im_buffers;

    ov::rocm_gpu::WorkbufferRequest wb_request{operation->GetWorkBufferRequest()};
    for (const auto size : wb_request.immutable_sizes) {
        im_buffers.emplace_back(rocm::DefaultStream::stream().malloc(size));
    }

    ov::rocm_gpu::IOperationExec::Buffers init_buffers;
    for (const auto& buf : im_buffers) {
        init_buffers.emplace_back(buf);
    }
    operation->InitSharedImmutableWorkbuffers(init_buffers);

    ov::rocm_gpu::Workbuffers workbuffers{};
    for (const auto& buf : im_buffers) {
        workbuffers.immutable_buffers.emplace_back(buf);
    }

    const std::array<T, length> in1{static_cast<T>(-1), 0, 1};
    const std::array<T, 1> in2{0};
    std::array<T, length> correct;
    for (std::size_t i = 0; i < length; ++i) {
        correct[i] = finite_cast<T>(static_cast<float>(in1[i]) / static_cast<float>(in2[0]));
    }

    std::vector<std::shared_ptr<ov::Tensor>> emptyTensor;
    std::map<std::string, std::size_t> emptyMapping;
    ov::rocm_gpu::CancellationToken token{};
    ov::rocm_gpu::SimpleExecutionDelegator simpleExecutionDelegator{};
    ov::rocm_gpu::rocmGraphContext rocmGraphContext;
    ov::rocm_gpu::InferenceRequestContext context{emptyTensor,
                                                    emptyMapping,
                                                    emptyTensor,
                                                    emptyMapping,
                                                    threadContext,
                                                    token,
                                                    simpleExecutionDelegator,
                                                    rocmGraphContext};
    auto& stream = context.getThreadContext().stream();
    stream.upload(in1_alloc, in1.data(), size_bytes);
    stream.upload(in2_alloc, in2.data(), sizeof(T));

    operation->Execute(context, inputs, outputs, workbuffers);

    std::array<T, length> out;
    out.fill(0);
    stream.download(out.data(), outputs[0], size_bytes);
    stream.synchronize();

    for (std::size_t i = 0; i < out.size(); i++) {
        if (is_inf(correct[i])) {
            EXPECT_TRUE(is_inf(out[i]));
            EXPECT_EQ(correct[i] > 0, out[i] > 0);
        } else if (is_nan(correct[i])) {
            EXPECT_TRUE(is_nan(out[i]));
        } else {
            EXPECT_EQ(out[i], correct[i]) << "at i == " << i;
        }
    }
}

struct ZeroDivTest : testing::Test {};

TEST_F(ZeroDivTest, canExecuteSync) {
    run_zero_div_test<float>();
    run_zero_div_test<ov::float16>();
    run_zero_div_test<int32_t>();
    run_zero_div_test<int16_t>();
    run_zero_div_test<uint8_t>();
}

}  // namespace
