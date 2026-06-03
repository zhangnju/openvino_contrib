// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <rocm/runtime.hpp>
#include <rocm/stl/atomic.hpp>
#include <rocm/stl/span.hpp>

using namespace ov::rocm_gpu;

class SpanTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

namespace {

template <typename T>
__global__ void verify_extents(rocm::Span<T> span) {
    assert(span.size() == 101);
    assert(blockDim.x == 101);
}

template <typename T>
__global__ void assign(rocm::Span<T> span) {
    assert(span.size() == 101);
    assert(blockDim.x == 101);
    const size_t x = threadIdx.x;
    span[x] = x;
}

template <typename T>
__global__ void verify(rocm::Span<T> span) {
    assert(span.size() == 101);
    assert(blockDim.x == 101);
    const size_t x = threadIdx.x;
    assert(span[x] == x);
    assert(*(span.data() + x) == x);
}

}  // namespace

TEST_F(SpanTest, Span_VerifyExtents) {
    using SpanTestType = rocm::Span<int>;

    rocm::Stream stream{};
    auto src = stream.malloc(SpanTestType::size_of(101));
    auto span = SpanTestType(src.get(), 101);
    verify_extents<<<1, 101, 0, stream.get()>>>(span);
    ASSERT_NO_THROW(stream.synchronize());
}

TEST_F(SpanTest, Span_Verify) {
    using SpanTestType = rocm::Span<int>;

    rocm::Stream stream{};
    auto src = stream.malloc(SpanTestType::size_of(101));
    auto span = SpanTestType(src.get(), 101);
    assign<<<1, 101, 0, stream.get()>>>(span);
    verify<<<1, 101, 0, stream.get()>>>(span);
    ASSERT_NO_THROW(stream.synchronize());
}
