// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gather_nd.hpp"

#include <fmt/format.h>

#include <numeric>
#include <rocm_operation_registry.hpp>
#include <openvino/op/gather_nd.hpp>

#include "converters.hpp"

namespace ov {
namespace rocm_gpu {

GatherNDOp::GatherNDOp(const CreationContext& context,
                       const ov::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const ov::element::Type_t data_type = node.get_input_element_type(0);
    switch (data_type) {
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::u1:
            throw_ov_exception(fmt::format("Data element type = {} is not supported by GatherND operation!",
                                           static_cast<ov::element::Type_t>(data_type)));
    }
    OPENVINO_ASSERT(node.get_output_element_type(0) == data_type, "Node name: ", GetName());

    const ov::element::Type_t indices_type = node.get_input_element_type(1);
    if (indices_type != ov::element::Type_t::i64 && indices_type != ov::element::Type_t::i32) {
        throw_ov_exception(
            fmt::format("Index element type = {} is not supported by GatherND operation!", indices_type));
    }

    // Only batch_dims == 0 is supported for now (matches all current target models).
    int64_t batch_dims = 0;
    if (const auto g8 = dynamic_cast<const ov::op::v8::GatherND*>(&node)) {
        batch_dims = g8->get_batch_dims();
    } else if (const auto g5 = dynamic_cast<const ov::op::v5::GatherND*>(&node)) {
        batch_dims = g5->get_batch_dims();
    }
    OPENVINO_ASSERT(batch_dims == 0, "Node name: ", GetName(), " GatherND batch_dims != 0 is not supported");

    const auto& data_shape = node.get_input_shape(0);
    const auto& indices_shape = node.get_input_shape(1);
    const auto indices_last_dim = indices_shape.back();
    OPENVINO_ASSERT(indices_last_dim <= data_shape.size(), "Node name: ", GetName());

    // Number of contiguous data elements gathered per index tuple.
    const size_t num_of_gather_elements =
        std::accumulate(data_shape.begin() + indices_last_dim, data_shape.end(), size_t{1}, std::multiplies<size_t>());

    // Number of index tuples (product of all indices dims except the last).
    const size_t num_of_gather_chunks =
        std::accumulate(indices_shape.cbegin(), indices_shape.cend() - 1, size_t{1}, std::multiplies<size_t>());

    const auto max_block_size = context.device().props().maxThreadsPerBlock;

    const bool thread_per_element = num_of_gather_elements > num_of_gather_chunks;
    const size_t num_of_items = thread_per_element ? num_of_gather_elements : num_of_gather_chunks;

    const size_t num_of_blocks{num_of_items % max_block_size == 0 ? num_of_items / max_block_size
                                                                  : num_of_items / max_block_size + 1};

    const size_t num_of_threads{num_of_blocks == 1 ? num_of_items : max_block_size};

    kernel_ = kernel::GatherND{convertDataType<ov::rocm_gpu::kernel::Type_t>(data_type),
                               convertDataType<ov::rocm_gpu::kernel::Type_t>(indices_type),
                               indices_last_dim,
                               num_of_gather_elements,
                               num_of_gather_chunks,
                               num_of_blocks,
                               num_of_threads,
                               thread_per_element};

    data_dim_padding_ = [&] {
        std::vector<size_t> padding(data_shape.size(), 1);
        for (size_t i{data_shape.size() - 1}; i > 0; --i) padding[i - 1] = padding[i] * data_shape[i];
        return padding;
    }();
}

void GatherNDOp::Execute(const InferenceRequestContext& context,
                         Inputs inputs,
                         Outputs outputs,
                         const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    (*kernel_)(context.getThreadContext().stream().get(),
               inputs[0].get(),
               inputs[1].get(),
               static_cast<const size_t*>(workbuffers.immutable_buffers[0].get()),
               outputs[0].get());
}

rocmGraphCompatibility GatherNDOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

template <typename T>
static auto size_in_bytes(const std::vector<T>& v) noexcept {
    return sizeof(T) * v.size();
}

WorkbufferRequest GatherNDOp::GetWorkBufferRequest() const {
    return {{size_in_bytes(data_dim_padding_)}, {}};
}

void GatherNDOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    auto& stream = rocm::DefaultStream::stream();
    stream.upload(buffers[0], data_dim_padding_.data(), size_in_bytes(data_dim_padding_));
}

OPERATION_REGISTER(GatherNDOp, GatherND);

}  // namespace rocm_gpu
}  // namespace ov
