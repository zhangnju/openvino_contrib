// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "topk.hpp"

#include <rocm_operation_registry.hpp>
#include "converters.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/util/topk_base.hpp"

namespace ov {
namespace rocm_gpu {

namespace {

std::vector<size_t> shapeByAxis(const std::vector<size_t>& shape, const std::size_t axis) {
    std::vector<size_t> newShape{shape};
    newShape.push_back(newShape[axis]);
    newShape.erase(newShape.begin() + axis);
    return newShape;
}

std::vector<size_t> shapeStrides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides;
    size_t stride = 1;
    std::vector<size_t> reversedShape{shape};
    std::reverse(reversedShape.begin(), reversedShape.end());
    for (const auto dim : reversedShape) {
        strides.insert(strides.begin(), stride);
        stride *= dim;
    }
    return strides;
}

std::vector<size_t> shapeStridesByAxis(const std::vector<size_t>& shape, const std::size_t axis) {
    std::vector<size_t> strides = shapeStrides(shape);
    strides.push_back(strides[axis]);
    strides.erase(strides.begin() + axis);
    return strides;
}

template <typename T>
T getK(const ov::op::v0::Constant* k_constant) {
    return *static_cast<const T*>(k_constant->get_data_ptr());
}

size_t getK(const ov::op::v0::Constant* k_constant) {
    switch (k_constant->get_element_type()) {
        case ov::element::Type_t::i8:
            return getK<int8_t>(k_constant);
        case ov::element::Type_t::i16:
            return getK<int16_t>(k_constant);
        case ov::element::Type_t::i32:
            return getK<int32_t>(k_constant);
        case ov::element::Type_t::i64:
            return getK<int64_t>(k_constant);
        case ov::element::Type_t::u8:
            return getK<uint8_t>(k_constant);
        case ov::element::Type_t::u16:
            return getK<uint16_t>(k_constant);
        case ov::element::Type_t::u32:
            return getK<uint32_t>(k_constant);
        case ov::element::Type_t::u64:
            return getK<uint64_t>(k_constant);
        default: {
            throw_ov_exception(
                fmt::format("k element type = {} is not supported by TopK operation "
                            "!!",
                            static_cast<ov::element::Type_t>(k_constant->get_element_type())));
        }
    }
}

}  // namespace

TopKOp::TopKOp(const CreationContext& context,
               const ov::Node& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto topKOp = ov::as_type<const ov::op::util::TopKBase>(&node);
    if (!topKOp) {
        throw_ov_exception(fmt::format("Unsupported TopK version! NodeName: {}", GetName()));
    }

    const ov::element::Type element_type{topKOp->get_input_element_type(0)};
    const ov::element::Type index_element_type{topKOp->get_index_element_type()};
    auto output_element_type = topKOp->get_output_element_type(0);

    OPENVINO_ASSERT(topKOp->get_input_size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(topKOp->get_output_size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(element_type == output_element_type, "Node name: ", GetName());
    const auto& input_shape = topKOp->get_input_shape(0);
    const auto& output_shape = topKOp->get_output_shape(0);
    const uint64_t axis = topKOp->get_axis();
    const size_t num_input_element = ov::shape_size(input_shape);
    const size_t num_output_element = ov::shape_size(output_shape);
    // Workspace for original serial path: N * (sizeof(T) + sizeof(TIdx))
    const size_t serial_ws = num_input_element * (element_type.size() + index_element_type.size());

    // Workspace for fast hipcub radix sort path (batch=1):
    //   Layout: keys_in[N] + keys_out[N] + idx_in[N] + idx_out[N] + hipcub_temp
    // Optimization: skip the low 8 bits of float representation for K/N < 10%.
    // This eliminates exactly 1 radix pass (8 bits/pass), saving ~25% sort time.
    // Both workspace allocation and the kernel use the SAME begin_bit computed here.
    // Workspace for fast hipcub radix sort path (batch=1):
    //   4 * N * 4 bytes (keys_in, keys_out, vals_in, vals_out as float/int)
    //   + 16MB for hipcub temp storage (conservative upper bound for any N and begin_bit).
    // We use a fixed 16MB rather than querying hipcub because the nullptr-arg query
    // can return incorrect values on some ROCm versions, leading to GPU memory faults.
    // The execution-time begin_bit=8 optimization needs ≤ temp storage than begin_bit=0,
    // so 16MB is always sufficient regardless of the chosen begin_bit.
    const size_t radix_ws = 4 * num_input_element * sizeof(float) + 16 * 1024 * 1024;
    workspace_size_ = std::max(serial_ws, radix_ws);

    OPENVINO_ASSERT(axis >= 0 && axis < input_shape.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(axis >= 0 && axis < output_shape.size(), "Node name: ", GetName());
    const std::size_t workspace_chunk_size = input_shape[axis];
    const auto& input_shape_axis = shapeByAxis(input_shape, axis);
    const auto& output_shape_axis = shapeByAxis(output_shape, axis);
    const auto& input_strides = shapeStridesByAxis(input_shape, axis);
    const auto& output_strides = shapeStridesByAxis(output_shape, axis);
    std::copy(input_shape_axis.begin(), input_shape_axis.end(), kernel_param_.input_shape_axis);
    std::copy(output_shape_axis.begin(), output_shape_axis.end(), kernel_param_.output_shape_axis);
    std::copy(input_strides.begin(), input_strides.end(), kernel_param_.input_strides);
    std::copy(output_strides.begin(), output_strides.end(), kernel_param_.output_strides);

    auto convertComputeType = [](const ov::op::TopKMode& compute_mode) {
        switch (compute_mode) {
            case ov::op::TopKMode::MAX:
                return kernel::TopK::ComputeType::Max;
            case ov::op::TopKMode::MIN:
                return kernel::TopK::ComputeType::Min;
        }
        throw_ov_exception(fmt::format("Unknown compute_mode {}", compute_mode));
    };
    auto convertSortType = [](const ov::op::TopKSortType& sort_type) {
        switch (sort_type) {
            case ov::op::TopKSortType::NONE:
                return kernel::TopK::SortType::None;
            case ov::op::TopKSortType::SORT_INDICES:
                return kernel::TopK::SortType::SortIndices;
            case ov::op::TopKSortType::SORT_VALUES:
                return kernel::TopK::SortType::SortValues;
        }
        throw_ov_exception(fmt::format("Unknown sort_type {}", sort_type));
    };

    const auto k_constant = dynamic_cast<ov::op::v0::Constant*>(topKOp->get_input_node_ptr(1));
    OPENVINO_ASSERT(k_constant, "Node name: ", GetName());
    const size_t k = getK(k_constant);

    const auto& prop = context.device().props();
    const std::size_t max_threads_per_block = prop.maxThreadsPerBlock;
    kernel_ = kernel::TopK{convertDataType<ov::rocm_gpu::kernel::Type_t>(element_type),
                           convertDataType<ov::rocm_gpu::kernel::Type_t>(index_element_type),
                           convertComputeType(topKOp->get_mode()),
                           convertSortType(topKOp->get_sort_type()),
                           num_input_element,
                           num_output_element,
                           k,
                           workspace_chunk_size,
                           max_threads_per_block};
}

void TopKOp::Execute(const InferenceRequestContext& context,
                     Inputs inputs,
                     Outputs outputs,
                     const Workbuffers& buffers) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(buffers.mutable_buffers.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(buffers.immutable_buffers.size() == 1, "Node name: ", GetName());
    auto& threadContext = context.getThreadContext();
    auto& stream = threadContext.stream();
    auto kernel_param = buffers.immutable_buffers[0];
    auto workspace = buffers.mutable_buffers[0];
    auto in_tensor = inputs[0];
    auto out_value_tensor = outputs[0];
    auto out_index_tensor = outputs[1];
    (*kernel_)(stream.get(),
               static_cast<const void*>(in_tensor.get()),
               static_cast<void*>(out_value_tensor.get()),
               static_cast<void*>(out_index_tensor.get()),
               static_cast<void*>(workspace.get()),
               static_cast<const void*>(kernel_param.get()));
}

rocmGraphCompatibility TopKOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::NONE; }

void TopKOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    OPENVINO_ASSERT(buffers.size() == 1, "Node name: ", GetName());
    size_t buffer_offset = 0;
    rocm::DefaultStream::stream().upload(
        buffers[0] + buffer_offset, kernel_param_.input_shape_axis, sizeof(kernel_param_.input_shape_axis));
    buffer_offset += sizeof(kernel_param_.input_shape_axis);
    rocm::DefaultStream::stream().upload(
        buffers[0] + buffer_offset, kernel_param_.output_shape_axis, sizeof(kernel_param_.output_shape_axis));
    buffer_offset += sizeof(kernel_param_.output_shape_axis);
    rocm::DefaultStream::stream().upload(
        buffers[0] + buffer_offset, kernel_param_.input_strides, sizeof(kernel_param_.input_strides));
    buffer_offset += sizeof(kernel_param_.input_strides);
    rocm::DefaultStream::stream().upload(
        buffers[0] + buffer_offset, kernel_param_.output_strides, sizeof(kernel_param_.output_strides));
    buffer_offset += sizeof(kernel_param_.output_strides);
}

WorkbufferRequest TopKOp::GetWorkBufferRequest() const { return {{sizeof(kernel_param_)}, {workspace_size_}}; }

OPERATION_REGISTER(TopKOp, TopK);
}  // namespace rocm_gpu
}  // namespace ov
