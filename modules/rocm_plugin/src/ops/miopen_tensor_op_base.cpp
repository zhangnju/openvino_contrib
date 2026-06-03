// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "miopen_tensor_op_base.hpp"

#include <fmt/ostream.h>

#include <rocm_operation_registry.hpp>
#include <openvino/op/util/attr_types.hpp>

#include "converters.hpp"
#include "rocm/constant_factory.hpp"

namespace ov {
namespace rocm_gpu {
namespace {

bool argTypesSupported(miopenDataType_t in0, miopenDataType_t in1, miopenDataType_t out) {
    if (in0 != in1) return false;
    switch (out) {
        case miopenFloat:
            switch (in0) {
                case miopenFloat:
                case miopenInt8:
                case miopenHalf:
                case miopenBFloat16:
                    return true;
                default:
                    return false;
            }
        case miopenDouble:
            return (in0 == out);
        case miopenHalf:
            return (in0 == out) || (in0 == miopenFloat);
        case miopenInt8:
            return (in0 == out) || (in0 == miopenFloat);
        case miopenBFloat16:
            return (in0 == out) || (in0 == miopenFloat);
        default:
            return false;
    }
}

template <typename T, std::size_t N>
std::array<T, N> toArray(const ov::Shape& shape) {
    std::array<T, N> a;
    a.fill(static_cast<T>(1));
    if (shape.empty()) return a;
    std::copy(shape.rbegin(), shape.rend(), a.rbegin());
    return a;
}

using ShapeArray = std::array<int, miopenTensorOpBase::max_supported_shape_size>;

rocm::DnnTensorDescriptor desc(const miopenDataType_t type, ShapeArray& dims) {
    ShapeArray strides;
    strides.back() = 1;
    for (int i = dims.size() - 1; i > 0; i--) strides[i - 1] = strides[i] * dims[i];
    return rocm::DnnTensorDescriptor{}.set(type, static_cast<int>(dims.size()), dims.data(), strides.data());
}
}  // namespace

miopenTensorOpBase::miopenTensorOpBase(const CreationContext& context,
                                     const std::shared_ptr<ov::Node>& node,
                                     IndexCollection&& inputIds,
                                     IndexCollection&& outputIds,
                                     const miopenTensorOp_t& opType,
                                     const miopenNanPropagation_t& nanPropogationType)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      in0(*node, IoParams::Type::INPUT, 0),
      in1(*node, IoParams::Type::INPUT, 1),
      out(*node, IoParams::Type::OUTPUT, 0),
      // According to https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenOpTensor
      // opTensorCompType for miopenOpTensor should always be FLOAT unless inputs and outputs are
      // double which is not supported by rocmPlugin
      //op_desc_{makeDnnOpTensorDescriptor(opType, miopenDataType_t::miopen_DATA_FLOAT, nanPropogationType)},
      op_type_(opType) {
    OPENVINO_ASSERT(node->get_input_size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(node->get_output_size() == 1, "Node name: ", GetName());
    if (!argTypesSupported(in0.type_, in1.type_, out.type_)) {
        // See https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenOpTensor for
        // supported argument types.
        throw_ov_exception(fmt::format(
            "miopenTensorOpBase: unsupported argument types: ({},{}) -> {}", in0.type_, in1.type_, out.type_));
    }
    const auto& in_partial_shape0 = node->get_input_partial_shape(0);
    const auto& in_partial_shape1 = node->get_input_partial_shape(1);
    const auto& out_partial_shape = node->get_output_partial_shape(0);
    if (in0.shape_.size() > max_supported_shape_size || in1.shape_.size() > max_supported_shape_size) {
        throw_ov_exception(
            fmt::format("Currently max supported shape size for miopenTensorOpBase operation "
                        "is: {} {} {}",
                        max_supported_shape_size,
                        in_partial_shape0,
                        in_partial_shape1));
    }
    if (out.shape_ != in0.shape_ && out.shape_ != in1.shape_) {
        throw_ov_exception(
            fmt::format("Currently at least one of the input shapes: {}, {} of "
                        "miopenTensorOpBase operation should be"
                        "equal to the output shape: {}",
                        in_partial_shape0,
                        in_partial_shape1,
                        out_partial_shape));
    }
    const auto size = in0.array_.size();
    OPENVINO_ASSERT(in1.array_.size() == size, "Node name: ", GetName());
    OPENVINO_ASSERT(out.array_.size() == size, "Node name: ", GetName());
    bool has_0_broadcasts = false;
    bool has_1_broadcasts = false;
    for (int i = 0; i < size; ++i) {
        if (in0.array_[i] != in1.array_[i]) {
            if (in0.array_[i] == 1) {
                has_0_broadcasts = true;
            } else if (in1.array_[i] == 1) {
                has_1_broadcasts = true;
            } else {
                throw_ov_exception(fmt::format(
                    "Unsupported shapes for miopenTensorOpBase operation: {} {}", in_partial_shape0, in_partial_shape1));
            }
        }
    }
    bias_index_ = 0;
    dest_index_ = 1;
    if (has_0_broadcasts || has_1_broadcasts) {
        auto broadcast_spec = node->get_autob();
        if (!(broadcast_spec == ov::op::AutoBroadcastType::NUMPY)) {
            throw_ov_exception(
                fmt::format("Unsupported broadcast type for miopenTensorOpBase operation: {}", broadcast_spec.m_type));
        }
        if (has_0_broadcasts && has_1_broadcasts) {
            throw_ov_exception(
                fmt::format("Currently miopenTensorOpBase operation supports "
                            "broadcasting only in one "
                            "of two input shapes: {} {}",
                            in_partial_shape0,
                            in_partial_shape1));
        }
        bias_index_ = has_0_broadcasts ? 0 : 1;
        dest_index_ = has_0_broadcasts ? 1 : 0;
    }
}

void miopenTensorOpBase::Execute(const InferenceRequestContext& context,
                                Inputs inputTensors,
                                Outputs outputTensors,
                                const Workbuffers&) const {
    OPENVINO_ASSERT(inputTensors.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());
    const auto& bias_input = bias_index_ == 0 ? in0 : in1;
    const auto& dest_input = bias_index_ == 0 ? in1 : in0;

    const void* alpha1 = &rocm::NumericConst<rocm::constants::one>(out.type_);
    const void* alpha2 = &rocm::NumericConst<rocm::constants::one>(out.type_);
    const void* beta = &rocm::NumericConst<rocm::constants::zero>(out.type_);

    context.getThreadContext().dnnHandle().opTensor(op_type_,
                                                    alpha1,
                                                    dest_input.desc_,
                                                    inputTensors[dest_index_].get(),
                                                    alpha2,
                                                    bias_input.desc_,
                                                    inputTensors[bias_index_].get(),
                                                    beta,
                                                    out.desc_,
                                                    outputTensors[0].get());
}

rocmGraphCompatibility miopenTensorOpBase::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

miopenTensorOpBase::IoParams::IoParams(const ov::Node& node, const Type& io_type, int index)
    : type_(convertDataType<miopenDataType_t>(io_type == Type::INPUT ? node.get_input_element_type(index)
                                                                    : node.get_output_element_type(index))),
      shape_(io_type == Type::INPUT ? node.get_input_shape(index) : node.get_output_shape(index)),
      array_(toArray<int, max_supported_shape_size>(shape_)),
      desc_(desc(type_, array_)) {}

}  // namespace rocm_gpu
}  // namespace ov
