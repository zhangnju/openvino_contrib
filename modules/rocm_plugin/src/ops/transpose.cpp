// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose.hpp"

#include <fmt/format.h>
#include <algorithm>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <rocm_operation_registry.hpp>

#include "kernels/transpose.hpp"

using namespace std::string_literals;

namespace ov {
namespace rocm_gpu {

TransposeOp::TransposeOp(const CreationContext& context,
                         const std::shared_ptr<ov::Node>& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto input_shape = node->get_input_shape(0);
    ndim_ = static_cast<int32_t>(input_shape.size());
    for (auto d : input_shape) src_shape_.push_back(static_cast<int64_t>(d));
    input_type_ = node->get_input_element_type(0);
    perm_type_ = (node->get_input_size() > 1) ? node->get_input_element_type(1) : ov::element::i32;

    size_t size = node->get_input_element_type(0).size();
    if (size == 0)
        throw_ov_exception(fmt::format("TransposeOp: unsupported element type {}", ov::element::Type{input_type_}.get_type_name()));
    element_size_ = size;

    perm_ = tryToExtractPermutation(*node, ndim_);

    // Compute total elements for fixed-perm case
    total_elements_ = 1;
    for (auto d : src_shape_) total_elements_ *= d;

    // Pre-allocate device stride/perm buffers to avoid hipMalloc at Execute time.
    // Only possible when perm is statically known (constant input or identity).
    if (perm_.has_value()) {
        device_buf_ = kernel::allocTransposeDeviceBuffers(src_shape_.data(), perm_->data(), ndim_);
    }

}

TransposeOp::~TransposeOp() {
    kernel::freeTransposeDeviceBuffers(device_buf_);
}

bool TransposeOp::isPermutationTensorSpecified(const ov::Node& node) {
    return node.get_input_size() == 2;
}

std::optional<std::vector<int32_t>> TransposeOp::tryToExtractPermutation(const ov::Node& node, int32_t ndim) {
    if (isPermutationTensorSpecified(node)) {
        auto src = node.input(1).get_source_output().get_node();
        if (ov::is_type<const ov::op::v0::Constant>(src)) {
            auto c = dynamic_cast<const ov::op::v0::Constant*>(src);
            // Safe: permutation may be i32 or i64
            const auto& _et = c->get_element_type();
            if (_et == ov::element::i64) {
                std::vector<int32_t> v;
                for (auto x : c->cast_vector<int64_t>()) v.push_back(static_cast<int32_t>(x));
                return v;
            }
            return c->cast_vector<int32_t>();
        }
        return std::nullopt;
    }
    std::vector<int32_t> rev(ndim);
    for (int32_t i = 0; i < ndim; ++i) rev[i] = ndim - 1 - i;
    return rev;
}

template <typename T>
std::vector<int32_t> TransposeOp::downloadPermutationVector(const InferenceRequestContext& context,
                                                             rocm::DevicePointer<const void*> ptr) const {
    std::vector<T> tmp(ndim_);
    context.getThreadContext().stream().download(tmp.data(), ptr, ndim_ * sizeof(T));
    context.getThreadContext().stream().synchronize();
    std::vector<int32_t> result(ndim_);
    for (int32_t i = 0; i < ndim_; ++i) result[i] = static_cast<int32_t>(tmp[i]);
    return result;
}

void TransposeOp::Execute(const InferenceRequestContext& context,
                          Inputs inputTensors,
                          Outputs outputTensors,
                          const Workbuffers&) const {
    OPENVINO_ASSERT(inputTensors.size() >= 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());

    std::vector<int32_t> perm;
    if (perm_.has_value()) {
        perm = perm_.value();
    } else {
        OPENVINO_ASSERT(inputTensors.size() == 2, "Node name: ", GetName());
        using ov::element::Type_t;
        switch (perm_type_) {
            case Type_t::i8:  perm = downloadPermutationVector<int8_t>(context, inputTensors[1]); break;
            case Type_t::i16: perm = downloadPermutationVector<int16_t>(context, inputTensors[1]); break;
            case Type_t::i32: perm = downloadPermutationVector<int32_t>(context, inputTensors[1]); break;
            case Type_t::i64: perm = downloadPermutationVector<int64_t>(context, inputTensors[1]); break;
            case Type_t::u8:  perm = downloadPermutationVector<uint8_t>(context, inputTensors[1]); break;
            case Type_t::u16: perm = downloadPermutationVector<uint16_t>(context, inputTensors[1]); break;
            case Type_t::u32: perm = downloadPermutationVector<uint32_t>(context, inputTensors[1]); break;
            case Type_t::u64: perm = downloadPermutationVector<uint64_t>(context, inputTensors[1]); break;
            default:
                throw_ov_exception("TransposeOp: permutation vector must be integer type.");
        }
    }

    if (device_buf_) {
        // Fast path: use pre-allocated device strides (hipGraph-safe, no hipMalloc).
        kernel::launchTranspose(inputTensors[0].get(),
                                outputTensors[0].get(),
                                device_buf_,
                                total_elements_,
                                ndim_,
                                element_size_,
                                context.getThreadContext().stream().get());
    } else {
        // Slow path: dynamic perm from input tensor (not hipGraph-compatible).
        // Allocate device buffers on the fly for this case.
        int64_t src_strides[8], dst_strides[8];
        int64_t dst_shape[8];
        for (int i = 0; i < ndim_; ++i) dst_shape[i] = src_shape_[perm[i]];
        src_strides[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; --i) src_strides[i] = src_strides[i + 1] * src_shape_[i + 1];
        dst_strides[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; --i) dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];
        int64_t total = 1;
        for (int i = 0; i < ndim_; ++i) total *= dst_shape[i];
        void* tmp_buf = kernel::allocTransposeDeviceBuffers(src_shape_.data(), perm.data(), ndim_);
        kernel::launchTranspose(inputTensors[0].get(),
                                outputTensors[0].get(),
                                tmp_buf,
                                total,
                                ndim_,
                                element_size_,
                                context.getThreadContext().stream().get());
        context.getThreadContext().stream().synchronize();
        kernel::freeTransposeDeviceBuffers(tmp_buf);
    }
}

rocmGraphCompatibility TransposeOp::GetrocmGraphCompatibility() const {
    return rocmGraphCompatibility::FULL;
}

OPERATION_REGISTER(TransposeOp, Transpose);

}  // namespace rocm_gpu
}  // namespace ov
