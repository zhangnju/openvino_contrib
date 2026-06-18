// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "layer_norm.hpp"
#include <rocm_operation_registry.hpp>
#include <rocm/descriptor_utils.hpp>
#include <openvino/core/except.hpp>
#include "converters.hpp"

namespace ov {
namespace rocm_gpu {

LayerNormOp::LayerNormOp(const CreationContext& context,
                         const std::shared_ptr<ov::Node>& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationMIOPEN(context, node, std::move(inputIds), std::move(outputIds)) {
    auto ln = std::dynamic_pointer_cast<nodes::LayerNorm>(node);
    OPENVINO_ASSERT(ln, "LayerNormOp: expected LayerNorm node, got: ", node->get_type_name());

    auto shape = ln->get_input_shape(0);
    int64_t axis = ln->get_axis();
    if (axis < 0) axis += static_cast<int64_t>(shape.size());
    OPENVINO_ASSERT(axis >= 0 && axis < (int64_t)shape.size(),
                    "LayerNormOp: axis out of range, node: ", GetName());

    normalized_dim_ = static_cast<int32_t>(axis);
    epsilon_ = static_cast<float>(ln->get_epsilon());
    has_scale_ = (ln->get_input_size() >= 2);
    has_bias_  = (ln->get_input_size() >= 3);

    // Build tensor descriptors matching miopenLayerNormForward signature
    x_desc_   = rocm::makeInputDnnTensorDescr(*ln, 0);
    y_desc_   = rocm::makeOutputDnnTensorDescr(*ln, 0);

    // mean/rstd have shape with axes-to-normalize set to 1
    auto reduced = shape;
    for (size_t i = (size_t)axis; i < shape.size(); ++i) reduced[i] = 1;
    const auto dtype = convertDataType<miopenDataType_t>(ln->get_input_element_type(0));
    mean_desc_ = rocm::makeDnnTensorDescr(ln->get_input_element_type(0), reduced);
    rstd_desc_ = rocm::makeDnnTensorDescr(ln->get_input_element_type(0), reduced);

    // Scale/bias descriptors (1D, inner dimension)
    if (has_scale_) {
        scale_desc_ = rocm::makeInputDnnTensorDescr(*ln, 1);
    }
    if (has_bias_) {
        bias_desc_  = rocm::makeInputDnnTensorDescr(*ln, 2);
    }

    mean_bytes_ = elementSize(dtype) * ov::shape_size(reduced);
    rstd_bytes_ = mean_bytes_;
}

WorkbufferRequest LayerNormOp::GetWorkBufferRequest() const {
    // Mutable buffers: [0]=mean, [1]=rstd (temporary, not output)
    return {{}, {mean_bytes_, rstd_bytes_}};
}

void LayerNormOp::Execute(const InferenceRequestContext& context,
                          Inputs inputs,
                          Outputs outputs,
                          const Workbuffers& workbuffers) const {
    auto& handle = context.getThreadContext().dnnHandle();

    const void* weight = has_scale_ ? inputs[1].get() : nullptr;
    const void* bias   = has_bias_  ? inputs[2].get() : nullptr;
    void* mean_buf     = workbuffers.mutable_buffers[0].get();
    void* rstd_buf     = workbuffers.mutable_buffers[1].get();

    const miopenNormMode_t mode = (has_scale_ || has_bias_)
        ? MIOPEN_WEIGHT_BIAS : MIOPEN_ELEMENTWISE_AFFINE;

    auto status = miopenLayerNormForward(
        handle.get(),
        mode,
        x_desc_.get(),    inputs[0].get(),
        has_scale_ ? scale_desc_.get() : x_desc_.get(), weight,
        has_bias_  ? bias_desc_.get()  : x_desc_.get(), bias,
        epsilon_,
        normalized_dim_,
        y_desc_.get(),    outputs[0].get(),
        mean_desc_.get(), mean_buf,
        rstd_desc_.get(), rstd_buf);

    OPENVINO_ASSERT(status == miopenStatusSuccess,
        "LayerNormOp: miopenLayerNormForward failed with status ", (int)status,
        ", node: ", GetName());
}

OPERATION_REGISTER(LayerNormOp, LayerNorm);

}  // namespace rocm_gpu
}  // namespace ov
