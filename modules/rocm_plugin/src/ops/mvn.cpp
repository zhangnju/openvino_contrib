// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"

#include <rocm/descriptor_utils.hpp>
#include <rocm_operation_registry.hpp>

#include "converters.hpp"

namespace ov {
namespace rocm_gpu {

inline bool isTypeSupported(miopenDataType_t type) {
    switch (type) {
        case miopenFloat:
        case miopenDouble:
        case miopenHalf:
        case miopenInt8:
            return true;
        default:
            return false;
    }
}

MvnOp::MvnOp(const CreationContext& context,
             const ov::Node& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      mvn_op_v1_{dynamic_cast<const ov::op::v0::MVN*>(&node)},
      mvn_op_v6_{dynamic_cast<const ov::op::v6::MVN*>(&node)},
      version_{validateAndGetVersion(node)},
      normalize_variance_{version_ == MvnV1 ? mvn_op_v1_->get_normalize_variance()
                                            : mvn_op_v6_->get_normalize_variance()},
      epsilon_{version_ == MvnV1 ? mvn_op_v1_->get_eps() : mvn_op_v6_->get_eps()},
      eps_mode_{version_ == MvnV1 ? ov::op::MVNEpsMode::INSIDE_SQRT : mvn_op_v6_->get_eps_mode()},
      comp_type_{ov::rocm_gpu::convertDataType<miopenDataType_t>(node.get_input_element_type(0))},
      op_desc_type_{comp_type_ != miopenDouble ? miopenFloat : miopenDouble},
      reduce_mean_desc_{op_desc_type_},
      /*
      sub_desc_(rocm::DnnOpTensorDescriptor{}.set(
          miopenTensorOp_t::miopenTensorOpAdd, op_desc_type_, miopenNanPropagation_t::MIOPEN_PROPAGATE_NAN)),
      mul_desc_(rocm::DnnOpTensorDescriptor{}.set(
          miopenTensorOp_t::miopenTensorOpMul, op_desc_type_, miopenNanPropagation_t::MIOPEN_PROPAGATE_NAN)),
      */
      //fix me , how to add operator desc 
      tensor_desc_{rocm::makeInputDnnTensorDescr(node, 0)},
      shape_{node.get_input_shape(0)},
      reduced_shape_{makeReducedShape(node)},
      reduced_tensor_desc_{makeReducedTensorDescriptor(node)},
      reduce_workspace_size_{reduceWorkSpaceSizeCompute(context)} {
    if (!isTypeSupported(op_desc_type_)) {
        throw_ov_exception(fmt::format("MvnOp: unsupported argument type: {}", toString(op_desc_type_)));
    }
    if (!reduced_shape_.empty()) {
        size_t size = shape_size(reduced_shape_);
        unsigned max_threads_per_block = context.device().props().maxThreadsPerBlock;
        unsigned blocks_number = 1 + size / max_threads_per_block;
        unsigned threads_per_block = (blocks_number == 1) ? size : max_threads_per_block;
        variance_normalization_factor_kernel_ = kernel::VarianceNormalizationFactor(
            blocks_number,
            threads_per_block,
            epsilon_,
            size,
            convertDataType<ov::rocm_gpu::kernel::Type_t>(node.get_input_element_type(0)),
            eps_mode_ == ov::op::MVNEpsMode::INSIDE_SQRT);
    }
}

void MvnOp::Execute(const InferenceRequestContext& context,
                    Inputs inputTensors,
                    Outputs outputTensors,
                    const Workbuffers& workbuffers) const {
    if (reduced_shape_.empty()) {
        // Edge case: no axes to reduce → output = x - x = 0. Keep legacy path.
        Context opContext{context, workbuffers, *this};
        opContext.subtract(
            {tensor_desc_, inputTensors[0]}, {tensor_desc_, inputTensors[0]}, {tensor_desc_, outputTensors[0]});
        return;
    }

    // Use miopenLayerNormForward for the full MVN computation in a single call.
    // This replaces: reduceMean → subtract → [multiply(sq) → reduceMean → varNormFactor → multiply]
    // miopenLayerNormForward computes: y = (x - mean(x)) / sqrt(var(x) + eps)  [when normalize_variance_]
    //                               or: y = x - mean(x)                         [when !normalize_variance_]
    // weight=nullptr, bias=nullptr → MIOPEN_ELEMENTWISE_AFFINE (no scale/bias, matches MVN-1/MVN-6).
    auto& handle = context.getThreadContext().dnnHandle();

    // mean and rstd buffers (required by API even if we don't use them downstream)
    auto mean_buf  = workbuffers.mutable_buffers[1];
    auto rstd_buf  = workbuffers.mutable_buffers[2];

    // normalized_dim: index of the first axis being normalized.
    // The reduced_shape_ has 1s at reduced axes; normalized_dim = first reduced axis.
    int32_t normalized_dim = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (reduced_shape_[i] == 1 && shape_[i] > 1) {
            normalized_dim = static_cast<int32_t>(i);
            break;
        }
    }

    auto status = miopenLayerNormForward(
        handle.get(),
        normalize_variance_ ? MIOPEN_ELEMENTWISE_AFFINE : MIOPEN_ELEMENTWISE_AFFINE,
        tensor_desc_.get(), inputTensors[0].get(),       // x
        nullptr, nullptr,                                 // weight (none)
        nullptr, nullptr,                                 // bias (none)
        static_cast<float>(epsilon_),
        normalized_dim,
        tensor_desc_.get(), outputTensors[0].get(),      // y
        reduced_tensor_desc_.get(), mean_buf.get(),      // mean (temp)
        reduced_tensor_desc_.get(), rstd_buf.get());     // rstd (temp)

    if (status != miopenStatusSuccess) {
        // Fallback to multi-step path on error
        Context opContext{context, workbuffers, *this};
        auto reducedTensor = workbuffers.mutable_buffers[1];
        opContext.reduceMean({tensor_desc_, inputTensors[0]}, {reduced_tensor_desc_, reducedTensor});
        opContext.subtract({tensor_desc_, inputTensors[0]},
                           {reduced_tensor_desc_, reducedTensor.cast<const void*>()},
                           {tensor_desc_, outputTensors[0]});
        if (!normalize_variance_) return;
        auto tmpTensor = workbuffers.mutable_buffers[2];
        opContext.multiply({tensor_desc_, outputTensors[0].cast<const void*>()},
                           {tensor_desc_, outputTensors[0].cast<const void*>()},
                           {tensor_desc_, tmpTensor});
        opContext.reduceMean({tensor_desc_, tmpTensor.cast<const void*>()}, {reduced_tensor_desc_, reducedTensor});
        opContext.computeVarianceNormalizationFactor({reduced_tensor_desc_, reducedTensor});
        opContext.multiply({tensor_desc_, outputTensors[0].cast<const void*>()},
                           {reduced_tensor_desc_, reducedTensor.cast<const void*>()},
                           {tensor_desc_, outputTensors[0]});
    }
}

rocmGraphCompatibility MvnOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

void MvnOp::Context::reduceMean(ConstTensor input, Tensor output) {
    context.getThreadContext().dnnHandle().reduceTensor(op.reduce_mean_desc_,
                                                        op.getReduceWorkspaceBuffer(workbuffers),
                                                        rocm::DnnScaleFactorOne{op.comp_type_},
                                                        input.descriptor,
                                                        input.data,
                                                        rocm::DnnScaleFactorZero{op.comp_type_},
                                                        output.descriptor,
                                                        output.data);
}

void MvnOp::Context::subtract(ConstTensor lhs, ConstTensor rhs, Tensor output) {
    // C = 1*A + (-1)*B + 0*C  →  C = A - B
    context.getThreadContext().dnnHandle().opTensor(
        miopenTensorOpAdd,
        op.dOne,
        lhs.descriptor,
        lhs.data.get(),
        op.dMinusOne,
        rhs.descriptor,
        rhs.data.get(),
        op.dZero,
        output.descriptor,
        output.data.get());
}

void MvnOp::Context::multiply(ConstTensor lhs, ConstTensor rhs, Tensor output) {
    // C = 1*A * 1*B + 0*C  →  C = A * B
    context.getThreadContext().dnnHandle().opTensor(
        miopenTensorOpMul,
        op.dOne,
        lhs.descriptor,
        lhs.data.get(),
        op.dOne,
        rhs.descriptor,
        rhs.data.get(),
        op.dZero,
        output.descriptor,
        output.data.get());
}

void MvnOp::Context::computeVarianceNormalizationFactor(Tensor in_out) {
    OPENVINO_ASSERT(op.variance_normalization_factor_kernel_);
    (*op.variance_normalization_factor_kernel_)(context.getThreadContext().stream().get(), in_out.data.get());
}

MvnOp::MvnVersion MvnOp::validateAndGetVersion(const ov::Node& node) {
    auto mvnOp_v1 = dynamic_cast<const ov::op::v0::MVN*>(&node);
    auto mvnOp_v6 = dynamic_cast<const ov::op::v6::MVN*>(&node);
    MvnVersion version;
    OPENVINO_ASSERT(mvnOp_v1 || mvnOp_v6);
    if (mvnOp_v1) {
        version = MvnV1;
        OPENVINO_ASSERT(node.get_input_size() == 1);
        if (mvnOp_v1->get_eps() <= 0) {
            throw_ov_exception(
                fmt::format("The eps attribute of the MVN-1 operation must be positive number, but value is {}.",
                            mvnOp_v1->get_eps()));
        }
    } else {
        version = MvnV6;
        OPENVINO_ASSERT(node.get_input_size() == 2);
        if (mvnOp_v6->get_eps() <= 0) {
            throw_ov_exception(
                fmt::format("The eps attribute of the MVN-6 operation must be positive number, but value is {}.",
                            mvnOp_v6->get_eps()));
        }
        if (ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(1)) == nullptr) {
            throw_ov_exception("The rocm_gpu MVN-6 operation implemented only for constant axes input.");
        }
    }
    if (!node.get_input_partial_shape(0).rank().is_static()) {
        throw_ov_exception("For not static input shape the MVN-1 operation was not implemented.");
    }
    OPENVINO_ASSERT(node.get_output_size() == 1);
    auto inputShape = node.get_input_shape(0);
    auto outputShape = node.get_output_shape(0);
    OPENVINO_ASSERT(inputShape == outputShape);
    if (version == MvnV6) {
        auto inputAxesShape = node.get_input_shape(1);
        OPENVINO_ASSERT(inputAxesShape.size() == 1);
        OPENVINO_ASSERT(inputAxesShape[0] <= inputShape.size());
    }
    OPENVINO_ASSERT(node.get_input_element_type(0) == node.get_output_element_type(0));
    const size_t max_shape_size = 5;  // miopenOpTensor operation limit. See note here
                                      // https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenOpTensor
    if (outputShape.size() > max_shape_size) {
        throw_ov_exception(
            fmt::format("ov::rocm_gpu::MvnOp: the tensor shape size ({}) is exceeded maximum supported value of {}.",
                        outputShape.size(),
                        max_shape_size));
    }
    return version;
}

size_t MvnOp::reduceWorkSpaceSizeCompute(const CreationContext& context) {
    if (!reduced_shape_.empty())
        return context.dnnHandle().getReductionWorkspaceSize(reduce_mean_desc_, tensor_desc_, reduced_tensor_desc_);
    return 0;
}

ov::Shape MvnOp::makeReducedShape(const ov::Node& node) {
    if (version_ == MvnV1) {
        auto reducedShape = node.get_input_shape(0);
        if (mvn_op_v1_->get_reduction_axes().empty()) {
            return {};
        } else {
            for (auto& reductionAxisIndex : mvn_op_v1_->get_reduction_axes()) {
                OPENVINO_ASSERT(reductionAxisIndex < reducedShape.size(), "Node name: ", GetName());
                reducedShape[reductionAxisIndex] = 1;
            }
        }
        return reducedShape;
    }
    if (version_ == MvnV6) {
        // Safely read axes constant - supports int32, int64, or float32 element types.
        // ONNX ReduceMean axes may be stored as i32 or f32 in some models.
        auto axes_const = ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(1));
        std::vector<int64_t> signed_axes;
        if (axes_const) {
            const auto& et = axes_const->get_element_type();
            if (et == ov::element::i64) {
                signed_axes = axes_const->cast_vector<int64_t>();
            } else if (et == ov::element::i32) {
                for (auto v : axes_const->cast_vector<int32_t>())
                    signed_axes.push_back(static_cast<int64_t>(v));
            } else {
                for (auto v : axes_const->cast_vector<float>())
                    signed_axes.push_back(static_cast<int64_t>(v));
            }
        }
        auto reducedShape = node.get_input_shape(0);
        ov::AxisSet axes;
        for (auto v : signed_axes) {
            auto size = static_cast<int64_t>(reducedShape.size());
            if (v >= size || v < -size) {
                throw_ov_exception(
                    fmt::format("ov::rocm_gpu::MVN-6: the axes entry ({}) out of range [{}; {}].", v, -size, size - 1));
            }
            axes.emplace(static_cast<size_t>((v + size) % size));
        }
        reducedShape = ov::util::reduce_keep_dims(reducedShape, axes);
        if (reducedShape == node.get_input_shape(0)) return {};
        return reducedShape;
    }
    return {};
}

rocm::DnnTensorDescriptor MvnOp::makeReducedTensorDescriptor(const ov::Node& node) {
    if (reduced_shape_.empty()) return {};
    return rocm::makeDnnTensorDescr(node.get_input_element_type(0), reduced_shape_);
}

OPERATION_REGISTER(MvnOp, MVN);
}  // namespace rocm_gpu
}  // namespace ov
