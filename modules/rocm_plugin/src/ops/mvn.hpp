// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

#include "converters.hpp"
#include "kernels/variance_normalization_factor.hpp"
#include "openvino/op/mvn.hpp"

namespace ov {
namespace rocm_gpu {

class MvnOp : public OperationMIOPEN {
public:
    MvnOp(const CreationContext& context,
          const ov::Node& node,
          IndexCollection&& inputIds,
          IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    enum MvnVersion {
        MvnV1,
        MvnV6,
    };

    struct ConstTensor {
        const rocm::DnnTensorDescriptor& descriptor;
        rocm::DevicePointer<const void*> data;
    };
    struct Tensor {
        const rocm::DnnTensorDescriptor& descriptor;
        rocm::DevicePointer<void*> data;
    };
    struct Context {
        const InferenceRequestContext& context;
        const Workbuffers& workbuffers;
        const MvnOp& op;

        void reduceMean(ConstTensor input, Tensor output);
        void subtract(ConstTensor lhs, ConstTensor rhs, Tensor output);
        void multiply(ConstTensor lhs, ConstTensor rhs, Tensor output);
        void computeVarianceNormalizationFactor(Tensor in_out);
    };

    static MvnVersion validateAndGetVersion(const ov::Node& node);
    size_t reduceWorkSpaceSizeCompute(const CreationContext& context);
    ov::Shape makeReducedShape(const ov::Node& node);
    rocm::DnnTensorDescriptor makeReducedTensorDescriptor(const ov::Node& node);
    rocm::DeviceBuffer<std::uint8_t> getReduceWorkspaceBuffer(const Workbuffers& workbuffers) const {
        return workbuffers.createMutableSpanFrom<0>(reduce_workspace_size_);
    }
    rocm::DevicePointer<void*> getReducedTensorBuffer(const Workbuffers& workbuffers) const {
        return workbuffers.mutable_buffers[1];
    }
    rocm::DevicePointer<void*> getTmpTensorBuffer(const Workbuffers& workbuffers) const {
        return workbuffers.mutable_buffers[2];
    }

private:
    const ov::op::v0::MVN* mvn_op_v1_;
    const ov::op::v6::MVN* mvn_op_v6_;
    MvnVersion version_;
    bool normalize_variance_;
    double epsilon_;
    ov::op::MVNEpsMode eps_mode_;
    miopenDataType_t comp_type_;
    miopenDataType_t op_desc_type_;
    rocm::DnnReduceAvgDescriptor reduce_mean_desc_;
    rocm::DnnTensorDescriptor sub_desc_;
    rocm::DnnTensorDescriptor mul_desc_;
    rocm::DnnTensorDescriptor tensor_desc_;
    ov::Shape shape_;
    ov::Shape reduced_shape_;
    rocm::DnnTensorDescriptor reduced_tensor_desc_;
    size_t reduce_workspace_size_;
    std::optional<kernel::VarianceNormalizationFactor> variance_normalization_factor_kernel_;
    const void* dOne{&rocm::NumericConst<rocm::constants::one>(comp_type_)};
    const void* dMinusOne{&rocm::NumericConst<rocm::constants::minusOne>(comp_type_)};
    const void* dZero{&rocm::NumericConst<rocm::constants::zero>(comp_type_)};
};

inline WorkbufferRequest MvnOp::GetWorkBufferRequest() const {
    if (!reduced_shape_.empty()) {
        const size_t reduced_bytes = elementSize(comp_type_) * ov::shape_size(reduced_shape_);
        // Buffers: [0]=reduce_workspace, [1]=mean, [2]=rstd (or tmp)
        // miopenLayerNormForward requires mean+rstd output buffers.
        // Allocate [2] always (rstd for miopenLayerNormForward, tmp for fallback).
        return {{},
                {reduce_workspace_size_,
                 reduced_bytes,
                 reduced_bytes}};
    }
    return {};
}

}  // namespace rocm_gpu
}  // namespace ov
