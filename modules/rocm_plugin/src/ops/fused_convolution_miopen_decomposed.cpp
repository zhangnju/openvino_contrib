// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_miopen_decomposed.hpp"

#include <miopen/miopen.h>

#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#include "rocm/constant_factory.hpp"

namespace ov {
namespace rocm_gpu {

FusedConvolutionmiopenDecomposed::FusedConvolutionmiopenDecomposed(
    const CreationContext& context,
    const ov::Node& node,
    IndexCollection&& inputIds,
    IndexCollection&& outputIds,
    std::shared_ptr<Convolution::Details::ConvolutionDescriptorsmiopen> convDescs,
    std::shared_ptr<rocm::DnnTensorDescriptor> biasDesc,
    std::shared_ptr<rocm::DnnTensorDescriptor> addDesc,
    std::shared_ptr<rocm::DnnActivationDescriptor> activationDesc)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      conv_descs_{convDescs},
      bias_desc_{biasDesc},
      activation_desc_{activationDesc},
      add_desc_{addDesc} {
    ThrowIfShouldNotDecompose();
}

void FusedConvolutionmiopenDecomposed::Execute(const InferenceRequestContext& context,
                                              Inputs inputs,
                                              Outputs outputs,
                                              const Workbuffers& workbuffers) const {
    using ArgIndices = Convolution::Details::FusedConvolutionIndices;

    const bool includesOnlyBiasAdd = inputs.size() == 3;
    const bool includesSecondAddition = inputs.size() == 4;
    OPENVINO_ASSERT((includesOnlyBiasAdd && add_desc_ == nullptr) || (includesSecondAddition && add_desc_),
                    "Node name: ",
                    GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    const auto& dnnHandle = context.getThreadContext().dnnHandle();
    const void* onePtr = &rocm::NumericConst<rocm::constants::one>(conv_descs_->ElementType());
    const void* zeroPtr = &rocm::NumericConst<rocm::constants::zero>(conv_descs_->ElementType());
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    const auto& outDesc = conv_descs_->Output();
    auto outTensor = outputs[ArgIndices::output].get();
 
    throwIfError(::miopenConvolutionForwardImmediate(dnnHandle.get(),
                                                    conv_descs_->Filter().get(),
                                                    inputs[ArgIndices::filter].get(),
                                                    conv_descs_->Input().get(),
                                                    inputs[ArgIndices::input].get(),
                                                    conv_descs_->Conv().get(),
                                                    outDesc.get(),
                                                    outTensor,
                                                    workbuffer,
                                                    conv_descs_->WorkspaceSize(),
                                                    conv_descs_->SolutionId()));
    throwIfError(::miopenConvolutionForwardBias(dnnHandle.get(),
                                               onePtr,
                                               bias_desc_->get(),
                                               inputs[ArgIndices::bias].get(),
                                               onePtr,
                                               outDesc.get(),
                                               outTensor));
    if (includesSecondAddition) {
        throwIfError(::miopenOpTensor(dnnHandle.get(),
                                       miopenTensorOpAdd,
                                       onePtr, outDesc.get(), outTensor,
                                       onePtr, add_desc_->get(), inputs[ArgIndices::add].get(),
                                       zeroPtr, outDesc.get(), outTensor));
    }
    miopenActivationMode_t mode;
    double aAlpha, aBeta, aGamma;
    throwIfError(::miopenGetActivationDescriptor(activation_desc_->get(), &mode, &aAlpha, &aBeta, &aGamma));
    if (mode != miopenActivationPASTHRU) {
        dnnHandle.activationForward(*activation_desc_, onePtr, outDesc, outTensor, zeroPtr, outDesc, outTensor);
    }
}

rocmGraphCompatibility FusedConvolutionmiopenDecomposed::GetrocmGraphCompatibility() const {
    return rocmGraphCompatibility::FULL;
}

WorkbufferRequest FusedConvolutionmiopenDecomposed::GetWorkBufferRequest() const {
    if (conv_descs_->WorkspaceSize() != 0) {
        return {{}, {conv_descs_->WorkspaceSize()}};
    } else {
        return {{}, {}};
    }
}

// miopenConvolutionBiasActivationForward() doesn't work properly with miopen_ACTIVATION_IDENTITY and any algorithm
// other than miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, so we should decompose the convolution node and call
// separate miopen functions.
// For more information see:
// https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenConvolutionBiasActivationForward
void FusedConvolutionmiopenDecomposed::ThrowIfShouldNotDecompose() const {
    miopenActivationMode_t mode;
    miopenNanPropagation_t prop;
    double alpha,beta,gamma;
    throwIfError(::miopenGetActivationDescriptor(activation_desc_->get(), &mode, &alpha, &beta,&gamma));
    if (mode != miopenActivationPASTHRU ||
        conv_descs_->Algo().fwd_algo == miopenConvolutionFwdAlgoImplicitGEMM) {
        throw_ov_exception(
            "ov::rocm_gpu::FusedConvolutionmiopenDecomposed: FusedConvolutionmiopenDecomposed should only be used for "
            "miopen_ACTIVATION_IDENTITY and an algo other than miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
    }
}

}  // namespace rocm_gpu
}  // namespace ov
