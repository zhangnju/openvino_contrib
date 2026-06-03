// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_miopen.hpp"

#include <miopen/miopen.h>

#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#include "rocm/constant_factory.hpp"
#include "transformer/nodes/activation_type.hpp"

namespace ov {
namespace rocm_gpu {

FusedConvolutionmiopen::FusedConvolutionmiopen(const CreationContext& context,
                                             const ov::Node& node,
                                             IndexCollection&& inputIds,
                                             IndexCollection&& outputIds,
                                             Convolution::Details::FusedConvolutionParams params)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)},
      conv_descs_{std::make_shared<Convolution::Details::ConvolutionDescriptorsmiopen>(context, params.conv_,
        std::vector<miopenDataType_t>{miopenHalf, miopenFloat})}, // 119703: investigate whether we need HALF here
      bias_desc_{Convolution::Details::MakeFusedAddDescriptor(params.bias_shape_, params.conv_.element_type_)},
      add_desc_{params.add_shape_ ? Convolution::Details::MakeFusedAddDescriptor(params.add_shape_.value(),
                                                                                 params.conv_.element_type_)
                                  : nullptr},
      activation_desc_{Convolution::Details::MakeFusedActivationDescriptor(params.activation_)} {
    ThrowIfShouldDecompose();
}

FusedConvolutionmiopen::FusedConvolutionmiopen(
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
      add_desc_{addDesc},
      activation_desc_{activationDesc} {
    ThrowIfShouldDecompose();
}

void FusedConvolutionmiopen::Execute(const InferenceRequestContext& context,
                                    Inputs inputs,
                                    Outputs outputs,
                                    const Workbuffers& workbuffers) const {
    using ArgIndices = Convolution::Details::FusedConvolutionIndices;

    const bool includesOnlyBiasAdd = inputs.size() == 3;
    const bool includesSecondAddition = inputs.size() == 4;
    OPENVINO_ASSERT(includesOnlyBiasAdd || includesSecondAddition, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    void* workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    const auto& dnnHandle = context.getThreadContext().dnnHandle();

    const rocm::constants::AnyNumeric* alpha2 = nullptr;
    miopenTensorDescriptor_t zTensorDesc;
    const void* zTensorIn = nullptr;
    if (includesOnlyBiasAdd) {
        alpha2 = &rocm::NumericConst<rocm::constants::zero>(conv_descs_->DescType());
        zTensorDesc = conv_descs_->Output().get();
        zTensorIn = outputs[ArgIndices::output].get();
    } else {
        alpha2 = &rocm::NumericConst<rocm::constants::one>(conv_descs_->DescType());
        zTensorDesc = add_desc_->get();
        zTensorIn = inputs[ArgIndices::add].get();
    }
    throwIfError(
        ::miopenConvolutionBiasActivationForward(dnnHandle.get(),
                                                &rocm::NumericConst<rocm::constants::one>(conv_descs_->DescType()),
                                                conv_descs_->Input().get(),
                                                inputs[ArgIndices::input].get(),
                                                conv_descs_->Filter().get(),
                                                inputs[ArgIndices::filter].get(),
                                                conv_descs_->Conv().get(),
                                                conv_descs_->Algo().fwd_algo,
                                                workbuffer,
                                                conv_descs_->Algo().memory,
                                                alpha2,
                                                zTensorDesc,
                                                zTensorIn,
                                                bias_desc_->get(),
                                                inputs[ArgIndices::bias].get(),
                                                activation_desc_->get(),
                                                conv_descs_->Output().get(),
                                                outputs[ArgIndices::output].get()));
}

rocmGraphCompatibility FusedConvolutionmiopen::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

WorkbufferRequest FusedConvolutionmiopen::GetWorkBufferRequest() const {
    if (conv_descs_->Algo().memory != 0)
        return {{}, {conv_descs_->Algo().memory}};
    else
        return {{}, {}};
}

// miopenConvolutionBiasActivationForward() doesn't work properly with miopen_ACTIVATION_IDENTITY and any algorithm
// other than miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, so we should decompose the convolution node and call
// separate miopen functions.
// For more information see:
// https://docs.rocm.com/deeplearning/miopen/api/index.html#miopenConvolutionBiasActivationForward
void FusedConvolutionmiopen::ThrowIfShouldDecompose() const {
    miopenActivationMode_t mode;
    miopenNanPropagation_t prop;
    double alpha,beta,gamma;
    throwIfError(::miopenGetActivationDescriptor(activation_desc_->get(), &mode, &alpha, &beta,&gamma));
    /*
    if (mode == miopen_ACTIVATION_IDENTITY &&
        conv_descs_->Algo().algo != miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
        throw_ov_exception(
            "ov::rocm_gpu::FusedConvolutionmiopen: miopen_ACTIVATION_IDENTITY can't be used with "
            "miopen_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
    }
    */
}

}  // namespace rocm_gpu
}  // namespace ov
