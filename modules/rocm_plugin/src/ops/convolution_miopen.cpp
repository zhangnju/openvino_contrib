// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_miopen.hpp"

#include <miopen/miopen.h>

#include <openvino/core/except.hpp>

#include "rocm/constant_factory.hpp"

namespace ov {
namespace rocm_gpu {

Convolutionmiopen::Convolutionmiopen(const CreationContext& context,
                                   const ov::Node& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds,
                                   const Convolution::Details::ConvolutionParams& params)
    : OperationMIOPEN{context, node, std::move(inputIds), std::move(outputIds)}, descs_{context, params} {}

void Convolutionmiopen::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers& workbuffer) const {
    MiopenLockGuard miopen_lock;
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    std::size_t workspace_size = descs_.WorkspaceSize();
    constexpr std::size_t kMaxWorkspace = 64 * 1024 * 1024; // 64MB cap for multi-stream safety
    if (workspace_size > kMaxWorkspace) workspace_size = 0;
    if( workspace_size ==0 )
    {
        uint64_t solution_id = descs_.SolutionId();
        miopenStatus_t status = :: miopenConvolutionForwardImmediate(context.getThreadContext().dnnHandle().get(),
                                                                     descs_.Filter().get(),
                                                                     inputs[Convolution::Details::ConvArgIndices::filter].get(),
                                                                     descs_.Input().get(),
                                                                     inputs[Convolution::Details::ConvArgIndices::input].get(),
                                                                     descs_.Conv().get(),
                                                                     descs_.Output().get(),
                                                                     outputs[Convolution::Details::ConvArgIndices::output].get(),
                                                                     nullptr,
                                                                     0,
                                                                     solution_id);
       throwIfError(status);
    }
    else
    {
        uint64_t solution_id = descs_.SolutionId();
        unsigned char* workspace_data = nullptr;
        hipMalloc((void**)&workspace_data, workspace_size);

       /*
        miopenConvolutionForward(context.getThreadContext().dnnHandle().get(),
                                                     &rocm::NumericConst<rocm::constants::one>(descs_.ElementType()),
                                                     descs_.Input().get(),
                                                     inputs[Convolution::Details::ConvArgIndices::input].get(),
                                                     descs_.Filter().get(),
                                                     inputs[Convolution::Details::ConvArgIndices::filter].get(),
                                                     descs_.Conv().get(),
                                                     descs_.Algo().fwd_algo,
                                                     workbuffer,
                                                     descs_.Algo().memory,
                                                     &rocm::NumericConst<rocm::constants::zero>(descs_.ElementType()),
                                                     descs_.Output().get(),
                                                     outputs[Convolution::Details::ConvArgIndices::output].get()); 
        */
        miopenStatus_t status = :: miopenConvolutionForwardImmediate(context.getThreadContext().dnnHandle().get(),
                                                                     descs_.Filter().get(),
                                                                     inputs[Convolution::Details::ConvArgIndices::filter].get(),
                                                                     descs_.Input().get(),
                                                                     inputs[Convolution::Details::ConvArgIndices::input].get(),
                                                                     descs_.Conv().get(),
                                                                     descs_.Output().get(),
                                                                     outputs[Convolution::Details::ConvArgIndices::output].get(),
                                                                     workspace_data,
                                                                     workspace_size,
                                                                     solution_id);
        if(workspace_data != nullptr)
            hipFree(workspace_data);
                                             
        throwIfError(status);
    }  
   
}

rocmGraphCompatibility Convolutionmiopen::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

WorkbufferRequest Convolutionmiopen::GetWorkBufferRequest() const {
    //if (descs_.Algo().memory != 0)
    //    return {{}, {descs_.Algo().memory}};
    //else
        return {{}, {}};
}

}  // namespace rocm_gpu
}  // namespace ov
