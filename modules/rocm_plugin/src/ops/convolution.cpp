// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <openvino/core/except.hpp>
#include <sstream>

#include "convolution_miopen.hpp"
#include "rocm_operation_registry.hpp"

#ifdef ENABLE_miopen_BACKEND_API
#include "convolution_miopen_be.hpp"
#endif  // ENABLE_miopen_BACKEND_API

namespace ov {
namespace rocm_gpu {

static OperationBase::Ptr convolutionFactory(const CreationContext& context,
                                             const std::shared_ptr<ov::Node>& node,
                                             OperationBase::IndexCollection&& inputIds,
                                             OperationBase::IndexCollection&& outputIds) {
    using IndexCollection = OperationBase::IndexCollection;
    const Convolution::Details::ConvolutionParams params{downcast<const ov::op::v1::Convolution>(node)};
    std::stringstream exception_msg;
#ifdef ENABLE_miopen_BACKEND_API
    try {
        //std::cout<<"miopen backend"<<std::endl;
        return std::make_shared<ConvolutionmiopenBE>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create ConvolutionmiopenBE impl: " << e.what();
    }
#endif  // ENABLE_miopen_BACKEND_API
    try {
        //std::cout<<"not miopen backend"<<std::endl;
        return std::make_shared<Convolutionmiopen>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "Failed to create Convolutionmiopen impl: " << e.what();
    }
    throw_ov_exception(fmt::format("Convolution node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(convolutionFactory, Convolution);

}  // namespace rocm_gpu
}  // namespace ov
