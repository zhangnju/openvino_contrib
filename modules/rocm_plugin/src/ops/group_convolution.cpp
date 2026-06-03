// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GroupConvolution is lowered to the same factory chain as Convolution:
//   rocMLIR (rock.conv with groupsize>1) > MIOpen-BE > MIOpen Immediate

#include "group_convolution.hpp"

#include <fmt/format.h>
#include <rocm_operation_registry.hpp>
#include <sstream>

#include "convolution_components/convolution_components.hpp"
#include "convolution_miopen.hpp"

#ifdef ENABLE_ROCMLIR
#include "convolution_rocmlir.hpp"
#endif

#ifdef ENABLE_miopen_BACKEND_API
#include "convolution_miopen_be.hpp"
#endif

namespace ov {
namespace rocm_gpu {

// Factory function: same priority chain as Convolution
static OperationBase::Ptr groupConvolutionFactory(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        OperationBase::IndexCollection&& inputIds,
        OperationBase::IndexCollection&& outputIds) {
    using IndexCollection = OperationBase::IndexCollection;
    const Convolution::Details::ConvolutionParams params{
        downcast<const ov::op::v1::GroupConvolution>(node)};
    std::stringstream exception_msg;

#ifdef ENABLE_ROCMLIR
    // rocmlir-gen supports groupsize > 1 via --groupsize flag
    try {
        return std::make_shared<ConvolutionRocMLIR>(
            context, *node,
            IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed ConvolutionRocMLIR: " << e.what();
    }
#endif

#ifdef ENABLE_miopen_BACKEND_API
    try {
        return std::make_shared<ConvolutionmiopenBE>(
            context, *node,
            IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed ConvolutionmiopenBE: " << e.what();
    }
#endif

    try {
        return std::make_shared<Convolutionmiopen>(
            context, *node,
            IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "Failed Convolutionmiopen: " << e.what();
    }
    throw_ov_exception(fmt::format("GroupConvolution not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(groupConvolutionFactory, GroupConvolution);

}  // namespace rocm_gpu
}  // namespace ov
