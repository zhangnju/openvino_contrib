// Copyright (C) 2018-2024 Intel Corporation
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

#ifdef ENABLE_ROCMLIR
#include "convolution_rocmlir.hpp"
#endif  // ENABLE_ROCMLIR

#ifdef ENABLE_CK_WMMA
#include "convolution_ck_wmma.hpp"
#endif  // ENABLE_CK_WMMA

namespace ov {
namespace rocm_gpu {

static OperationBase::Ptr convolutionFactory(const CreationContext& context,
                                             const std::shared_ptr<ov::Node>& node,
                                             OperationBase::IndexCollection&& inputIds,
                                             OperationBase::IndexCollection&& outputIds) {
    using IndexCollection = OperationBase::IndexCollection;
    const Convolution::Details::ConvolutionParams params{downcast<const ov::op::v1::Convolution>(node)};
    std::stringstream exception_msg;

    // Priority 0: CK WMMA + rocMLIR auto-selection (RDNA4/gfx12xx, FP16 only)
    // On first run profiles both kernels and permanently selects the faster one.
#ifdef ENABLE_CK_WMMA
    try {
        return std::make_shared<ConvolutionCkWmma>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed ConvolutionCkWmma (likely non-RDNA4 or non-f16): " << e.what();
    }
#endif  // ENABLE_CK_WMMA

    // Priority 1: rocMLIR backend (rock.conv → HSACO → hipLaunchKernel)
    // Bypasses MIOpen entirely, avoids gfx950 solver instability.
#ifdef ENABLE_ROCMLIR
    try {
        return std::make_shared<ConvolutionRocMLIR>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create ConvolutionRocMLIR impl: " << e.what();
    }
#endif  // ENABLE_ROCMLIR

    // Priority 2: MIOpen Backend API
#ifdef ENABLE_miopen_BACKEND_API
    try {
        return std::make_shared<ConvolutionmiopenBE>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create ConvolutionmiopenBE impl: " << e.what();
    }
#endif  // ENABLE_miopen_BACKEND_API

    // Priority 3: MIOpen Immediate Mode (legacy fallback)
    try {
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
