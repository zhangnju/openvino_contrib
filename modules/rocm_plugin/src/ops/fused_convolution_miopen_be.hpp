// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <vector>

#include "convolution_components/convolution_components.hpp"
#include "rocm/dnn_be.hpp"
#include "rocm_operation_base.hpp"
#include "ops/convolution_components/convolution_miopen_components.hpp"

namespace ov {
namespace rocm_gpu {

/**
 * @brief Implements `ov::op::v1::Convolution` using miopen Backend API.
 *
 * miopen Backend API was introduced in miopen version 8 and among other
 * features provides support for asymmetric padding.
 */
class FusedConvolutionmiopenBE : public OperationMIOPEN {
public:
    FusedConvolutionmiopenBE(const CreationContext& context,
                            const ov::Node& node,
                            IndexCollection&& inputIds,
                            IndexCollection&& outputIds,
                            const Convolution::Details::FusedConvolutionParams& params);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    static std::shared_ptr<rocm::DnnBETensorDescriptor> MakeTensorDescriptor(int64_t id,
                                                                             miopenDataType_t element_type,
                                                                             const ov::Shape& shape,
                                                                             const miopenTensorLayout_t format,
                                                                             bool isVirtual = false);
    std::shared_ptr<rocm::DnnBEExecutionPlan> performBenchmarks(
        const rocm::DnnHandle& context, std::vector<std::shared_ptr<rocm::DnnBEExecutionPlan>>& plans);

    std::shared_ptr<rocm::DnnBEEngineConfigDescriptor> engine_config_;
    int64_t workspace_size_ = 0;
    const Convolution::Details::FusedConvolutionParams params_;
};

}  // namespace rocm_gpu
}  // namespace ov
