// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <rocm_operation_base.hpp>
#include <kernels/pad.hpp>
#include <openvino/op/pad.hpp>

namespace ov {
namespace rocm_gpu {

class PadOp : public OperationBase {
public:
    using NodeOp = ov::op::v1::Pad;
    explicit PadOp(const CreationContext& context,
                   const ov::op::v1::Pad& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers&) override;

private:
    enum WorkbufferIndex {
        kSrcShape,
        kDstShape,
    };

    enum InputIndex {
        kSrc,
        kPadsBegin,
        kPadsEnd,
        kPadValue,
    };

    enum OutputIndex {
        kDst,
    };

    kernel::ConstModePad kernel_;
    ov::Shape src_shape_;
    ov::Shape dst_shape_;
};

}  // namespace rocm_gpu
}  // namespace ov
