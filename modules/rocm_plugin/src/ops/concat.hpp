// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <miopen/miopen.h>

#include <atomic>
#include <rocm/device_pointers.hpp>
#include <rocm_operation_base.hpp>
#include <kernels/concat.hpp>
#include <openvino/op/concat.hpp>

namespace ov {
namespace rocm_gpu {

class ConcatOp : public OperationBase {
public:
    using NodeOp = ov::op::v0::Concat;
    ConcatOp(const CreationContext& context,
             const NodeOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);
    ~ConcatOp();
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers&) override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override;

private:
    size_t immutableWbSize() const { return concat_kernel_.value().immutableWbSize(); }
    size_t mutableWbSize() const { return concat_kernel_.value().mutableWbSize(); }

    std::size_t num_inputs_;
    std::optional<kernel::Concat> concat_kernel_;
    // Pinned host buffer for HIP Graph-compatible pointer table uploads.
    // Allocated once, filled at Execute time. H2D from pinned memory can be
    // captured in HIP Graph; the pointer values are stable for static models.
    mutable void* pinned_ptr_table_ = nullptr;
    mutable std::atomic<bool> ptr_table_initialized_{false};
};

}  // namespace rocm_gpu
}  // namespace ov
