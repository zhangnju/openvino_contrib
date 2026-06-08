// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// VariadicSplitAliasOp: zero-copy VariadicSplit for contiguous channel-axis splits.
//
// When ROCM_ZEROCOPY_SPLIT=1 and the split is along axis=1 (channels) with
// contiguous outputs, the memory planner (rocm_op_buffers_extractor) assigns
// output[i] as a sub-range of the input buffer with the appropriate channel
// offset. Execute() is therefore a no-op — the data is already in the right place.
//
// This mirrors MIGraphX's `load[offset, end](@scratch)` mechanism which makes
// Split effectively zero-cost by aliasing output pointers into the scratch buffer.

#pragma once

#include <rocm_operation_base.hpp>
#include <openvino/op/op.hpp>
#include <openvino/op/variadic_split.hpp>

namespace ov {
namespace rocm_gpu {

// Custom OV node type that marks a VariadicSplit as alias-able.
// Used by extractSplitAliasTensors() in the memory planner to assign
// output buffers as sub-ranges (with offset) of the input buffer.
namespace nodes {
class VariadicSplitAlias : public ov::op::v1::VariadicSplit {
public:
    OPENVINO_OP("VariadicSplitAlias", "rocm_gpu", ov::op::v1::VariadicSplit);
    using ov::op::v1::VariadicSplit::VariadicSplit;  // inherit ctors

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        ov::check_new_args_count(this, new_args);
        return std::make_shared<VariadicSplitAlias>(new_args[0], new_args[1], new_args[2]);
    }
};
}  // namespace nodes

// The actual op implementation: Execute() is a no-op.
class VariadicSplitAliasOp : public OperationBase {
public:
    VariadicSplitAliasOp(const CreationContext& context,
                          const ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds)
        : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {}

    void Execute(const InferenceRequestContext& /*ctx*/,
                 Inputs /*inputs*/,
                 Outputs /*outputs*/,
                 const Workbuffers& /*wbs*/) const override {
        // Zero-copy: output buffers are already aliases of input buffer (set by memory planner).
        // No GPU operation needed.
    }

    WorkbufferRequest GetWorkBufferRequest() const override { return {{}, {}}; }

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }
};

}  // namespace rocm_gpu
}  // namespace ov
