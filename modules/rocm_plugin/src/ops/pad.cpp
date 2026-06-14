// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "pad.hpp"

#include <cstddef>
#include <memory>

#include "converters.hpp"
#include "rocm/runtime.hpp"
#include "rocm_operation_registry.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace rocm_gpu {

static bool isNCHWConvolutionPadding(const PadOp::NodeOp& node) {
    auto padsBegin = ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(1));
    auto padsEnd = ov::as_type_ptr<op::v0::Constant>(node.get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(padsBegin && padsEnd, "Non-constant paddings are unsupported!");
    // Safe reading: pads may be stored as int32 or int64, both need to work
    auto safe_read_pads = [](const std::shared_ptr<ov::op::v0::Constant>& c) {
        const auto& et = c->get_element_type();
        if (et == ov::element::i32) {
            std::vector<size_t> v;
            for (auto x : c->cast_vector<int32_t>()) v.push_back(static_cast<size_t>(x));
            return v;
        }
        if (et == ov::element::i64) {
            std::vector<size_t> v;
            for (auto x : c->cast_vector<int64_t>()) v.push_back(static_cast<size_t>(x));
            return v;
        }
        return c->cast_vector<size_t>();
    };
    const auto padsBeginCoord = safe_read_pads(padsBegin);
    const auto padsEndCoord = safe_read_pads(padsEnd);
    return node.get_input_shape(0).size() == 4 && padsBeginCoord[0] == 0 && padsBeginCoord[1] == 0 &&
           padsEndCoord[0] == 0 && padsEndCoord[1] == 0;
}

PadOp::PadOp(const CreationContext& context,
             const NodeOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds)
    : OperationBase{context, node, std::move(inputIds), std::move(outputIds)},
      kernel_{eltwise::KernelExecAttrs{
                  node.get_output_shape(0),
                  kernel::ConstModePad::kWarpsPerBlock * static_cast<unsigned>(context.device().props().warpSize),
                  kernel::ConstModePad::kElementsPerThread},
              convertDataType<kernel::Type_t>(node.get_output_element_type(0)),
              node.get_output_shape(0).size(),
              context.device().props().maxThreadsPerBlock,
              ov::shape_size(node.get_output_shape(0)),
              isNCHWConvolutionPadding(node)},
      src_shape_{node.get_input_shape(0)},
      dst_shape_{node.get_output_shape(0)} {
    OPENVINO_ASSERT(node.get_input_element_type(0) == node.get_output_element_type(0), "Node name: ", GetName());
    OPENVINO_ASSERT(ov::op::PadMode::CONSTANT == node.get_pad_mode(), "Node name: ", GetName());
}

void PadOp::Execute(const InferenceRequestContext& context,
                    Inputs inputTensors,
                    Outputs outputTensors,
                    const Workbuffers& workbuffers) const {
    kernel_(context.getThreadContext().stream().get(),
            inputTensors[InputIndex::kSrc].get(),
            outputTensors[OutputIndex::kDst].get(),
            inputTensors[InputIndex::kPadsBegin].get(),
            workbuffers.immutable_buffers[WorkbufferIndex::kSrcShape].cast<const std::size_t*>().get(),
            workbuffers.immutable_buffers[WorkbufferIndex::kDstShape].cast<const std::size_t*>().get(),
            inputTensors[InputIndex::kPadValue].get());
}

rocmGraphCompatibility PadOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

WorkbufferRequest PadOp::GetWorkBufferRequest() const {
    auto rank = src_shape_.size();
    return {{rank * sizeof(std::size_t), rank * sizeof(std::size_t)}, {}};
}

void PadOp::InitSharedImmutableWorkbuffers(const Buffers& devicePointers) {
    rocm::DefaultStream::stream().upload(
        devicePointers[WorkbufferIndex::kSrcShape], src_shape_.data(), src_shape_.size() * sizeof(std::size_t));
    rocm::DefaultStream::stream().upload(
        devicePointers[WorkbufferIndex::kDstShape], dst_shape_.data(), src_shape_.size() * sizeof(std::size_t));
}

OPERATION_REGISTER(PadOp, Pad);

}  // namespace rocm_gpu
}  // namespace ov
