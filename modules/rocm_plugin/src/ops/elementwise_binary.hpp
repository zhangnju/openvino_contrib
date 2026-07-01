// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <rocm_operation_base.hpp>

#include <type_traits>
#include "components/numpy_broadcast_params.h"
#include "converters.hpp"

namespace ov {
namespace rocm_gpu {

namespace detail {
template <typename K, typename = void>
struct has_bcast_periodic : std::false_type {};
template <typename K>
struct has_bcast_periodic<K, std::void_t<decltype(K::has_bcast_periodic)>> : std::bool_constant<K::has_bcast_periodic> {};
}  // namespace detail

template <typename nGraphNode, typename Kernel>
class ElementwiseBinaryOp : public OperationBase {
public:
    using NodeOp = nGraphNode;
    ElementwiseBinaryOp(const CreationContext& context,
                        const NodeOp& node,
                        IndexCollection&& inputIds,
                        IndexCollection&& outputIds)
        : OperationBase{context, node, std::move(inputIds), std::move(outputIds)},
          in0_broadcast_params_{NumpyBroadcastParams::create(node.get_input_shape(0), node.get_output_shape(0))},
          in1_broadcast_params_{NumpyBroadcastParams::create(node.get_input_shape(1), node.get_output_shape(0))} {
        OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());
        OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

        const auto element_type = node.get_output_element_type(0);
        const bool types_are_expected =
            (element_type == node.get_input_element_type(0)) && (element_type == node.get_input_element_type(1));
        if (!types_are_expected) {
            throw_ov_exception("Element types combination is not supported");
        }

        in0_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);
        in1_broadcast_params_->addWorkbufferRequests(immutable_buffer_sizes_);

        const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
        out_num_elements_ = ov::shape_size(node.get_output_shape(0));
        kernel_ = Kernel{
            convertDataType<ov::rocm_gpu::kernel::Type_t>(element_type), out_num_elements_, max_threads_per_block};

        // Detect periodic broadcast on in1 for fast path (Multiply only).
        // in1 broadcasts along exactly one axis k → period=out[k], repeat=product(dims after k).
        if constexpr (detail::has_bcast_periodic<Kernel>::value) {
            auto in1_shape = node.get_input_shape(1);
            auto out_shape = node.get_output_shape(0);
            if (element_type == ov::element::f16 && out_num_elements_ % 2 == 0 &&
                in1_shape.size() == out_shape.size()) {
                int bcast_axis = -1;
                for (size_t i = 0; i < out_shape.size(); i++) {
                    if (in1_shape[i] != out_shape[i]) {
                        if (in1_shape[i] == 1 && bcast_axis == -1) {
                            // Found a broadcast axis — but we want the NON-broadcast axis of in1
                        } else if (in1_shape[i] != 1) {
                            bcast_axis = -2; break; // complex broadcast
                        }
                    }
                }
                // Find the single non-trivial axis of in1
                int in1_axis = -1;
                for (size_t i = 0; i < in1_shape.size(); i++) {
                    if (in1_shape[i] > 1) {
                        if (in1_axis >= 0) { in1_axis = -2; break; } // multiple non-trivial axes
                        in1_axis = i;
                    }
                }
                if (in1_axis >= 0 && in1_axis != -2) {
                    bcast_period_ = in1_shape[in1_axis];
                    bcast_repeat_ = 1;
                    for (size_t i = in1_axis + 1; i < out_shape.size(); i++)
                        bcast_repeat_ *= out_shape[i];
                    if (bcast_repeat_ == 1) {
                        // Last-dim broadcast: period must be even for half2 alignment
                        if (bcast_period_ % 2 == 0)
                            bcast_fast_ = true;
                    } else {
                        // Inner-dim broadcast: repeat must be even for half2 alignment
                        if (bcast_repeat_ % 2 == 0)
                            bcast_fast_ = true;
                    }
                }
            }
        }
    }

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {
        OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
        OPENVINO_ASSERT(inputTensors.size() == 2, "Node name: ", GetName());
        OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());
        auto& stream = context.getThreadContext().stream();

        // Fast periodic broadcast path for Multiply fp16
        if constexpr (detail::has_bcast_periodic<Kernel>::value) {
            if (bcast_fast_) {
                kernel_->launch_bcast_periodic_f16(stream.get(),
                    inputTensors[0].get(), inputTensors[1].get(),
                    outputTensors[0].get(),
                    out_num_elements_, bcast_period_, bcast_repeat_);
                return;
            }
        }

        (*kernel_)(stream.get(),
                   static_cast<const void*>(inputTensors[0].get()),
                   in0_broadcast_params_->mapper(workbuffers.immutable_buffers),
                   static_cast<const void*>(inputTensors[1].get()),
                   in1_broadcast_params_->mapper(workbuffers.immutable_buffers),
                   static_cast<void*>(outputTensors[0].get()));
    }

    rocmGraphCompatibility GetrocmGraphCompatibility() const override { return rocmGraphCompatibility::FULL; }

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) override {
        in0_broadcast_params_->initWorkbuffers(buffers);
        in1_broadcast_params_->initWorkbuffers(buffers);
    }

    WorkbufferRequest GetWorkBufferRequest() const override { return {immutable_buffer_sizes_, {}}; }

private:
    std::vector<WorkbufferRequest::size_in_bytes_t> immutable_buffer_sizes_;
    std::unique_ptr<NumpyBroadcastParams> in0_broadcast_params_;
    std::unique_ptr<NumpyBroadcastParams> in1_broadcast_params_;

    std::optional<Kernel> kernel_;
    size_t out_num_elements_ = 0;
    bool bcast_fast_ = false;
    size_t bcast_period_ = 0;
    size_t bcast_repeat_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
