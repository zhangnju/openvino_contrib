// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fused_elementwise.hpp"
#include "rocm_operation_registry.hpp"
#include "rocm/runtime.hpp"

#include <openvino/core/except.hpp>
#include <fmt/format.h>

namespace ov {
namespace rocm_gpu {

// ─── Constructor ─────────────────────────────────────────────────────────────

FusedElementwiseOp::FusedElementwiseOp(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds))
{
    auto* fe = dynamic_cast<const nodes::FusedElementwise*>(node.get());
    OPENVINO_ASSERT(fe, "FusedElementwiseOp: expected FusedElementwise node");

    const auto& steps = fe->steps();
    chain_len_ = static_cast<int>(steps.size());
    OPENVINO_ASSERT(chain_len_ > 0 && chain_len_ <= kernel::kFusedEwMaxChain,
        "FusedElementwiseOp: chain length out of range: ", chain_len_);

    is_fp16_ = (node->get_input_element_type(0) == ov::element::f16);

    // Compute output element count
    const auto& shape = node->get_output_shape(0);
    num_elements_ = 1;
    for (auto d : shape) num_elements_ *= static_cast<int64_t>(d);

    // Encode steps
    num_aux_ = 0;
    for (const auto& s : steps) {
        uint8_t op_code = 0;
        float   param   = 0.f;
        using Op = kernel::FusedEwOp;

        if      (s.op_type == "Relu")       op_code = uint8_t(Op::Relu);
        else if (s.op_type == "Sigmoid")    op_code = uint8_t(Op::Sigmoid);
        else if (s.op_type == "Swish"
              || s.op_type == "SiLU")       op_code = uint8_t(Op::SiLU);
        else if (s.op_type == "Tanh")       op_code = uint8_t(Op::Tanh);
        else if (s.op_type == "Gelu")       op_code = uint8_t(Op::Gelu);
        else if (s.op_type == "HardSwish")  op_code = uint8_t(Op::HardSwish);
        else if (s.op_type == "Abs")        op_code = uint8_t(Op::Abs);
        else if (s.op_type == "Neg")        op_code = uint8_t(Op::Neg);
        else if (s.op_type == "Sqrt")       op_code = uint8_t(Op::Sqrt);
        else if (s.op_type == "Exp")        op_code = uint8_t(Op::Exp);
        else if (s.op_type == "Log")        op_code = uint8_t(Op::Log);
        else if (s.op_type == "Erf")        op_code = uint8_t(Op::Erf);
        else if (s.op_type == "Add")        op_code = uint8_t(Op::Add);
        else if (s.op_type == "Subtract")   op_code = uint8_t(Op::Sub);
        else if (s.op_type == "Multiply")   op_code = uint8_t(Op::Mul);
        else if (s.op_type == "Divide")     op_code = uint8_t(Op::Div);
        else if (s.op_type == "LeakyRelu")  { op_code = uint8_t(Op::LeakyRelu); param = s.param; }
        else OPENVINO_THROW("FusedElementwiseOp: unknown op '", s.op_type, "'");

        host_ops_.push_back(op_code);
        host_params_.push_back(param);
        step_has_aux_.push_back(s.has_aux);
        if (s.has_aux) num_aux_++;
    }
}

// ─── Workbuffers ─────────────────────────────────────────────────────────────
// Immutable buffers (uploaded once, reused each inference):
//   [0]: ops array    — chain_len_ bytes
//   [1]: params array — chain_len_ * sizeof(float)
//   [2]: aux_ptrs array — chain_len_ * sizeof(void*) (updated per-Execute via Execute workbuf)

WorkbufferRequest FusedElementwiseOp::GetWorkBufferRequest() const {
    const size_t ops_bytes    = chain_len_ * sizeof(uint8_t);
    const size_t params_bytes = chain_len_ * sizeof(float);
    // Mutable: aux pointer array (chain_len_ void* entries, updated each Execute)
    const size_t aux_bytes    = chain_len_ * sizeof(void*);
    return {{ops_bytes, params_bytes}, {aux_bytes}};
}

void FusedElementwiseOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    OPENVINO_ASSERT(buffers.size() == 2, "FusedElementwiseOp: expected 2 immutable workbuffers");
    rocm::DefaultStream::stream().upload(buffers[0], host_ops_.data(),    chain_len_ * sizeof(uint8_t));
    rocm::DefaultStream::stream().upload(buffers[1], host_params_.data(), chain_len_ * sizeof(float));
}

// ─── Execute ─────────────────────────────────────────────────────────────────

void FusedElementwiseOp::Execute(const InferenceRequestContext& ctx,
                                  Inputs inputs,
                                  Outputs outputs,
                                  const Workbuffers& wbs) const {
    OPENVINO_ASSERT(!inputs.empty(),  GetName(), ": no inputs");
    OPENVINO_ASSERT(!outputs.empty(), GetName(), ": no outputs");
    OPENVINO_ASSERT(wbs.immutable_buffers.size() == 2, GetName(), ": need 2 immutable wbs");
    OPENVINO_ASSERT(!wbs.mutable_buffers.empty(),       GetName(), ": need mutable wb for aux ptrs");

    // Build aux pointer array on host, then upload to the mutable workbuffer
    // The mutable workbuffer holds chain_len_ void* entries
    std::vector<const void*> aux_host(chain_len_, nullptr);
    int aux_idx = 1;  // inputs[0] is primary; aux inputs start at [1]
    for (int s = 0; s < chain_len_; ++s) {
        if (step_has_aux_[s] && aux_idx < (int)inputs.size()) {
            aux_host[s] = inputs[aux_idx].get();
            aux_idx++;
        }
    }

    // Upload aux pointer array to mutable workbuffer (D2H pointer values → D2D copy of pointers)
    void* aux_ptrs_device = wbs.mutable_buffers[0].get();
    hipMemcpyAsync(aux_ptrs_device, aux_host.data(),
                   chain_len_ * sizeof(void*), hipMemcpyHostToDevice,
                   ctx.getThreadContext().stream().get());

    // Launch fused kernel
    const uint8_t* ops_dev    = static_cast<const uint8_t*>(wbs.immutable_buffers[0].get());
    const float*   params_dev = static_cast<const float*>(wbs.immutable_buffers[1].get());

    kernel::launch_fused_elementwise(
        inputs[0].get(),
        static_cast<const void* const*>(aux_ptrs_device),
        outputs[0].get(),
        num_elements_,
        ops_dev,
        params_dev,
        chain_len_,
        is_fp16_,
        ctx.getThreadContext().stream().get());
}

OPERATION_REGISTER(FusedElementwiseOp, FusedElementwise);

}  // namespace rocm_gpu
}  // namespace ov
