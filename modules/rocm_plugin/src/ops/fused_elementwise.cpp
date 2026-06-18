// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fused_elementwise.hpp"
#include "rocm_operation_registry.hpp"
#include "rocm/runtime.hpp"

#include <cstring>
#include <cstdlib>
#include <cstdio>
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
    // i32 primary input = the chain head is a Convert(i32->f32). The kernel reads
    // the primary as int32 and converts to float on load; aux/out stay f32.
    prim_is_i32_ = (node->get_input_element_type(0) == ov::element::i32);

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
        else if (s.op_type == "Round")      op_code = uint8_t(Op::Round);
        else if (s.op_type == "Clamp")      op_code = uint8_t(Op::Clamp);
        else if (s.op_type == "Convert")    op_code = uint8_t(Op::Cast);
        else OPENVINO_THROW("FusedElementwiseOp: unknown op '", s.op_type, "'");

        host_ops_.push_back(op_code);
        host_params_.push_back(param);
        step_has_aux_.push_back(s.has_aux);
        if (s.has_aux) num_aux_++;
    }

    // Determine, per step, how its aux input must be indexed by the kernel:
    //   * scalar (1 element)            → read aux[0]      → ops bit 7
    //   * last-dim/per-channel [C]      → read aux[i % C]  → ops bit 6, C in params
    //   * full (== output element count)→ read aux[i]      → no flag
    // Aux inputs are appended to the node inputs in chain order (input 0 = primary).
    // The flags are packed into the high bits of the ops byte (op codes are < 32).
    const auto& out_shape = node->get_output_shape(0);
    const size_t channel = out_shape.empty() ? 1 : out_shape.back();
    host_aux_scalar_.assign(chain_len_, 0);
    {
        int aux_in = 1;
        for (int s = 0; s < chain_len_; ++s) {
            if (!step_has_aux_[s]) continue;
            if (aux_in < static_cast<int>(node->get_input_size())) {
                const auto& aux_shape = node->get_input_shape(aux_in);
                size_t aux_elems = 1;
                for (auto d : aux_shape) aux_elems *= d;
                if (aux_elems == 1) {
                    host_aux_scalar_[s] = 1; host_ops_[s] |= 0x80;        // scalar broadcast
                } else if (aux_elems == channel && channel > 1 && num_elements_ != (int64_t)channel) {
                    host_ops_[s] |= 0x40;                                 // per-channel broadcast
                    host_params_[s] = static_cast<float>(channel);        // C (exact for C < 2^24)
                }
                ++aux_in;
            }
        }
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
    const size_t aux_bytes    = chain_len_ * sizeof(void*);
    // pinned_sizes[0]: page-locked host mirror for aux_ptrs (stable address for hipGraph).
    // pinned_sizes[1]: shadow copy for change detection (P3 optimisation: skip H2D if unchanged).
    // The per-step aux-is-scalar flag is packed into bit 7 of each ops byte (see
    // constructor), so no extra immutable buffer is needed.
    return {{ops_bytes, params_bytes}, {aux_bytes}, {aux_bytes, aux_bytes}};
}

void FusedElementwiseOp::InitSharedImmutableWorkbuffers(const Buffers& buffers) {
    OPENVINO_ASSERT(buffers.size() == 2, "FusedElementwiseOp: expected 2 immutable workbuffers");
    rocm::DefaultStream::stream().upload(buffers[0], host_ops_.data(),    chain_len_ * sizeof(uint8_t));
    rocm::DefaultStream::stream().upload(buffers[1], host_params_.data(), chain_len_ * sizeof(float));
}

// ─── Execute ─────────────────────────────────────────────────────────────────

// Helper: write current aux input addresses into the pinned host slot.
static void update_aux_ptrs(void* pinned_slot,
                             const IOperationExec::Inputs& inputs,
                             const std::vector<bool>& step_has_aux,
                             int chain_len) {
    auto* dst = static_cast<const void**>(pinned_slot);
    int aux_idx = 1;
    for (int s = 0; s < chain_len; ++s) {
        dst[s] = (step_has_aux[s] && aux_idx < (int)inputs.size())
                 ? inputs[aux_idx++].get()
                 : nullptr;
    }
}

void FusedElementwiseOp::Execute(const InferenceRequestContext& ctx,
                                  Inputs inputs,
                                  Outputs outputs,
                                  const Workbuffers& wbs) const {
    OPENVINO_ASSERT(!inputs.empty(),  GetName(), ": no inputs");
    OPENVINO_ASSERT(!outputs.empty(), GetName(), ": no outputs");
    OPENVINO_ASSERT(wbs.immutable_buffers.size() == 2, GetName(), ": need 2 immutable wbs");
    OPENVINO_ASSERT(!wbs.mutable_buffers.empty(),       GetName(), ": need mutable wb for aux ptrs");
    OPENVINO_ASSERT(!wbs.pinned_buffers.empty(),        GetName(), ": need pinned wb for aux ptrs");

    // Write aux ptrs into page-locked host slot, then H2D to device workbuffer.
    // Optimization (P3): if aux_ptrs are unchanged from the previous call
    // (tensor addresses are stable in OV's memory pool for fixed shapes),
    // skip the H2D upload. This eliminates ~26 hipMemcpyAsync calls per
    // BERT inference, saving ~50µs of copyBuffer overhead.
    void* pinned = wbs.pinned_buffers[0];
    update_aux_ptrs(pinned, inputs, step_has_aux_, chain_len_);
    void* aux_ptrs_device = wbs.mutable_buffers[0].get();

    const size_t aux_bytes = chain_len_ * sizeof(void*);
    // Always upload the aux-pointer table. (A previous "skip H2D if unchanged"
    // optimization compared against an uninitialized shadow buffer and left stale
    // device pointers in steady state → GPU memory-access faults on the INT8
    // dequant chains. The saved ~50µs is not worth the correctness hazard.)
    hipMemcpyAsync(aux_ptrs_device, pinned, aux_bytes, hipMemcpyHostToDevice,
                   ctx.getThreadContext().stream().get());

    const uint8_t* ops_dev    = static_cast<const uint8_t*>(wbs.immutable_buffers[0].get());
    const float*   params_dev = static_cast<const float*>(wbs.immutable_buffers[1].get());
    kernel::launch_fused_elementwise(
        inputs[0].get(),
        static_cast<const void* const*>(aux_ptrs_device),
        outputs[0].get(),
        num_elements_, ops_dev, params_dev, chain_len_, is_fp16_, prim_is_i32_,
        ctx.getThreadContext().stream().get());
}

void FusedElementwiseOp::Capture(InferenceRequestContext& ctx,
                                  Inputs inputs,
                                  Outputs outputs,
                                  const Workbuffers& wbs) const {
    // Warmup Execute already populated the pinned slot with correct addresses.
    // During capture the hipMemcpyAsync H2D is recorded as a graph node.
    // Source = wbs.pinned_buffers[0] (stable per-request pinned pool address),
    // so replay reads from that same stable address — no dangling pointer.
    OPENVINO_ASSERT(!wbs.pinned_buffers.empty(), GetName(), ": need pinned wb");
    void* pinned = wbs.pinned_buffers[0];
    void* aux_ptrs_device = wbs.mutable_buffers[0].get();
    hipMemcpyAsync(aux_ptrs_device, pinned,
                   chain_len_ * sizeof(void*), hipMemcpyHostToDevice,
                   ctx.getThreadContext().stream().get());

    const uint8_t* ops_dev    = static_cast<const uint8_t*>(wbs.immutable_buffers[0].get());
    const float*   params_dev = static_cast<const float*>(wbs.immutable_buffers[1].get());
    kernel::launch_fused_elementwise(
        inputs[0].get(),
        static_cast<const void* const*>(aux_ptrs_device),
        outputs[0].get(),
        num_elements_, ops_dev, params_dev, chain_len_, is_fp16_, prim_is_i32_,
        ctx.getThreadContext().stream().get());
}

void FusedElementwiseOp::ExecuteGraph(InferenceRequestContext& ctx,
                                       Inputs inputs,
                                       Outputs outputs,
                                       const Workbuffers& wbs) const {
    // Called BEFORE hipGraphLaunch() during replay. Update the pinned slot so that
    // the captured H2D node copies fresh input addresses into the device workbuffer.
    if (!wbs.pinned_buffers.empty())
        update_aux_ptrs(wbs.pinned_buffers[0], inputs, step_has_aux_, chain_len_);
    // H2D + kernel launch are handled by the hipGraph replay.
}

OPERATION_REGISTER(FusedElementwiseOp, FusedElementwise);

}  // namespace rocm_gpu
}  // namespace ov
