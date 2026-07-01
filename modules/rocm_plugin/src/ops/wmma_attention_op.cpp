#include "wmma_attention_op.hpp"
#include "transformer/nodes/wmma_attention_node.hpp"
#include "rocm_creation_context.hpp"
#include <rocm_operation_registry.hpp>
#include <cstdio>

namespace ov { namespace rocm_gpu {

WMMAAttentionOp::WMMAAttentionOp(
    const CreationContext& ctx, const std::shared_ptr<ov::Node>& node,
    IndexCollection&& in, IndexCollection&& out)
    : OperationBase(ctx, node, std::move(in), std::move(out))
{
    auto n = std::dynamic_pointer_cast<nodes::WMMAAttention>(node);
    batch_ = n->batch(); heads_ = n->heads();
    seqlen_q_ = n->seqlen_q(); seqlen_k_ = n->seqlen_k();
    headdim_ = n->headdim(); scale_ = n->scale();

    std::string arch = ctx.device().props().gcnArchName;
    if (auto p = arch.find(':'); p != std::string::npos) arch = arch.substr(0, p);

    kernel_ = wmma_attn::compile(seqlen_q_, seqlen_k_, headdim_, arch);
    if (!kernel_)
        throw std::runtime_error("WMMA attention compile failed");

    fprintf(stderr, "[WMMAAttn] compiled: B=%d H=%d Sq=%d Sk=%d D=%d\n",
            batch_, heads_, seqlen_q_, seqlen_k_, headdim_);
}

void WMMAAttentionOp::Execute(
    const InferenceRequestContext& ctx, Inputs inputs, Outputs outputs, const Workbuffers&) const
{
    wmma_attn::launch(*kernel_, ctx.getThreadContext().stream().get(),
                      inputs[0].get(), inputs[1].get(), inputs[2].get(), outputs[0].get(),
                      batch_, heads_, scale_);
}

OPERATION_REGISTER(WMMAAttentionOp, WMMAAttention);

}  // namespace rocm_gpu
}  // namespace ov
