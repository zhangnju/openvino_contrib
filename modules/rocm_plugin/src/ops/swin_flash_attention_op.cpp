#include "swin_flash_attention_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "transformer/nodes/swin_flash_attention_node.hpp"
#include "kernels/swin_flash_attention.hpp"

namespace ov {
namespace rocm_gpu {

SwinFlashAttentionOp::SwinFlashAttentionOp(
        const CreationContext& ctx,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputs, IndexCollection&& outputs)
    : OperationBase(ctx, node, std::move(inputs), std::move(outputs)) {
    auto sfa = std::dynamic_pointer_cast<nodes::SwinFlashAttention>(node);
    OPENVINO_ASSERT(sfa, "Expected SwinFlashAttention node");
    nW_  = sfa->get_nW();
    H_   = sfa->get_H();
    sq_  = sfa->get_sq();
    sk_  = sfa->get_sk();
    hd_  = sfa->get_hd();
    has_bias_ = sfa->get_has_bias();
    scale_ = sfa->get_scale();

    fprintf(stderr, "[SwinFlashAttn] nW=%d H=%d sq=%d sk=%d hd=%d bias=%d scale=%.4f\n",
            nW_, H_, sq_, sk_, hd_, has_bias_, scale_);
}

void SwinFlashAttentionOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    // inputs: [0]=Q, [1]=K, [2]=V, [3]=bias (optional)
    const void* bias = has_bias_ ? inputs[3].get() : nullptr;

    kernel::launchSwinFlashAttention(
        ctx.getThreadContext().stream().get(),
        inputs[0].get(), inputs[1].get(), inputs[2].get(),
        bias, outputs[0].get(),
        nW_, H_, sq_, sk_, hd_, scale_);
}

OPERATION_REGISTER(SwinFlashAttentionOp, SwinFlashAttention);

}  // namespace rocm_gpu
}  // namespace ov
