#include "swin_fused_softmax_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "transformer/nodes/swin_fused_softmax_node.hpp"
#include "kernels/fused_reduce.hpp"

namespace ov {
namespace rocm_gpu {

SwinFusedSoftmaxOp::SwinFusedSoftmaxOp(
        const CreationContext& ctx,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputs, IndexCollection&& outputs)
    : OperationBase(ctx, node, std::move(inputs), std::move(outputs)) {
    auto sfs = std::dynamic_pointer_cast<nodes::SwinFusedSoftmax>(node);
    OPENVINO_ASSERT(sfs, "Expected SwinFusedSoftmax node");
    B_  = sfs->get_B();
    H_  = sfs->get_H();
    sq_ = sfs->get_sq();
    sk_ = sfs->get_sk();
    has_bias_ = sfs->get_has_bias();
}

void SwinFusedSoftmaxOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    auto stream = ctx.getThreadContext().stream().get();

    if (has_bias_) {
        OPENVINO_ASSERT(inputs.size() == 2);
        kernel::launch_swin_scale_bias_softmax(
            stream,
            inputs[0].get(),  // scores [B,H,sq,sk]
            inputs[1].get(),  // bias [H,sq,sk]
            nullptr,          // mask (not used)
            outputs[0].get(), // output [B,H,sq,sk]
            B_, H_, sq_, sk_,
            nullptr);         // temp scale (not used)
    } else {
        OPENVINO_ASSERT(inputs.size() == 1);
        size_t total_rows = (size_t)B_ * H_ * sq_;
        kernel::launch_plain_softmax(
            stream,
            inputs[0].get(),
            outputs[0].get(),
            total_rows, sk_);
    }
}

OPERATION_REGISTER(SwinFusedSoftmaxOp, SwinFusedSoftmax);

}  // namespace rocm_gpu
}  // namespace ov
