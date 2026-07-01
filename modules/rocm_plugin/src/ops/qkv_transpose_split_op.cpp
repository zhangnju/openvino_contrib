#include "qkv_transpose_split_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "transformer/nodes/qkv_transpose_split_node.hpp"

namespace ov {
namespace rocm_gpu {

QKVTransposeSplitOp::QKVTransposeSplitOp(
        const CreationContext& ctx,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputs, IndexCollection&& outputs)
    : OperationBase(ctx, node, std::move(inputs), std::move(outputs)) {
    auto qkv = std::dynamic_pointer_cast<nodes::QKVTransposeSplit>(node);
    OPENVINO_ASSERT(qkv, "Expected QKVTransposeSplit node");
    nW_  = qkv->get_nW();
    sq_  = qkv->get_sq();
    H_   = qkv->get_H();
    hd_  = qkv->get_hd();
    is_fp16_ = qkv->get_fp16();
    norm_q_  = qkv->get_norm_q();
    norm_k_  = qkv->get_norm_k();
    eps_     = qkv->get_eps();
}

void QKVTransposeSplitOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 3);
    kernel::launchQKVTransposeSplit(
        ctx.getThreadContext().stream().get(),
        inputs[0].get(), outputs[0].get(), outputs[1].get(), outputs[2].get(),
        nW_, sq_, H_, hd_, is_fp16_, norm_q_, norm_k_, eps_);
}

OPERATION_REGISTER(QKVTransposeSplitOp, QKVTransposeSplit);

}  // namespace rocm_gpu
}  // namespace ov
