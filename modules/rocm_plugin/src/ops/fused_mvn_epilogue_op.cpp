#include "fused_mvn_epilogue_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "transformer/nodes/fused_mvn_epilogue_node.hpp"
#include "kernels/fused_reduce.hpp"
#include "kernels/mvn_fast.hpp"

namespace ov {
namespace rocm_gpu {

FusedMVNEpilogueOp::FusedMVNEpilogueOp(
        const CreationContext& ctx,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputs, IndexCollection&& outputs)
    : OperationBase(ctx, node, std::move(inputs), std::move(outputs)) {
    auto fme = std::dynamic_pointer_cast<nodes::FusedMVNEpilogue>(node);
    OPENVINO_ASSERT(fme, "Expected FusedMVNEpilogue node");
    rows_ = fme->get_rows();
    cols_ = fme->get_cols();
    eps_  = fme->get_eps();
}

void FusedMVNEpilogueOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    kernel::launchMvnFastEpi(
        ctx.getThreadContext().stream().get(),
        inputs[0].get(),   // x
        outputs[0].get(),  // y
        inputs[1].get(),   // scale (gamma)
        inputs[2].get(),   // bias (beta)
        nullptr,           // residual (none)
        rows_, cols_, eps_);
}

OPERATION_REGISTER(FusedMVNEpilogueOp, FusedMVNEpilogue);

}  // namespace rocm_gpu
}  // namespace ov
