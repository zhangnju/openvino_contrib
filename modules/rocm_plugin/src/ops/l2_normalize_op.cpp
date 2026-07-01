#include "l2_normalize_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "transformer/nodes/l2_normalize_node.hpp"
#include "kernels/l2_normalize.hpp"

namespace ov {
namespace rocm_gpu {

L2NormalizeOp::L2NormalizeOp(
        const CreationContext& ctx,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputs, IndexCollection&& outputs)
    : OperationBase(ctx, node, std::move(inputs), std::move(outputs)) {
    auto ln = std::dynamic_pointer_cast<nodes::L2Normalize>(node);
    OPENVINO_ASSERT(ln, "Expected L2Normalize node");

    auto shape = node->get_output_shape(0);
    int axis = ln->get_axis();
    if (axis < 0) axis += shape.size();

    cols_ = shape[axis];
    rows_ = 1;
    for (int i = 0; i < axis; i++) rows_ *= shape[i];
    eps_ = ln->get_eps();
}

void L2NormalizeOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    kernel::launchL2Normalize(
        ctx.getThreadContext().stream().get(),
        inputs[0].get(), outputs[0].get(),
        rows_, cols_, eps_);
}

OPERATION_REGISTER(L2NormalizeOp, L2Normalize);

}  // namespace rocm_gpu
}  // namespace ov
