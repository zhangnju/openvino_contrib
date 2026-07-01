#include "roll_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "transformer/nodes/roll_node.hpp"
#include "kernels/roll.hpp"

namespace ov {
namespace rocm_gpu {

RollOp::RollOp(const CreationContext& ctx, const std::shared_ptr<ov::Node>& node,
               IndexCollection&& inputs, IndexCollection&& outputs)
    : OperationBase(ctx, node, std::move(inputs), std::move(outputs)) {
    auto r = std::dynamic_pointer_cast<nodes::Roll>(node);
    OPENVINO_ASSERT(r, "Expected Roll node");
    auto s = node->get_output_shape(0);
    shape_.assign(s.begin(), s.end());
    shift_ = r->get_shift();
    axes_ = r->get_axes();
}

void RollOp::Execute(const InferenceRequestContext& ctx,
                     Inputs inputs, Outputs outputs, const Workbuffers&) const {
    kernel::launchRoll(ctx.getThreadContext().stream().get(),
                       inputs[0].get(), outputs[0].get(),
                       shape_, shift_, axes_);
}

OPERATION_REGISTER(RollOp, Roll);

}  // namespace rocm_gpu
}  // namespace ov
