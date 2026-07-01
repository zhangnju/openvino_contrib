// Zero-Constant Propagation pass for OV ROCm Plugin.
//
// Algebraic simplification similar to MIGraphX's find_zero_ops + find_unit_ops:
//   - Multiply(x, 0) → 0
//   - Multiply(0, x) → 0
//   - Add(x, 0) → x
//   - Add(0, x) → x
//   - MVN with gamma=0 and beta=0 → output is 0
//
// After this pass, dead code (attention blocks whose output is zero and gets
// added to a residual skip connection) becomes unreachable from Result nodes
// and can be eliminated by NopElimination or RocmDCE.
#pragma once
#include <openvino/pass/pass.hpp>

namespace ov {
namespace rocm_gpu {
namespace pass {

class ZeroConstantPropagation : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ZeroConstantPropagation", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace rocm_gpu
}  // namespace ov
