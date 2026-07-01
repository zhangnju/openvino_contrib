// L2Normalize: fuses Power(2) → ReduceSum(axis) → Sqrt → Maximum(eps) → Divide
// into a single warp-per-row kernel.
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class L2Normalize : public ov::op::Op {
public:
    OPENVINO_OP("L2Normalize", "rocm_gpu");
    L2Normalize() = default;

    L2Normalize(const ov::Output<ov::Node>& input, int axis, float eps)
        : Op({input}), axis_(axis), eps_(eps) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<L2Normalize>(inputs[0], axis_, eps_);
    }

    bool visit_attributes(ov::AttributeVisitor& v) override {
        v.on_attribute("axis", axis_);
        v.on_attribute("eps", eps_);
        return true;
    }

    int get_axis() const { return axis_; }
    float get_eps() const { return eps_; }

private:
    int axis_ = -1;
    float eps_ = 1e-12f;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
