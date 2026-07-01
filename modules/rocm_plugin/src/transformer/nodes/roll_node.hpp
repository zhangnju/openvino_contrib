// Roll: circular shift along one or more axes. Replaces Slice+Concat pattern.
#pragma once
#include <openvino/op/op.hpp>
#include <vector>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class Roll : public ov::op::Op {
public:
    OPENVINO_OP("Roll", "rocm_gpu");
    Roll() = default;

    Roll(const ov::Output<ov::Node>& input,
         const std::vector<int64_t>& shift,
         const std::vector<int64_t>& axes)
        : Op({input}), shift_(shift), axes_(axes) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<Roll>(inputs[0], shift_, axes_);
    }

    bool visit_attributes(ov::AttributeVisitor& v) override {
        v.on_attribute("shift", shift_);
        v.on_attribute("axes", axes_);
        return true;
    }

    const std::vector<int64_t>& get_shift() const { return shift_; }
    const std::vector<int64_t>& get_axes() const { return axes_; }

private:
    std::vector<int64_t> shift_;
    std::vector<int64_t> axes_;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
