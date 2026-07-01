// FusedMVNEpilogue: MVN + Scale(gamma) + Bias(beta) in one kernel.
// Replaces OV's decomposed LayerNorm: ReduceMean→Sub→Pow→ReduceMean→Add(eps)→Sqrt→Div→Mul(gamma)→Add(beta)
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class FusedMVNEpilogue : public ov::op::Op {
public:
    OPENVINO_OP("FusedMVNEpilogue", "rocm_gpu");
    FusedMVNEpilogue() = default;

    // x [rows, cols], gamma [cols], beta [cols] → y [rows, cols]
    FusedMVNEpilogue(const ov::Output<ov::Node>& x,
                     const ov::Output<ov::Node>& gamma,
                     const ov::Output<ov::Node>& beta,
                     int rows, int cols, float eps)
        : Op({x, gamma, beta}), rows_(rows), cols_(cols), eps_(eps) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<FusedMVNEpilogue>(inputs[0], inputs[1], inputs[2], rows_, cols_, eps_);
    }

    bool visit_attributes(ov::AttributeVisitor& v) override {
        v.on_attribute("rows", rows_); v.on_attribute("cols", cols_); v.on_attribute("eps", eps_);
        return true;
    }

    int get_rows() const { return rows_; }
    int get_cols() const { return cols_; }
    float get_eps() const { return eps_; }

private:
    int rows_ = 0, cols_ = 0;
    float eps_ = 1e-5f;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
