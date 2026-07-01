// FusedRTD: MatMul with optional input/output Reshape+Transpose+Reshape.
// Folds window-partition/reverse transpose into the GEMM kernel via rocMLIR.
//
// Input-side pattern:  Reshape(in_pre→in_r1) → Transpose(in_perm) → Reshape(in_r2) → MatMul
// Output-side pattern: MatMul → Reshape(out_r1) → Transpose(out_perm) → Reshape(out_r2)
// Both sides are optional; compiled as mlir_reshape_transpose_reshape_dot.
#pragma once
#include <openvino/op/op.hpp>
#include <vector>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class FusedReshapeTransposeDot : public ov::op::Op {
public:
    OPENVINO_OP("FusedReshapeTransposeDot", "rocm_gpu");
    FusedReshapeTransposeDot() = default;

    // Full constructor: input-side + output-side transforms
    FusedReshapeTransposeDot(const ov::Output<ov::Node>& A,
                             const ov::Output<ov::Node>& B,
                             bool transpose_b,
                             // input-side (empty = no input transform)
                             const std::vector<int64_t>& in_pre,
                             const std::vector<int64_t>& in_r1,
                             const std::vector<int64_t>& in_perm,
                             const std::vector<int64_t>& in_r2,
                             // output-side
                             const std::vector<int64_t>& out_r1,
                             const std::vector<int64_t>& out_perm,
                             const std::vector<int64_t>& out_r2,
                             const ov::Shape& output_shape)
        : Op({A, B}), transpose_b_(transpose_b),
          in_pre_(in_pre), in_r1_(in_r1), in_perm_(in_perm), in_r2_(in_r2),
          out_r1_(out_r1), out_perm_(out_perm), out_r2_(out_r2),
          output_shape_(output_shape) {
        constructor_validate_and_infer_types();
    }

    // Legacy: output-side only (backward compat)
    FusedReshapeTransposeDot(const ov::Output<ov::Node>& A,
                             const ov::Output<ov::Node>& B,
                             bool transpose_b,
                             const std::vector<int64_t>& out_r1,
                             const std::vector<int64_t>& out_perm,
                             const std::vector<int64_t>& out_r2,
                             const ov::Shape& output_shape)
        : Op({A, B}), transpose_b_(transpose_b),
          out_r1_(out_r1), out_perm_(out_perm), out_r2_(out_r2),
          output_shape_(output_shape) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0),
                        ov::PartialShape(output_shape_));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<FusedReshapeTransposeDot>(
            inputs[0], inputs[1], transpose_b_,
            in_pre_, in_r1_, in_perm_, in_r2_,
            out_r1_, out_perm_, out_r2_, output_shape_);
    }

    bool visit_attributes(ov::AttributeVisitor& v) override {
        v.on_attribute("transpose_b", transpose_b_);
        v.on_attribute("in_pre", in_pre_);
        v.on_attribute("in_r1", in_r1_);
        v.on_attribute("in_perm", in_perm_);
        v.on_attribute("in_r2", in_r2_);
        v.on_attribute("out_r1", out_r1_);
        v.on_attribute("out_perm", out_perm_);
        v.on_attribute("out_r2", out_r2_);
        return true;
    }

    bool get_transpose_b() const { return transpose_b_; }
    bool has_input_transform() const { return !in_perm_.empty(); }
    const std::vector<int64_t>& get_in_pre() const { return in_pre_; }
    const std::vector<int64_t>& get_in_r1() const { return in_r1_; }
    const std::vector<int64_t>& get_in_perm() const { return in_perm_; }
    const std::vector<int64_t>& get_in_r2() const { return in_r2_; }
    const std::vector<int64_t>& get_out_r1() const { return out_r1_; }
    const std::vector<int64_t>& get_out_perm() const { return out_perm_; }
    const std::vector<int64_t>& get_out_r2() const { return out_r2_; }
    const ov::Shape& get_output_shape_val() const { return output_shape_; }

private:
    bool transpose_b_ = false;
    std::vector<int64_t> in_pre_, in_r1_, in_perm_, in_r2_;
    std::vector<int64_t> out_r1_, out_perm_, out_r2_;
    ov::Shape output_shape_;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
