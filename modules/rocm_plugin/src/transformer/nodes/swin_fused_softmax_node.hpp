// SwinFusedSoftmax: fuses Add(scores, bias) + Softmax into one kernel.
// Pattern 1 (with bias): scores [B,H,sq,sk] + bias [H,sq,sk] → Softmax
// Pattern 2 (no bias):   scores [B,H,sq,sk] → Softmax (just optimized softmax)
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class SwinFusedSoftmax : public ov::op::Op {
public:
    OPENVINO_OP("SwinFusedSoftmax", "rocm_gpu");
    SwinFusedSoftmax() = default;

    // With bias: scores + bias → softmax
    SwinFusedSoftmax(const ov::Output<ov::Node>& scores,
                     const ov::Output<ov::Node>& bias,
                     int B, int H, int sq, int sk)
        : Op({scores, bias}), B_(B), H_(H), sq_(sq), sk_(sk), has_bias_(true) {
        constructor_validate_and_infer_types();
    }

    // Without bias: scores → softmax
    SwinFusedSoftmax(const ov::Output<ov::Node>& scores,
                     int B, int H, int sq, int sk)
        : Op({scores}), B_(B), H_(H), sq_(sq), sk_(sk), has_bias_(false) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0),
                        ov::PartialShape{B_, H_, sq_, sk_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        if (has_bias_)
            return std::make_shared<SwinFusedSoftmax>(inputs[0], inputs[1], B_, H_, sq_, sk_);
        return std::make_shared<SwinFusedSoftmax>(inputs[0], B_, H_, sq_, sk_);
    }

    bool visit_attributes(ov::AttributeVisitor& v) override {
        v.on_attribute("B", B_); v.on_attribute("H", H_);
        v.on_attribute("sq", sq_); v.on_attribute("sk", sk_);
        v.on_attribute("has_bias", has_bias_);
        return true;
    }

    int get_B() const { return B_; }
    int get_H() const { return H_; }
    int get_sq() const { return sq_; }
    int get_sk() const { return sk_; }
    bool get_has_bias() const { return has_bias_; }

private:
    int B_ = 0, H_ = 0, sq_ = 0, sk_ = 0;
    bool has_bias_ = false;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
