// SwinFlashAttention: fuses QK MatMul + [bias_add] + Softmax + PV MatMul
// into a single WMMA kernel for small attention (sq,sk <= 64, hd <= 64).
// Supports optional relative position bias and L2-normalized cosine attention.
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class SwinFlashAttention : public ov::op::Op {
public:
    OPENVINO_OP("SwinFlashAttention", "rocm_gpu");
    SwinFlashAttention() = default;

    // Standard: Q, K, V, [bias]
    SwinFlashAttention(const ov::OutputVector& args,
                       int nW, int H, int sq, int sk, int hd,
                       bool has_bias, float scale)
        : Op(args), nW_(nW), H_(H), sq_(sq), sk_(sk), hd_(hd),
          has_bias_(has_bias), scale_(scale) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0),
                        ov::PartialShape{nW_, H_, sq_, hd_});
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<SwinFlashAttention>(inputs, nW_, H_, sq_, sk_, hd_,
                                                    has_bias_, scale_);
    }

    bool visit_attributes(ov::AttributeVisitor& v) override {
        v.on_attribute("nW", nW_); v.on_attribute("H", H_);
        v.on_attribute("sq", sq_); v.on_attribute("sk", sk_);
        v.on_attribute("hd", hd_);
        v.on_attribute("has_bias", has_bias_);
        v.on_attribute("scale", scale_);
        return true;
    }

    int get_nW() const { return nW_; }
    int get_H() const { return H_; }
    int get_sq() const { return sq_; }
    int get_sk() const { return sk_; }
    int get_hd() const { return hd_; }
    bool get_has_bias() const { return has_bias_; }
    float get_scale() const { return scale_; }

private:
    int nW_ = 0, H_ = 0, sq_ = 0, sk_ = 0, hd_ = 0;
    bool has_bias_ = false;
    float scale_ = 1.0f;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
