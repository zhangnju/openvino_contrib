// QKVTransposeSplit: fuses Reshape + Transpose + 3×Gather for Swin attention QKV unbind.
// Pattern: x [nW, sq, 3*H*hd] → Reshape [nW, sq, 3, H, hd] → Transpose [3, nW, H, sq, hd]
//          → Gather(0) → Q [nW, H, sq, hd]
//          → Gather(1) → K [nW, H, sq, hd]
//          → Gather(2) → V [nW, H, sq, hd]
// Fused: single kernel reads x with transposed indexing, writes Q/K/V directly.
#pragma once
#include <openvino/op/op.hpp>

namespace ov {
namespace rocm_gpu {
namespace nodes {

class QKVTransposeSplit : public ov::op::Op {
public:
    OPENVINO_OP("QKVTransposeSplit", "rocm_gpu");

    QKVTransposeSplit() = default;

    QKVTransposeSplit(const ov::Output<ov::Node>& qkv_input,
                      int nW, int sq, int H, int hd, bool is_fp16,
                      bool norm_q = false, bool norm_k = false, float eps = 1e-12f)
        : Op({qkv_input}), nW_(nW), sq_(sq), H_(H), hd_(hd), is_fp16_(is_fp16),
          norm_q_(norm_q), norm_k_(norm_k), eps_(eps) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto et = get_input_element_type(0);
        auto out_shape = ov::PartialShape{nW_, H_, sq_, hd_};
        set_output_type(0, et, out_shape);  // Q
        set_output_type(1, et, out_shape);  // K
        set_output_type(2, et, out_shape);  // V
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<QKVTransposeSplit>(inputs[0], nW_, sq_, H_, hd_, is_fp16_,
                                                   norm_q_, norm_k_, eps_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("nW", nW_);
        visitor.on_attribute("sq", sq_);
        visitor.on_attribute("H", H_);
        visitor.on_attribute("hd", hd_);
        visitor.on_attribute("is_fp16", is_fp16_);
        visitor.on_attribute("norm_q", norm_q_);
        visitor.on_attribute("norm_k", norm_k_);
        visitor.on_attribute("eps", eps_);
        return true;
    }

    int get_nW()  const { return nW_; }
    int get_sq()  const { return sq_; }
    int get_H()   const { return H_; }
    int get_hd()  const { return hd_; }
    bool get_fp16() const { return is_fp16_; }
    bool get_norm_q() const { return norm_q_; }
    bool get_norm_k() const { return norm_k_; }
    float get_eps() const { return eps_; }

private:
    int nW_ = 0, sq_ = 0, H_ = 0, hd_ = 0;
    bool is_fp16_ = true;
    bool norm_q_ = false, norm_k_ = false;
    float eps_ = 1e-12f;
};

}  // namespace nodes
}  // namespace rocm_gpu
}  // namespace ov
