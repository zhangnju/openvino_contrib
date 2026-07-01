#pragma once
#include <openvino/op/op.hpp>

namespace ov { namespace rocm_gpu { namespace nodes {

class FlashAttentionTriton : public ov::op::Op {
public:
    OPENVINO_OP("FlashAttentionTriton", "rocm_gpu");
    FlashAttentionTriton() = default;

    FlashAttentionTriton(const ov::Output<ov::Node>& Q,
                         const ov::Output<ov::Node>& K,
                         const ov::Output<ov::Node>& V,
                         int batch, int heads, int seqlen_q, int seqlen_k, int headdim, float scale)
        : Op({Q, K, V}), batch_(batch), heads_(heads),
          seqlen_q_(seqlen_q), seqlen_k_(seqlen_k),
          headdim_(headdim), scale_(scale) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<FlashAttentionTriton>(
            inputs[0], inputs[1], inputs[2],
            batch_, heads_, seqlen_q_, seqlen_k_, headdim_, scale_);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("batch", batch_);
        visitor.on_attribute("heads", heads_);
        visitor.on_attribute("seqlen_q", seqlen_q_);
        visitor.on_attribute("seqlen_k", seqlen_k_);
        visitor.on_attribute("headdim", headdim_);
        visitor.on_attribute("scale", scale_);
        return true;
    }

    int batch() const { return batch_; }
    int heads() const { return heads_; }
    int seqlen_q() const { return seqlen_q_; }
    int seqlen_k() const { return seqlen_k_; }
    int headdim() const { return headdim_; }
    float scale() const { return scale_; }

private:
    int batch_ = 1, heads_ = 1;
    int seqlen_q_ = 0, seqlen_k_ = 0;
    int headdim_ = 32;
    float scale_ = 0.1767f;
};

}}} // namespace
