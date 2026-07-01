#pragma once
#include <rocm_operation_base.hpp>

namespace ov {
namespace rocm_gpu {

class SwinFlashAttentionOp : public OperationBase {
public:
    SwinFlashAttentionOp(const CreationContext& ctx,
                         const std::shared_ptr<ov::Node>& node,
                         IndexCollection&& inputs, IndexCollection&& outputs);
    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs, const Workbuffers&) const override;
    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }
private:
    int nW_, H_, sq_, sk_, hd_;
    bool has_bias_;
    float scale_;
};

}  // namespace rocm_gpu
}  // namespace ov
