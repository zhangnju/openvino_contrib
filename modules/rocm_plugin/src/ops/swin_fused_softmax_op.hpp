#pragma once
#include <rocm_operation_base.hpp>

namespace ov {
namespace rocm_gpu {

class SwinFusedSoftmaxOp : public OperationBase {
public:
    SwinFusedSoftmaxOp(const CreationContext& ctx,
                       const std::shared_ptr<ov::Node>& node,
                       IndexCollection&& inputs, IndexCollection&& outputs);
    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs, const Workbuffers&) const override;
    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }
private:
    int B_, H_, sq_, sk_;
    bool has_bias_;
};

}  // namespace rocm_gpu
}  // namespace ov
