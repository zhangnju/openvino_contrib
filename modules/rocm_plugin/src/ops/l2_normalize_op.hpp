#pragma once
#include <rocm_operation_base.hpp>

namespace ov {
namespace rocm_gpu {

class L2NormalizeOp : public OperationBase {
public:
    L2NormalizeOp(const CreationContext& ctx,
                  const std::shared_ptr<ov::Node>& node,
                  IndexCollection&& inputs, IndexCollection&& outputs);
    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs, const Workbuffers&) const override;
    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }
private:
    int rows_, cols_;
    float eps_;
};

}  // namespace rocm_gpu
}  // namespace ov
