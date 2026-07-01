#pragma once
#include <rocm_operation_base.hpp>
#include <vector>

namespace ov {
namespace rocm_gpu {

class RollOp : public OperationBase {
public:
    RollOp(const CreationContext& ctx, const std::shared_ptr<ov::Node>& node,
           IndexCollection&& inputs, IndexCollection&& outputs);
    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs, const Workbuffers&) const override;
    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }
private:
    std::vector<size_t> shape_;
    std::vector<int64_t> shift_, axes_;
};

}  // namespace rocm_gpu
}  // namespace ov
