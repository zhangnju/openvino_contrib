#pragma once
#include <rocm_operation_base.hpp>
#include <hip/hip_runtime.h>
#include <vector>

namespace ov {
namespace rocm_gpu {

class FusedRTDOp : public OperationBase {
public:
    FusedRTDOp(const CreationContext& ctx,
               const std::shared_ptr<ov::Node>& node,
               IndexCollection&& inputs, IndexCollection&& outputs);
    ~FusedRTDOp();

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs, const Workbuffers&) const override;
    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

private:
    hipModule_t module_ = nullptr;
    hipFunction_t func_ = nullptr;
    unsigned grid_x_ = 0, block_x_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
