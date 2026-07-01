#pragma once
#include <rocm_operation_base.hpp>
#include "kernels/qkv_transpose_split.hpp"

namespace ov {
namespace rocm_gpu {

class QKVTransposeSplitOp : public OperationBase {
public:
    QKVTransposeSplitOp(const CreationContext& ctx,
                        const std::shared_ptr<ov::Node>& node,
                        IndexCollection&& inputs, IndexCollection&& outputs);

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

private:
    int nW_, sq_, H_, hd_;
    bool is_fp16_, norm_q_, norm_k_;
    float eps_;
};

}  // namespace rocm_gpu
}  // namespace ov
