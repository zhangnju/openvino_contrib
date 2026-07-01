#pragma once
#include <rocm_operation_base.hpp>
#include "rocm/triton_flash_attn.hpp"

namespace ov { namespace rocm_gpu {

class FlashAttentionTritonOp : public OperationBase {
public:
    FlashAttentionTritonOp(const CreationContext& ctx, const std::shared_ptr<ov::Node>& node,
                           IndexCollection&& in, IndexCollection&& out);
    void Execute(const InferenceRequestContext& ctx, Inputs inputs, Outputs outputs,
                 const Workbuffers& wb) const override;
    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::NONE; // hipStreamSync not graph-compatible
    }
    WorkbufferRequest GetWorkBufferRequest() const override;

private:
    int batch_, heads_, seqlen_q_, seqlen_k_, headdim_;
    float scale_;
    int sq_rounded_;
    std::shared_ptr<triton_fa::TritonFAKernel> kernel_;
};

}} // namespace
