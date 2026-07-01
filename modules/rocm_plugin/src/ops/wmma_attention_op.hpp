#pragma once
#include <rocm_operation_base.hpp>
#include "rocm/wmma_attention_generic.hpp"

namespace ov { namespace rocm_gpu {

class WMMAAttentionOp : public OperationBase {
public:
    WMMAAttentionOp(const CreationContext& ctx, const std::shared_ptr<ov::Node>& node,
                    IndexCollection&& in, IndexCollection&& out);
    void Execute(const InferenceRequestContext& ctx, Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;
    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }
    WorkbufferRequest GetWorkBufferRequest() const override { return {}; }
private:
    int batch_, heads_, seqlen_q_, seqlen_k_, headdim_;
    float scale_;
    std::shared_ptr<wmma_attn::WMMAAttnKernel> kernel_;
};

}} // namespace
