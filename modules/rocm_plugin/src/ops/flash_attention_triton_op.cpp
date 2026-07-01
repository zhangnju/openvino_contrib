#include "flash_attention_triton_op.hpp"
#include "transformer/nodes/flash_attention_triton_node.hpp"
#include "rocm_creation_context.hpp"
#include <cmath>
#include <cstdio>
#include <mutex>
#include <rocm_operation_registry.hpp>

namespace ov { namespace rocm_gpu {

FlashAttentionTritonOp::FlashAttentionTritonOp(
    const CreationContext& ctx, const std::shared_ptr<ov::Node>& node,
    IndexCollection&& in, IndexCollection&& out)
    : OperationBase(ctx, node, std::move(in), std::move(out))
{
    auto fa = std::dynamic_pointer_cast<nodes::FlashAttentionTriton>(node);
    batch_ = fa->batch();
    heads_ = fa->heads();
    seqlen_q_ = fa->seqlen_q();
    seqlen_k_ = fa->seqlen_k();
    headdim_ = fa->headdim();
    scale_ = fa->scale();
    sq_rounded_ = ((seqlen_q_ + 127) / 128) * 128;

    std::string arch = ctx.device().props().gcnArchName;
    if (auto p = arch.find(':'); p != std::string::npos) arch = arch.substr(0, p);

    if (std::getenv("ROCM_DISABLE_TRITON_FA"))
        throw std::runtime_error("Triton FA disabled");

    kernel_ = triton_fa::compile(seqlen_q_, seqlen_k_, headdim_, arch);
    if (!kernel_)
        throw std::runtime_error("Triton FA AOT compile failed");

    fprintf(stderr, "[FlashAttnTriton] compiled: B=%d H=%d Sq=%d Sk=%d D=%d shared=%d\n",
            batch_, heads_, seqlen_q_, seqlen_k_, headdim_, kernel_->meta.shared_mem);
}

WorkbufferRequest FlashAttentionTritonOp::GetWorkBufferRequest() const {
    size_t lse_bytes = (size_t)batch_ * heads_ * sq_rounded_ * sizeof(float);
    return {{}, {lse_bytes, lse_bytes, triton_fa::kFAArgsSize}};
}

void FlashAttentionTritonOp::Execute(
    const InferenceRequestContext& ctx,
    Inputs inputs, Outputs outputs, const Workbuffers& wb) const
{
    const auto& stream = ctx.getThreadContext().stream();
    void* Lse = const_cast<void*>(wb.mutable_buffers.at(0).get());
    void* TMP = const_cast<void*>(wb.mutable_buffers.at(1).get());
    void* args_buf = const_cast<void*>(wb.mutable_buffers.at(2).get());

    // Q/K/V: [B, S, H, D] contiguous
    int stride_qb = seqlen_q_ * heads_ * headdim_;
    int stride_qm = heads_ * headdim_;
    int stride_qh = headdim_;
    int stride_kb = seqlen_k_ * heads_ * headdim_;
    int stride_kn = heads_ * headdim_;
    int stride_kh = headdim_;

    // Only serialize large FA kernels that can deadlock in multi-stream mode.
    // Small FA (seq_len <= 4096) finishes in <1ms — hipStreamSynchronize returns
    // quickly and concurrent streams don't contend. Large FA (petr_large seq=41760)
    // takes ~100ms and causes GPU hang when multiple streams sync simultaneously.
    static std::mutex fa_mutex;
    const bool needs_serialize = (seqlen_q_ > 4096 || seqlen_k_ > 4096);
    std::unique_lock<std::mutex> fa_lock(fa_mutex, std::defer_lock);
    if (needs_serialize) fa_lock.lock();

    // Drain this stream's prior work before launching Triton kernel.
    // Required on gfx1100: hipModuleLaunchKernel (Triton HSACO) does not
    // respect same-stream ordering with hipLaunchKernelGGL (OV static kernels).
    hipStreamSynchronize(stream.get());
    triton_fa::launch(*kernel_, stream.get(), args_buf,
                      inputs[0].get(), inputs[1].get(), inputs[2].get(),
                      nullptr, outputs[0].get(), Lse, TMP,
                      batch_, heads_, seqlen_q_, seqlen_k_, headdim_, scale_,
                      stride_qb, stride_qh, stride_qm,
                      stride_kb, stride_kh, stride_kn,
                      stride_kb, stride_kh, stride_kn,
                      stride_qb, stride_qh, stride_qm);
    if (needs_serialize) {
        hipStreamSynchronize(stream.get());
    }
}

OPERATION_REGISTER(FlashAttentionTritonOp, FlashAttentionTriton);

}  // namespace rocm_gpu
}  // namespace ov
