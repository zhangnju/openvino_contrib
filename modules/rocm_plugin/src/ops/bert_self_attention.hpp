// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <rocm_operation_base.hpp>
#include <transformer/nodes/bert_self_attention_node.hpp>
#include <string>

namespace ov {
namespace rocm_gpu {

class BertSelfAttentionOp : public OperationBase {
public:
    BertSelfAttentionOp(const CreationContext& context,
                        const std::shared_ptr<ov::Node>& node,
                        IndexCollection&& inputIds,
                        IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers&) const override;

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

    WorkbufferRequest GetWorkBufferRequest() const override {
        // Workbuffer for intermediate QK^T softmax result if needed
        return {};
    }

private:
    // Compiled kernel info
    std::vector<char>  hsaco_;
    std::string        kernel_name_{"rock_attention"};
    hipFunction_t      func_{nullptr};
    hipModule_t        module_{nullptr};

    // Attention params
    int64_t seq_len_      = 0;
    int64_t num_heads_    = 0;
    int64_t head_dim_     = 0;
    bool    combined_qkv_ = false;  // true when using FusedQKVProjection (ptr+offset, no D2D copy)

    // Grid/block dimensions (from compiled kernel metadata)
    unsigned grid_x_  = 0;
    unsigned block_x_ = 0;
};

}  // namespace rocm_gpu
}  // namespace ov
