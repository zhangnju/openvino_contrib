// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RocmAttentionSoftmaxOp: replaces the scale+softmax between QKT and AV MatMuls
// with a MIGraphX-compiled fused kernel (convert_mul_reduce_max_sub_exp_reduce_sum_div_convert).
//
// The transformer pass marks eligible Softmax nodes with rt_info tags:
//   "rocm_attn_softmax"   = "1"
//   "rocm_attn_nheads"    = number of attention heads
//   "rocm_attn_seq"       = sequence length N (N×N attention matrix)
//   "rocm_attn_scale"     = scale factor (1/sqrt(dq)) as string

#pragma once

#include <rocm_operation_base.hpp>
#include <hip/hip_runtime.h>
#include <string>
#include <mutex>

namespace ov {
namespace rocm_gpu {

class RocmAttentionSoftmaxOp : public OperationBase {
public:
    RocmAttentionSoftmaxOp(const CreationContext& ctx,
                           const ::ov::Node& node,
                           IndexCollection&& inputIds,
                           IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override { return {{}, {}}; }

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

    static bool isEnabled();

private:
    hipModule_t   hip_module_ = nullptr;
    hipFunction_t hip_func_   = nullptr;
    int           grid_x_     = 1;
    int           block_x_    = 256;
};

}  // namespace rocm_gpu
}  // namespace ov
