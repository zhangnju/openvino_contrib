// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RocmAttentionMatMulOp: replaces slow rocBLAS MatMul for multi-head attention
// with MIGraphX-compiled MLIR kernels (mlir_reshape_slice_transpose_*_dot).
//
// The transformer pass marks eligible MatMul nodes with rt_info tags:
//   "rocm_attn_kind"   = "qkt" or "av"
//   "rocm_attn_nheads" = number of attention heads
//   "rocm_attn_dim_q"  = query/key dimension per head
//   "rocm_attn_dim_v"  = value dimension per head
//   "rocm_attn_seq"    = sequence length (H*W)
//   "rocm_attn_H"      = spatial height
//   "rocm_attn_W"      = spatial width
//   "rocm_attn_qkv"    = name of the pre-Reshape QKV tensor source node
//
// For QKT kind: factory replaces op with this impl, Execute calls qkt kernel
// For AV  kind: factory replaces op with this impl, Execute calls av kernel
// Both kernels receive the full QKV tensor as first arg (from rt_info stored node)

#pragma once

#include <rocm_operation_base.hpp>
#include <hip/hip_runtime.h>
#include <string>
#include <unordered_map>
#include <mutex>

namespace ov {
namespace rocm_gpu {

class RocmAttentionMatMulOp : public OperationBase {
public:
    RocmAttentionMatMulOp(const CreationContext& ctx,
                          const ::ov::Node& node,
                          IndexCollection&& inputIds,
                          IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& ctx,
                 Inputs inputs, Outputs outputs,
                 const Workbuffers&) const override;

    WorkbufferRequest GetWorkBufferRequest() const override {
        // pe kernel needs workspace for conv(V) intermediate result (pe_work)
        if (has_pe_conv_ && pe_workspace_bytes_ > 0)
            return {{}, {pe_workspace_bytes_}};
        return {{}, {}};
    }

    rocmGraphCompatibility GetrocmGraphCompatibility() const override {
        return rocmGraphCompatibility::FULL;
    }

    static bool isEnabled();

private:
    hipModule_t   hip_module_ = nullptr;
    hipFunction_t hip_func_   = nullptr;
    int           grid_x_     = 1;
    int           block_x_    = 256;
    std::string   kind_;       // "qkt" or "av"

    // pe(V) depthwise conv kernel: compiled when AV MatMul has a downstream pe+add
    // pe(V) = depthwise_conv3x3(V) + bias, then add to AV output
    hipModule_t   pe_module_   = nullptr;
    hipFunction_t pe_func_     = nullptr;
    int           pe_grid_x_   = 1;
    int           pe_block_x_  = 256;
    bool          has_pe_conv_ = false;
    size_t        pe_workspace_bytes_ = 0;  // bytes for conv(V) workspace
    int           pe_out_elems_ = 0;        // total fp16 elements = K*H*W
    size_t        pe_v_offset_bytes_ = 0;   // not used (kernel reads V via rock.transform)
    int           H_pe_ = 0, W_pe_ = 0;    // spatial dims for pe add kernel
};

}  // namespace rocm_gpu
}  // namespace ov
