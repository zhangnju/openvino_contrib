// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// FusedFCGELUOp: hipBLASLt GEMM+bias+GELU single kernel (tanh GELU approximation).
// Falls back to rocBLAS GEMM + native HIP bias+GELU if hipBLASLt unavailable.

#include "fused_fc_gelu_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include "kernels/fused_reduce.hpp"
#include "rocm/rocmlir_gemm.hpp"
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <fstream>
#include <sys/stat.h>
#include <fmt/format.h>

namespace ov {
namespace rocm_gpu {

namespace {

static std::string gelu_cache_dir(const std::string& arch) {
    const char* h = std::getenv("HOME");
    return std::string(h ? h : "/tmp") + "/.cache/ov_hipblaslt_" + arch;
}

static std::string gelu_algo_path(const std::string& arch, int M, int N, int K) {
    return fmt::format("{}/fcgelu_{}_{}_{}.algo", gelu_cache_dir(arch), M, N, K);
}

static bool load_algo(const std::string& arch, int M, int N, int K, hipblasLtMatmulAlgo_t& a) {
    std::ifstream f(gelu_algo_path(arch, M, N, K), std::ios::binary);
    return f && static_cast<bool>(f.read(reinterpret_cast<char*>(&a), sizeof(a)));
}

static void save_algo(const std::string& arch, int M, int N, int K, const hipblasLtMatmulAlgo_t& a) {
    mkdir(gelu_cache_dir(arch).c_str(), 0755);
    std::ofstream f(gelu_algo_path(arch, M, N, K), std::ios::binary);
    if (f) f.write(reinterpret_cast<const char*>(&a), sizeof(a));
}

static hipblasLtMatmulAlgo_t tune_gelu_algo(
        hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
        hipblasLtMatrixLayout_t lW, hipblasLtMatrixLayout_t lX, hipblasLtMatrixLayout_t lD,
        const void* W, const void* X, void* bias, void* D,
        void* ws, size_t ws_bytes, hipStream_t stream) {
    hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));
    hipblasLtMatmulHeuristicResult_t cands[50]; int nf = 0;
    hipblasLtMatmulAlgoGetHeuristic(handle, desc, lW, lX, lD, lD, pref, 50, cands, &nf);
    hipblasLtMatmulPreferenceDestroy(pref);
    float alpha = 1.f, beta = 0.f;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*));
    auto run = [&](const hipblasLtMatmulAlgo_t& a) {
        hipblasLtMatmul(handle, desc, &alpha, W, lW, X, lX, &beta, D, lD, D, lD, &a, ws, ws_bytes, stream);
    };
    float best = 1e9f; int bi = 0;
    hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i = 0; i < nf; ++i) {
        for (int w = 0; w < 5; ++w) run(cands[i].algo);
        hipStreamSynchronize(stream);
        hipEventRecord(t0, stream);
        for (int r = 0; r < 20; ++r) run(cands[i].algo);
        hipEventRecord(t1, stream); hipStreamSynchronize(stream);
        float ms; hipEventElapsedTime(&ms, t0, t1); ms /= 20;
        if (ms < best) { best = ms; bi = i; }
    }
    hipEventDestroy(t0); hipEventDestroy(t1);
    fprintf(stderr, "[FCGelu-tune] best algo idx=%d (%.3f ms)\n", bi, best);
    return cands[bi].algo;
}

}  // namespace

FusedFCGELUOp::FusedFCGELUOp(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {

    auto fc_gelu = std::dynamic_pointer_cast<nodes::FusedFCGELU>(node);
    OPENVINO_ASSERT(fc_gelu, "FusedFCGELUOp: expected FusedFCGELU node");
    seq_     = static_cast<int>(fc_gelu->get_seq());
    in_dim_  = static_cast<int>(fc_gelu->get_in_dim());
    out_dim_ = static_cast<int>(fc_gelu->get_out_dim());

    const auto& props = context.device().props();
    arch_ = props.gcnArchName;
    if (auto p = arch_.find(':'); p != std::string::npos) arch_ = arch_.substr(0, p);

    // ── Try hipBLASLt GEMM+bias+GELU epilogue ────────────────────────────────
    // Same conv as FullyConnectedOp: W[N,K] × X[K,M] → D[N,M] in col-major
    // GELU applied to each output element (tanh approximation)
    const int M = seq_, N = out_dim_, K = in_dim_;
    bool ok = false;
    do {
        if (hipblasLtCreate(&lt_handle_) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatmulDescCreate(&lt_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) break;

        hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_GELU_BIAS;  // = 36
        if (hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                &epi, sizeof(epi)) != HIPBLAS_STATUS_SUCCESS) break;

        hipblasOperation_t op_n = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

        // W stored as [N, K] row-major, lda=N (col-major)
        if (hipblasLtMatrixLayoutCreate(&lt_layout_W_, HIP_R_16F, N, K, N) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&lt_layout_X_, HIP_R_16F, K, M, K) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&lt_layout_D_, HIP_R_16F, N, M, N) != HIPBLAS_STATUS_SUCCESS) break;

        lt_workspace_bytes_ = 4 * 1024 * 1024;
        hipMalloc(&lt_workspace_, lt_workspace_bytes_);

        // Load cached algo or get heuristic (tuning deferred to first Execute)
        if (!load_algo(arch_, M, N, K, lt_algo_)) {
            hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
            hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &lt_workspace_bytes_, sizeof(lt_workspace_bytes_));
            hipblasLtMatmulHeuristicResult_t res; int nf = 0;
            hipblasLtMatmulAlgoGetHeuristic(lt_handle_, lt_desc_,
                lt_layout_W_, lt_layout_X_, lt_layout_D_, lt_layout_D_, pref, 1, &res, &nf);
            hipblasLtMatmulPreferenceDestroy(pref);
            if (nf == 0) break;
            lt_algo_ = res.algo;
        } else {
            lt_tuned_ = true;
            fprintf(stderr, "[FCGelu] hipBLASLt algo loaded from disk (M=%d N=%d K=%d)\n", M, N, K);
        }
        use_hipblaslt_ = true;
        ok = true;
        fprintf(stderr, "[FCGelu] hipBLASLt GEMM+bias+GELU: M=%d N=%d K=%d\n", M, N, K);
    } while (false);

    if (!ok) {
        // Clean up hipBLASLt
        if (lt_layout_D_) { hipblasLtMatrixLayoutDestroy(lt_layout_D_); lt_layout_D_ = nullptr; }
        if (lt_layout_X_) { hipblasLtMatrixLayoutDestroy(lt_layout_X_); lt_layout_X_ = nullptr; }
        if (lt_layout_W_) { hipblasLtMatrixLayoutDestroy(lt_layout_W_); lt_layout_W_ = nullptr; }
        if (lt_desc_)     { hipblasLtMatmulDescDestroy(lt_desc_);       lt_desc_     = nullptr; }
        if (lt_handle_)   { hipblasLtDestroy(lt_handle_);               lt_handle_   = nullptr; }

        // Fallback: rocMLIR BiasGELU → rocBLAS + native GELU
        auto maybe = rocmlir_gemm::compile_rocmlir_gemm(
            seq_, out_dim_, in_dim_, true, arch_, props.multiProcessorCount,
            rocmlir_gemm::Epilogue::BiasGELU);
        if (maybe) {
            rocmlir_kernel_ = std::make_shared<rocmlir_gemm::GemmKernel>(std::move(*maybe));
        } else {
            gemm_buf_bytes_ = (size_t)seq_ * out_dim_ * sizeof(__half);
            OPENVINO_ASSERT(hipMalloc(&gemm_buf_, gemm_buf_bytes_) == hipSuccess,
                            "FusedFCGELUOp: hipMalloc failed");
        }
    }
}

FusedFCGELUOp::~FusedFCGELUOp() {
    if (lt_workspace_) hipFree(lt_workspace_);
    if (lt_layout_D_)  hipblasLtMatrixLayoutDestroy(lt_layout_D_);
    if (lt_layout_X_)  hipblasLtMatrixLayoutDestroy(lt_layout_X_);
    if (lt_layout_W_)  hipblasLtMatrixLayoutDestroy(lt_layout_W_);
    if (lt_desc_)      hipblasLtMatmulDescDestroy(lt_desc_);
    if (lt_handle_)    hipblasLtDestroy(lt_handle_);
    if (gemm_buf_)     hipFree(gemm_buf_);
}

void FusedFCGELUOp::Execute(
        const InferenceRequestContext& context,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 3 && outputs.size() == 1);
    hipStream_t stream = context.getThreadContext().stream().get();

    // ── hipBLASLt GEMM+bias+GELU path ────────────────────────────────────────
    if (use_hipblaslt_) {
        // Lazy tuning + set bias pointer once (stable weight address → hipGraph compatible)
        if (!lt_tuned_) {
            const int M = seq_, N = out_dim_, K = in_dim_;
            void* bias = const_cast<void*>(inputs[2].get());
            // Set bias pointer permanently (weight is constant between inferences)
            hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bias, sizeof(void*));
            auto best = tune_gelu_algo(lt_handle_, lt_desc_,
                lt_layout_W_, lt_layout_X_, lt_layout_D_,
                inputs[1].get(), inputs[0].get(), bias, outputs[0].get(),
                lt_workspace_, lt_workspace_bytes_, stream);
            const_cast<FusedFCGELUOp*>(this)->lt_algo_ = best;
            const_cast<FusedFCGELUOp*>(this)->lt_tuned_ = true;
            save_algo(arch_, M, N, K, best);
        }
        const float alpha = 1.f, beta = 0.f;
        hipblasLtMatmul(
            lt_handle_, lt_desc_, &alpha,
            inputs[1].get(), lt_layout_W_,
            inputs[0].get(), lt_layout_X_,
            &beta,
            outputs[0].get(), lt_layout_D_,
            outputs[0].get(), lt_layout_D_,
            &lt_algo_, lt_workspace_, lt_workspace_bytes_, stream);
        return;
    }

    // ── rocMLIR BiasGELU kernel (if compiled) ─────────────────────────────────
    if (rocmlir_kernel_) {
        rocmlir_gemm::launch_rocmlir_gemm_bias(
            *rocmlir_kernel_, stream,
            const_cast<void*>(inputs[0].get()),
            const_cast<void*>(inputs[1].get()),
            const_cast<void*>(inputs[2].get()),
            outputs[0].get());
        return;
    }

    // ── rocBLAS + native HIP bias+GELU fallback ───────────────────────────────
    auto& handle = context.getThreadContext().rocBlasHandle();
    rocblas_set_stream(handle.get(), stream);
    const float alpha = 1.0f, beta = 0.0f;
    rocblas_gemm_ex(handle.get(),
        rocblas_operation_none, rocblas_operation_transpose,
        out_dim_, seq_, in_dim_, &alpha,
        inputs[1].get(), rocblas_datatype_f16_r, out_dim_,
        inputs[0].get(), rocblas_datatype_f16_r, in_dim_,
        &beta,
        gemm_buf_, rocblas_datatype_f16_r, out_dim_,
        gemm_buf_, rocblas_datatype_f16_r, out_dim_,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    kernel::launch_bias_gelu_fused(stream, gemm_buf_, inputs[2].get(),
                                   outputs[0].get(), seq_, out_dim_);
}

OPERATION_REGISTER(FusedFCGELUOp, FusedFCGELU);

}  // namespace rocm_gpu
}  // namespace ov
