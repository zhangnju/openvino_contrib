// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fused_qkv_op.hpp"
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <fmt/format.h>
#include <string>

namespace ov {
namespace rocm_gpu {

namespace {

static std::string qkv_cache_dir(const std::string& arch) {
    const char* h = std::getenv("HOME");
    return std::string(h ? h : "/tmp") + "/.cache/ov_hipblaslt_" + arch;
}

static std::string qkv_algo_path(const std::string& arch, int M, int N, int K) {
    return fmt::format("{}/qkv_gemm_{}_{}_{}.algo", qkv_cache_dir(arch), M, N, K);
}

static bool load_algo(const std::string& arch, int M, int N, int K, hipblasLtMatmulAlgo_t& algo) {
    std::ifstream f(qkv_algo_path(arch, M, N, K), std::ios::binary);
    if (!f) return false;
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&algo), sizeof(algo)));
}

static void save_algo(const std::string& arch, int M, int N, int K,
                      const hipblasLtMatmulAlgo_t& algo) {
    mkdir(qkv_cache_dir(arch).c_str(), 0755);
    std::ofstream f(qkv_algo_path(arch, M, N, K), std::ios::binary);
    if (f) f.write(reinterpret_cast<const char*>(&algo), sizeof(algo));
}

static hipblasLtMatmulAlgo_t tune_algo(
        hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc,
        hipblasLtMatrixLayout_t lW, hipblasLtMatrixLayout_t lX, hipblasLtMatrixLayout_t lD,
        const void* W, const void* X, void* bias, void* D,
        void* ws, size_t ws_bytes, hipStream_t stream) {

    hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &ws_bytes, sizeof(ws_bytes));
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
    fprintf(stderr, "[FusedQKV-tune] best algo idx=%d (%.3f ms)\n", bi, best);
    return cands[bi].algo;
}

}  // namespace

FusedQKVProjectionOp::FusedQKVProjectionOp(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {

    auto qkv = std::dynamic_pointer_cast<nodes::FusedQKVProjection>(node);
    OPENVINO_ASSERT(qkv, "FusedQKVProjectionOp: expected FusedQKVProjection node");
    seq_len_ = static_cast<int>(qkv->get_seq_len());
    hidden_  = static_cast<int>(qkv->get_hidden());

    const auto& props = context.device().props();
    arch_ = props.gcnArchName;
    if (auto p = arch_.find(':'); p != std::string::npos) arch_ = arch_.substr(0, p);
    num_cu_ = props.multiProcessorCount;

    // hipBLASLt setup: D[3H,M] = W[3H,H] × X[H,M] + bias[3H]
    // col-major convention: M=seq, N=3*hidden, K=hidden
    const int M = seq_len_, N = 3 * hidden_, K = hidden_;
    bool ok = false;
    do {
        if (hipblasLtCreate(&lt_handle_) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatmulDescCreate(&lt_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) break;

        hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_BIAS;
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        hipblasOperation_t op_n = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

        // W[3H, H] stored row-major = [3H, H] col-major, lda=3H
        // X[M, H] stored row-major = X^T[H, M] col-major, ldb=H
        if (hipblasLtMatrixLayoutCreate(&lt_layout_W_, HIP_R_16F, N, K, N) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&lt_layout_X_, HIP_R_16F, K, M, K) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&lt_layout_D_, HIP_R_16F, N, M, N) != HIPBLAS_STATUS_SUCCESS) break;

        lt_workspace_bytes_ = 4 * 1024 * 1024;
        hipMalloc(&lt_workspace_, lt_workspace_bytes_);

        // Load cached algo or get heuristic; tuning deferred to first Execute
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
            fprintf(stderr, "[FusedQKV] algo loaded from disk cache (M=%d N=%d K=%d)\n", M, N, K);
        }
        ok = true;
    } while (false);

    OPENVINO_ASSERT(ok, "FusedQKVProjectionOp: hipBLASLt setup failed");
    fprintf(stderr, "[FusedQKV] Initialized: M=%d N=%d K=%d (1 GEMM replaces 3×%dx%dx%d)\n",
            M, N, K, seq_len_, hidden_, hidden_);

    // Decide backend by tuning-cache presence (no runtime benchmark). A tuned
    // perf_config exists ⇒ rocMLIR (offline-verified faster on RDNA3); else hipBLASLt.
    // Under ROCMLIR_ENABLE_TUNING=1, get_tuned_gemm_config does the sweep+save.
    const std::string cfg = rocmlir_gemm::get_tuned_gemm_config(
        M, N, K, true, arch_, num_cu_, rocmlir_gemm::Epilogue::Bias);
    if (!cfg.empty()) {
        auto maybe = rocmlir_gemm::compile_rocmlir_gemm(
            M, N, K, /*transB=*/true, arch_, num_cu_, rocmlir_gemm::Epilogue::Bias, cfg);
        if (maybe) {
            rocmlir_kernel_ = std::make_shared<rocmlir_gemm::GemmKernel>(std::move(*maybe));
            use_rocmlir_ = true;
            fprintf(stderr, "[FusedQKV] using tuned rocMLIR GEMM+bias (cfg=%s)\n", cfg.c_str());
        }
    }
}

FusedQKVProjectionOp::~FusedQKVProjectionOp() {
    if (lt_workspace_) hipFree(lt_workspace_);
    if (lt_layout_D_)  hipblasLtMatrixLayoutDestroy(lt_layout_D_);
    if (lt_layout_X_)  hipblasLtMatrixLayoutDestroy(lt_layout_X_);
    if (lt_layout_W_)  hipblasLtMatrixLayoutDestroy(lt_layout_W_);
    if (lt_desc_)      hipblasLtMatmulDescDestroy(lt_desc_);
    if (lt_handle_)    hipblasLtDestroy(lt_handle_);
}

void FusedQKVProjectionOp::Execute(
        const InferenceRequestContext& context,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 3 && outputs.size() == 1);

    hipStream_t stream = context.getThreadContext().stream().get();
    void* bias = const_cast<void*>(inputs[2].get());

    // rocMLIR path: decided at construction, pure dispatch (no benchmark).
    if (use_rocmlir_ && rocmlir_kernel_) {
        rocmlir_gemm::launch_rocmlir_gemm_bias(*rocmlir_kernel_, stream,
            const_cast<void*>(inputs[0].get()), const_cast<void*>(inputs[1].get()),
            bias, outputs[0].get());
        return;
    }

    // hipBLASLt path: lazy algo tuning on first Execute, then dispatch.
    if (!lt_tuned_) {
        const int M = seq_len_, N = 3 * hidden_, K = hidden_;
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*));
        auto best = tune_algo(lt_handle_, lt_desc_,
            lt_layout_W_, lt_layout_X_, lt_layout_D_,
            inputs[1].get(), inputs[0].get(), bias, outputs[0].get(),
            lt_workspace_, lt_workspace_bytes_, stream);
        const_cast<FusedQKVProjectionOp*>(this)->lt_algo_ = best;
        const_cast<FusedQKVProjectionOp*>(this)->lt_tuned_ = true;
        save_algo(arch_, M, N, K, best);
    }
    hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*));
    const float alpha = 1.f, beta = 0.f;
    hipblasLtMatmul(
        lt_handle_, lt_desc_, &alpha,
        inputs[1].get(),  lt_layout_W_,
        inputs[0].get(),  lt_layout_X_,
        &beta,
        outputs[0].get(), lt_layout_D_,
        outputs[0].get(), lt_layout_D_,
        &lt_algo_, lt_workspace_, lt_workspace_bytes_, stream);
}

OPERATION_REGISTER(FusedQKVProjectionOp, FusedQKVProjection);

}  // namespace rocm_gpu
}  // namespace ov
