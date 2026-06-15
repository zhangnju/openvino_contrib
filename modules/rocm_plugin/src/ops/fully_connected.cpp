// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

#include <rocm/blas.hpp>
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/matmul.hpp>
#include <utility>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>

#include "converters.hpp"
#include "rocm/constant_factory.hpp"
#include "matmul.hpp"
#include <sys/stat.h>
#include <fstream>
#include <mutex>
#include <unordered_set>
#include <fmt/format.h>

namespace {
// Vectorized f16 bias broadcast: processes 2 elements per thread via __half2.
// cols must be even (BERT hidden=768/3072 always is). Falls back to scalar otherwise.
__global__ void broadcast_bias_add_h2(
        __half2* __restrict__ out_h2,
        const __half2* __restrict__ bias_h2,
        int rows, int cols2) {           // cols2 = cols/2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols2;
    if (idx >= total) return;
    int col2 = idx % cols2;             // pair index within a row
    out_h2[idx] = __hadd2(out_h2[idx], bias_h2[col2]);
}

// Scalar fallback for f32 or odd cols
template<typename T>
__global__ void broadcast_bias_add_scalar(T* out, const T* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    out[idx] += bias[idx % cols];
}

void launch_broadcast_bias_add(hipStream_t stream, void* out, const void* bias,
                               int rows, int cols, bool is_fp16) {
    constexpr int THREADS = 256;
    if (is_fp16 && (cols % 2 == 0)) {
        int cols2  = cols / 2;
        int total2 = rows * cols2;
        int blocks = (total2 + THREADS - 1) / THREADS;
        hipLaunchKernelGGL(broadcast_bias_add_h2, blocks, THREADS, 0, stream,
            reinterpret_cast<__half2*>(out),
            reinterpret_cast<const __half2*>(bias),
            rows, cols2);
    } else if (is_fp16) {
        int blocks = (rows * cols + THREADS - 1) / THREADS;
        hipLaunchKernelGGL(broadcast_bias_add_scalar<__half>, blocks, THREADS, 0, stream,
            static_cast<__half*>(out), static_cast<const __half*>(bias), rows, cols);
    } else {
        int blocks = (rows * cols + THREADS - 1) / THREADS;
        hipLaunchKernelGGL(broadcast_bias_add_scalar<float>, blocks, THREADS, 0, stream,
            static_cast<float*>(out), static_cast<const float*>(bias), rows, cols);
    }
}
}  // namespace

namespace {

// ── hipBLASLt algorithm tuning cache ─────────────────────────────────────────
// Disk cache: ~/.cache/ov_hipblaslt_<arch>/gemm_bias_<M>_<N>_<K>_tB<0|1>.algo
// Content: raw bytes of hipblasLtMatmulAlgo_t (architecture-specific, safe to store)

static std::string hipblaslt_cache_dir(const std::string& arch) {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.cache/ov_hipblaslt_" + arch;
}

static std::string hipblaslt_algo_path(const std::string& arch, int M, int N, int K, bool tB) {
    return fmt::format("{}/gemm_bias_{}_{}_{}_tB{}.algo",
                       hipblaslt_cache_dir(arch), M, N, K, (int)tB);
}

static bool hipblaslt_load_algo(const std::string& arch, int M, int N, int K, bool tB,
                                 hipblasLtMatmulAlgo_t& algo) {
    std::ifstream f(hipblaslt_algo_path(arch, M, N, K, tB), std::ios::binary);
    if (!f) return false;
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&algo), sizeof(algo)));
}

static void hipblaslt_save_algo(const std::string& arch, int M, int N, int K, bool tB,
                                 const hipblasLtMatmulAlgo_t& algo) {
    const std::string dir = hipblaslt_cache_dir(arch);
    mkdir(dir.c_str(), 0755);
    std::ofstream f(hipblaslt_algo_path(arch, M, N, K, tB), std::ios::binary);
    if (f) f.write(reinterpret_cast<const char*>(&algo), sizeof(algo));
}

// In-process set to avoid re-tuning the same shape in multiple FullyConnectedOp instances
static std::mutex                    g_tune_mu;
static std::unordered_set<std::string> g_tuned_shapes;

// Enumerate up to MAX_CAND hipBLASLt algorithms for the given descriptor/layouts,
// time each with real data, return the fastest algo.
static hipblasLtMatmulAlgo_t tune_hipblaslt(
        hipblasLtHandle_t handle,
        hipblasLtMatmulDesc_t desc,
        hipblasLtMatrixLayout_t layout_B,
        hipblasLtMatrixLayout_t layout_A,
        hipblasLtMatrixLayout_t layout_D,
        const void* B_ptr, const void* A_ptr,
        void* bias_ptr, void* D_ptr,
        void* workspace, size_t workspace_bytes,
        hipStream_t stream,
        int M, int N, int K, bool transB) {

    constexpr int MAX_CAND = 50;
    constexpr int WARMUP   = 5;
    constexpr int REPS     = 20;

    // Build preference with workspace limit
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    hipblasLtMatmulPreferenceSetAttribute(pref,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));

    hipblasLtMatmulHeuristicResult_t candidates[MAX_CAND];
    int n_found = 0;
    hipblasLtMatmulAlgoGetHeuristic(handle, desc,
        layout_B, layout_A, layout_D, layout_D,
        pref, MAX_CAND, candidates, &n_found);
    hipblasLtMatmulPreferenceDestroy(pref);

    if (n_found == 0) return candidates[0].algo;  // shouldn't happen

    // Set bias pointer for timing
    hipblasLtMatmulDescSetAttribute(desc,
        HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(void*));

    const float alpha = 1.f, beta = 0.f;
    auto run = [&](const hipblasLtMatmulAlgo_t& algo) {
        hipblasLtMatmul(handle, desc, &alpha,
            B_ptr, layout_B, A_ptr, layout_A,
            &beta, D_ptr, layout_D, D_ptr, layout_D,
            &algo, workspace, workspace_bytes, stream);
    };

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);

    float best_ms = 1e9f;
    int   best_idx = 0;
    for (int i = 0; i < n_found; ++i) {
        for (int w = 0; w < WARMUP; ++w) run(candidates[i].algo);
        hipStreamSynchronize(stream);
        hipEventRecord(t0, stream);
        for (int r = 0; r < REPS; ++r) run(candidates[i].algo);
        hipEventRecord(t1, stream);
        hipStreamSynchronize(stream);
        float ms; hipEventElapsedTime(&ms, t0, t1);
        ms /= REPS;
        if (ms < best_ms) { best_ms = ms; best_idx = i; }
    }

    hipEventDestroy(t0); hipEventDestroy(t1);
    fprintf(stderr, "[FC-tune] %dx%dx%d tB=%d: %d candidates, best=%d (%.3f ms)\n",
            M, N, K, (int)transB, n_found, best_idx, best_ms);
    return candidates[best_idx].algo;
}

}  // namespace

namespace ov {
namespace rocm_gpu {

FullyConnectedOp::FullyConnectedOp(const CreationContext& context,
                                   const NodeOp& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationRocBlas(context, node, std::move(inputIds), std::move(outputIds)),
      matmul_op_{
          context, node, IndexCollection{input_ids_.begin(), input_ids_.end() - 1}, IndexCollection(output_ids_)} {
    bias_size_ = node.get_input_tensor(2).size();
    auto biasShape = node.get_input_shape(2);
    auto matrixShape = node.get_output_shape(0);
    OPENVINO_ASSERT(biasShape.size() > 0, "Node name: ", GetName());
    MatMulOp::BroadcastToMatrix(biasShape);
    const auto biasShapeSize = ov::shape_size(biasShape);
    const auto matrixShapeSize = ov::shape_size(matrixShape);
    OPENVINO_ASSERT(matrixShapeSize >= biasShapeSize, "Node name: ", GetName());
    auto batchBiasCount = MatMulOp::GetMatrixNumBatches(biasShape);
    auto matMulBatchCount = matmul_op_.GetBatchCount();
    OPENVINO_ASSERT(matMulBatchCount >= batchBiasCount, "Node name: ", GetName());
    batch_bias_count_ = matrixShapeSize / biasShapeSize;
    bias_cols_ = biasShapeSize;
    bias_rows_ = batch_bias_count_;
    is_fp16_ = (node.get_output_element_type(0) == ov::element::f16);

    if (!is_fp16_ || matmul_op_.GetBatchCount() != 1) return;

    const int M = static_cast<int>(matmul_op_.GetM());
    const int N = static_cast<int>(matmul_op_.GetN());
    const int K = static_cast<int>(matmul_op_.GetK());
    const bool transB = (matmul_op_.GetTransposeB() != rocblas_operation_none);

    const auto& props = context.device().props();
    arch_ = props.gcnArchName;
    if (auto p = arch_.find(':'); p != std::string::npos) arch_ = arch_.substr(0, p);
    num_cu_ = props.multiProcessorCount;

    // ── Try hipBLASLt GEMM + bias epilogue (single kernel, no separate bias add) ─
    // Computes: D[M,N] = A[M,K] × B^T[K,N] + bias[N]  (bias broadcast over rows)
    // In col-major rocBLAS convention: [N,M] = B[N,K] × A^T[K,M] + bias[N]
    bool hipblaslt_ok = false;
    do {
        if (hipblasLtCreate(&lt_handle_) != HIPBLAS_STATUS_SUCCESS) break;

        // Matmul descriptor with BIAS epilogue
        if (hipblasLtMatmulDescCreate(&lt_desc_,
                HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) break;

        hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_BIAS;
        hipblasLtMatmulDescSetAttribute(lt_desc_,
            HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));

        // Transpose ops: rocBLAS convention for row-major A[M,K] × B^T[K,N]
        // → col-major: B[N,K] × A^T[K,M] = col-major [N,M]
        // hipBLASLt transa = NONE (B not transposed in col-major)
        // hipBLASLt transb = NONE (A^T already col-major)
        hipblasOperation_t op_none = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(lt_desc_,
            HIPBLASLT_MATMUL_DESC_TRANSA, &op_none, sizeof(op_none));
        hipblasLtMatmulDescSetAttribute(lt_desc_,
            HIPBLASLT_MATMUL_DESC_TRANSB, &op_none, sizeof(op_none));

        // Matrix layouts (col-major):
        //   "A" for hipBLASLt = our weight B stored as [N,K] (lda=N)
        //   "B" for hipBLASLt = our input A stored as [K,M] (ldb=K, interpreted as A^T)
        //   "D" = output [N,M] (ldd=N)
        const int lda = transB ? N : K;   // leading dim of B (our weight)
        if (hipblasLtMatrixLayoutCreate(&lt_layout_B_, HIP_R_16F, N, K, lda)
                != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&lt_layout_A_, HIP_R_16F, K, M, K)
                != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatrixLayoutCreate(&lt_layout_D_, HIP_R_16F, N, M, N)
                != HIPBLAS_STATUS_SUCCESS) break;

        // Algorithm search
        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);
        lt_workspace_bytes_ = 4 * 1024 * 1024;  // 4 MB workspace
        hipblasLtMatmulPreferenceSetAttribute(pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &lt_workspace_bytes_, sizeof(lt_workspace_bytes_));

        hipblasLtMatmulHeuristicResult_t result;
        int returned = 0;
        auto status = hipblasLtMatmulAlgoGetHeuristic(
            lt_handle_, lt_desc_,
            lt_layout_B_, lt_layout_A_, lt_layout_D_, lt_layout_D_,
            pref, 1, &result, &returned);
        hipblasLtMatmulPreferenceDestroy(pref);

        if (status != HIPBLAS_STATUS_SUCCESS || returned == 0) break;
        lt_algo_ = result.algo;

        if (lt_workspace_bytes_ > 0)
            hipMalloc(&lt_workspace_, lt_workspace_bytes_);

        lt_M_ = M; lt_N_ = N; lt_K_ = K; lt_transB_ = transB;
        use_hipblaslt_ = true;   // tentative; overridden by benchmark if rocMLIR wins
        hipblaslt_ok = true;
        fprintf(stderr, "[FC] hipBLASLt GEMM+bias: M=%d N=%d K=%d transB=%d\n",
                M, N, K, (int)transB);
    } while (false);

    // Compile rocMLIR GEMM-only kernel regardless of hipBLASLt outcome.
    // The first-Execute benchmark will compare hipBLASLt+bias vs rocMLIR+bias_add
    // and switch to rocMLIR if it wins (e.g., FFN down 256x768x3072 on gfx1201).
    {
        auto maybe = rocmlir_gemm::compile_rocmlir_gemm(M, N, K, transB, arch_, num_cu_);
        if (maybe)
            rocmlir_kernel_ = std::make_shared<rocmlir_gemm::GemmKernel>(std::move(*maybe));
    }

    if (!hipblaslt_ok) {
        // hipBLASLt unavailable — clean up and use rocMLIR/rocBLAS fallback
        if (lt_layout_D_) { hipblasLtMatrixLayoutDestroy(lt_layout_D_); lt_layout_D_ = nullptr; }
        if (lt_layout_A_) { hipblasLtMatrixLayoutDestroy(lt_layout_A_); lt_layout_A_ = nullptr; }
        if (lt_layout_B_) { hipblasLtMatrixLayoutDestroy(lt_layout_B_); lt_layout_B_ = nullptr; }
        if (lt_desc_)     { hipblasLtMatmulDescDestroy(lt_desc_);       lt_desc_     = nullptr; }
        if (lt_handle_)   { hipblasLtDestroy(lt_handle_);               lt_handle_   = nullptr; }
    }
}

FullyConnectedOp::~FullyConnectedOp() {
    if (lt_workspace_) hipFree(lt_workspace_);
    if (lt_layout_D_)  hipblasLtMatrixLayoutDestroy(lt_layout_D_);
    if (lt_layout_A_)  hipblasLtMatrixLayoutDestroy(lt_layout_A_);
    if (lt_layout_B_)  hipblasLtMatrixLayoutDestroy(lt_layout_B_);
    if (lt_desc_)      hipblasLtMatmulDescDestroy(lt_desc_);
    if (lt_handle_)    hipblasLtDestroy(lt_handle_);
}

void FullyConnectedOp::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    auto& stream = context.getThreadContext().stream();

    // ── hipBLASLt path: single kernel for GEMM + bias add ────────────────────
    if (use_hipblaslt_) {
        // ── Lazy algorithm tuning (on first Execute with real buffers) ──────────
        if (!lt_tuned_) {
            auto* self = const_cast<FullyConnectedOp*>(this);
            const std::string shape_key = arch_ + fmt::format("_{}_{}_{}_tB{}",
                                                lt_M_, lt_N_, lt_K_, (int)lt_transB_);

            // In-process dedup: multiple FC instances with same shape share result
            std::unique_lock<std::mutex> lk(g_tune_mu);
            if (g_tuned_shapes.count(shape_key)) {
                // Another instance already tuned this shape — reload from disk
                hipblaslt_load_algo(arch_, lt_M_, lt_N_, lt_K_, lt_transB_, self->lt_algo_);
                self->lt_tuned_ = true;
                lk.unlock();
            } else {
                lk.unlock();
                // Try disk cache first
                hipblasLtMatmulAlgo_t cached_algo;
                if (hipblaslt_load_algo(arch_, lt_M_, lt_N_, lt_K_, lt_transB_, cached_algo)) {
                    self->lt_algo_ = cached_algo;
                    fprintf(stderr, "[FC-tune] disk-cache hit %dx%dx%d tB=%d\n",
                            lt_M_, lt_N_, lt_K_, (int)lt_transB_);
                } else {
                    // Step 1: tune hipBLASLt (enumerate 50 algo candidates)
                    void* bias_tune = const_cast<void*>(inputs[2].get());
                    // Set bias pointer permanently (model weight = stable address)
                    hipblasLtMatmulDescSetAttribute(lt_desc_,
                        HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_tune, sizeof(void*));

                    self->lt_algo_ = tune_hipblaslt(
                        lt_handle_, lt_desc_,
                        lt_layout_B_, lt_layout_A_, lt_layout_D_,
                        inputs[1].get(), inputs[0].get(),
                        bias_tune, outputs[0].get(),
                        lt_workspace_, lt_workspace_bytes_,
                        stream.get(), lt_M_, lt_N_, lt_K_, lt_transB_);
                    hipblaslt_save_algo(arch_, lt_M_, lt_N_, lt_K_, lt_transB_, self->lt_algo_);

                    // Step 2: compare with rocMLIR+bias_add if rocMLIR compiled
                    // rocMLIR GEMM-only + vectorized bias_add may be faster for some shapes
                    // (e.g., FFN down 256x768x3072 on gfx1201: rocMLIR 17.5µs vs hipBLASLt 22µs)
                    if (rocmlir_kernel_) {
                        auto& rblas_handle = context.getThreadContext().rocBlasHandle();
                        auto select = rocmlir_gemm::select_backend(
                            rblas_handle.get(), stream.get(),
                            lt_M_, lt_N_, lt_K_, lt_transB_,
                            inputs[0].get(), inputs[1].get(), outputs[0].get(),
                            arch_, num_cu_);

                        if (select.backend == rocmlir_gemm::Backend::ROCMLIR && select.kernel) {
                            // Time hipBLASLt+bias
                            const float alpha=1.f, beta=0.f;
                            // Inline median timing (5 warmup + 20 reps)
                            auto fc_median_us = [&](auto fn) -> double {
                                for (int w = 0; w < 5; ++w) fn();
                                hipStreamSynchronize(stream.get());
                                std::vector<float> ms(20);
                                hipEvent_t t0, t1; hipEventCreate(&t0); hipEventCreate(&t1);
                                for (int r = 0; r < 20; ++r) {
                                    hipEventRecord(t0, stream.get()); fn(); hipEventRecord(t1, stream.get());
                                    hipStreamSynchronize(stream.get());
                                    hipEventElapsedTime(&ms[r], t0, t1);
                                }
                                hipEventDestroy(t0); hipEventDestroy(t1);
                                std::sort(ms.begin(), ms.end());
                                return (double)ms[10] * 1000.0;
                            };

                            auto lt_us = fc_median_us([&]() {
                                hipblasLtMatmul(lt_handle_, lt_desc_, &alpha,
                                    inputs[1].get(), lt_layout_B_,
                                    inputs[0].get(), lt_layout_A_,
                                    &beta, outputs[0].get(), lt_layout_D_,
                                    outputs[0].get(), lt_layout_D_,
                                    &self->lt_algo_, lt_workspace_, lt_workspace_bytes_, stream.get());
                            });
                            // Time rocMLIR GEMM + bias_add
                            auto rocmlir_us = fc_median_us([&]() {
                                rocmlir_gemm::launch_rocmlir_gemm(*select.kernel, stream.get(),
                                    const_cast<void*>(inputs[0].get()),
                                    const_cast<void*>(inputs[1].get()),
                                    outputs[0].get());
                                // bias_add (vectorized half2)
                                launch_broadcast_bias_add(stream.get(), outputs[0].get(),
                                    inputs[2].get(),
                                    static_cast<int>(bias_rows_),
                                    static_cast<int>(bias_cols_), is_fp16_);
                            });
                            if (rocmlir_us < lt_us) {
                                fprintf(stderr, "[FC-tune] rocMLIR+bias (%.2fus) beats hipBLASLt (%.2fus) "
                                        "for %dx%dx%d → switching\n",
                                        rocmlir_us, lt_us, lt_M_, lt_N_, lt_K_);
                                self->use_hipblaslt_ = false;
                                self->use_rocmlir_   = true;
                                self->rocmlir_kernel_ = select.kernel;
                            } else {
                                fprintf(stderr, "[FC-tune] hipBLASLt+bias (%.2fus) beats rocMLIR (%.2fus) "
                                        "for %dx%dx%d\n", lt_us, rocmlir_us, lt_M_, lt_N_, lt_K_);
                            }
                        }
                    }
                }
                std::lock_guard<std::mutex> lk2(g_tune_mu);
                g_tuned_shapes.insert(shape_key);
                self->lt_tuned_ = true;
            }
        }

        // After tuning, use_hipblaslt_ might have been cleared in favour of rocMLIR.
        // If still true: launch hipBLASLt (bias included in epilogue, no separate add).
        if (use_hipblaslt_) {
            const float alpha = 1.f, beta = 0.f;
            hipblasLtMatmul(
                lt_handle_, lt_desc_, &alpha,
                inputs[1].get(),  lt_layout_B_,
                inputs[0].get(),  lt_layout_A_,
                &beta,
                outputs[0].get(), lt_layout_D_,
                outputs[0].get(), lt_layout_D_,
                &lt_algo_, lt_workspace_, lt_workspace_bytes_, stream.get());
            return;
        }
        // Otherwise fall through to rocMLIR path below (use_rocmlir_ = true set during tuning)
    }

    // ── rocMLIR benchmark (deferred to first Execute) ─────────────────────────
    if (rocmlir_kernel_ && !rocmlir_benchmarked_) {
        auto& handle = context.getThreadContext().rocBlasHandle();
        const int M = static_cast<int>(matmul_op_.GetM());
        const int N = static_cast<int>(matmul_op_.GetN());
        const int K = static_cast<int>(matmul_op_.GetK());
        const bool transB = (matmul_op_.GetTransposeB() != rocblas_operation_none);

        auto res = rocmlir_gemm::select_backend(
            handle.get(), stream.get(),
            M, N, K, transB,
            inputs[0].get(), inputs[1].get(), outputs[0].get(),
            arch_, num_cu_);

        auto* self = const_cast<FullyConnectedOp*>(this);
        self->rocmlir_benchmarked_ = true;
        self->use_rocmlir_ = (res.backend == rocmlir_gemm::Backend::ROCMLIR);
        if (!use_rocmlir_) self->rocmlir_kernel_.reset();
    }

    // ── GEMM ──────────────────────────────────────────────────────────────────
    if (use_rocmlir_ && rocmlir_kernel_) {
        rocmlir_gemm::launch_rocmlir_gemm(
            *rocmlir_kernel_, stream.get(),
            const_cast<void*>(inputs[0].get()),
            const_cast<void*>(inputs[1].get()),
            outputs[0].get());
    } else {
        matmul_op_.Execute(context, inputs.first(inputs.size() - 1), outputs, workbuffers);
    }

    // ── Bias add (only when not using hipBLASLt) ──────────────────────────────
    launch_broadcast_bias_add(stream.get(), outputs[0].get(), inputs[2].get(),
                              static_cast<int>(bias_rows_), static_cast<int>(bias_cols_), is_fp16_);
}

rocmGraphCompatibility FullyConnectedOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

OPERATION_REGISTER(FullyConnectedOp, FullyConnected);
}  // namespace rocm_gpu
}  // namespace ov
