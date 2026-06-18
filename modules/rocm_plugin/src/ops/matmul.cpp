// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.hpp"

#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas-beta.h>
#include <rocm/blas.hpp>
#include <rocm/float16.hpp>
#include <rocm_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <transformer/nodes/fully_connected.hpp>
#include <utility>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "converters.hpp"
#include "rocm/constant_factory.hpp"

// ── rocBLAS GEMM solution-index cache ────────────────────────────────────────
// When ROCM_TUNE_GEMM=1 is set, the first call to a new GEMM shape enumerates
// all available solutions and benchmarks them. The winning solution_index is
// cached globally and used for all subsequent calls with the same shape.
namespace {

struct GemmKey {
    int m, n, k, batch;
    int transa, transb;
    int dt, ct;   // rocblas_datatype enum values
    bool operator==(const GemmKey& o) const {
        return m==o.m && n==o.n && k==o.k && batch==o.batch &&
               transa==o.transa && transb==o.transb &&
               dt==o.dt && ct==o.ct;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const {
        size_t h = 14695981039346656037ULL;
        auto mix = [&](size_t v){ h ^= v; h *= 1099511628211ULL; };
        mix(k.m); mix(k.n); mix(k.k); mix(k.batch);
        mix(k.transa); mix(k.transb); mix(k.dt); mix(k.ct);
        return h;
    }
};

struct GemmCache {
    std::mutex mtx;
    std::unordered_map<GemmKey, rocblas_int, GemmKeyHash> best;

    static GemmCache& instance() {
        static GemmCache g;
        return g;
    }

    // Returns the cached solution_index, or -1 if not yet tuned.
    rocblas_int get(const GemmKey& key) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = best.find(key);
        return (it != best.end()) ? it->second : -1;
    }

    void set(const GemmKey& key, rocblas_int sol) {
        std::lock_guard<std::mutex> lk(mtx);
        best[key] = sol;
    }
};

// Tune a rocBLAS strided-batched GEMM and return the best solution_index.
// Uses hipEvent timing over WARMUP+BENCH iterations for each solution.
rocblas_int tuneBatchedGemm(
        rocblas_handle handle,
        rocblas_operation ta, rocblas_operation tb,
        int m, int n, int k,
        const void* alpha, const void* A, rocblas_datatype dt_a, int lda, long long sA,
        const void* B, rocblas_datatype dt_b, int ldb, long long sB,
        const void* beta, void* C, rocblas_datatype dt_c, int ldc, long long sC,
        int batch, rocblas_datatype ct)
{
    constexpr int WARMUP = 2;
    constexpr int BENCH  = 5;

    // Enumerate solutions for strided-batched GEMM
    rocblas_int list_size = 0;
    rocblas_gemm_strided_batched_ex_get_solutions(
        handle, ta, tb, m, n, k,
        alpha,
        A, dt_a, lda, sA,
        B, dt_b, ldb, sB,
        beta,
        C, dt_c, ldc, sC,
        C, dt_c, ldc, sC,
        batch, ct,
        rocblas_gemm_algo_solution_index, 0,
        nullptr, &list_size);

    if (list_size <= 0) return 0;

    std::vector<rocblas_int> solutions(list_size);
    rocblas_gemm_strided_batched_ex_get_solutions(
        handle, ta, tb, m, n, k,
        alpha,
        A, dt_a, lda, sA,
        B, dt_b, ldb, sB,
        beta,
        C, dt_c, ldc, sC,
        C, dt_c, ldc, sC,
        batch, ct,
        rocblas_gemm_algo_solution_index, 0,
        solutions.data(), &list_size);

    hipEvent_t ev_start, ev_stop;
    hipEventCreate(&ev_start);
    hipEventCreate(&ev_stop);

    rocblas_int best_sol = solutions[0];
    float best_ms = 1e9f;

    auto run_sol = [&](rocblas_int sol) -> bool {
        return rocblas_gemm_strided_batched_ex(
            handle, ta, tb, m, n, k, alpha,
            A, dt_a, lda, sA,
            B, dt_b, ldb, sB,
            beta,
            C, dt_c, ldc, sC,
            C, dt_c, ldc, sC,
            batch, ct,
            rocblas_gemm_algo_solution_index, sol, 0) == rocblas_status_success;
    };

    for (rocblas_int sol : solutions) {
        // warmup
        bool ok = true;
        for (int w = 0; w < WARMUP && ok; ++w) ok = run_sol(sol);
        if (!ok) continue;

        // benchmark
        hipEventRecord(ev_start);
        for (int r = 0; r < BENCH && ok; ++r) ok = run_sol(sol);
        hipEventRecord(ev_stop);
        hipEventSynchronize(ev_stop);
        if (!ok) continue;

        float ms = 0.f;
        hipEventElapsedTime(&ms, ev_start, ev_stop);
        ms /= BENCH;
        if (ms < best_ms) { best_ms = ms; best_sol = sol; }
    }

    hipEventDestroy(ev_start);
    hipEventDestroy(ev_stop);

    fprintf(stderr, "[ROCM_TUNE_GEMM] m=%d n=%d k=%d batch=%d -> sol=%d (%.3f ms)\n",
            m, n, k, batch, best_sol, best_ms);
    return best_sol;
}

bool gemmTuningEnabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("ROCM_TUNE_GEMM");
        v = (e && e[0] != '0') ? 1 : 0;
    }
    return v == 1;
}

}  // anonymous namespace

namespace ov {
namespace rocm_gpu {

template <typename TOperation>
MatMulOp::MatMulOp(const CreationContext& context,
                   const TOperation& op,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationRocBlas(context, op, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(op.get_input_size() >= 2, "Node name: ", GetName());
    OPENVINO_ASSERT(op.get_output_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(convertDataType<hipDataType>(op.get_input_element_type(0)) ==
                        convertDataType<hipDataType>(op.get_input_element_type(1)),
                    "Node name: ",
                    GetName());
    data_type_ = convertDataType<hipDataType>(op.get_input_element_type(0));
    compute_type_ = GetComputeType(data_type_, convertDataType<hipDataType>(op.get_output_element_type(0)));
    auto inputAShape = op.get_input_shape(0);
    auto inputBShape = op.get_input_shape(1);
    auto outputCShape = op.get_output_shape(0);
    OPENVINO_ASSERT(inputAShape.size() > 0, "Node name: ", GetName());
    OPENVINO_ASSERT(inputBShape.size() > 0, "Node name: ", GetName());
    bool transposeA = op.get_transpose_a();
    bool transposeB = op.get_transpose_b();
    const int batchACount = GetMatrixNumBatches(inputAShape);
    const int batchBCount = GetMatrixNumBatches(inputBShape);
    BroadcastShapes(inputAShape, transposeA, inputBShape, transposeB, outputCShape);
    batch_count_ = std::max(batchACount, batchBCount);
    const size_t rowsA = *(inputAShape.end() - !transposeA - 1);
    const size_t colsA = *(inputAShape.end() - transposeA - 1);
    const size_t rowsB = *(inputBShape.end() - !transposeB - 1);
    const size_t colsB = *(inputBShape.end() - transposeB - 1);
    OPENVINO_ASSERT(colsA == rowsB, "Node name: ", GetName());
    m_ = rowsA;
    k_ = colsA;
    n_ = colsB;
    ld_a_ = *(inputAShape.end() - 1);
    ld_b_ = *(inputBShape.end() - 1);
    ld_c_ = *(outputCShape.end() - 1);
    stride_a_ = (batchACount > 1) ? (m_ * k_) : 0;
    stride_b_ = (batchBCount > 1) ? (k_ * n_) : 0;
    stride_c_ = (m_ * n_);
    rocblas_transpose_a_ = transposeA ? rocblas_operation_transpose : rocblas_operation_none;
    rocblas_transpose_b_ = transposeB ? rocblas_operation_transpose : rocblas_operation_none;

    // TODO: Read GEMM stride overrides from rt_info when BertAttentionTransposeFusion is ready.
    if constexpr (std::is_same_v<TOperation, nodes::FullyConnected>) {
        beta_ = &rocm::NumericConst<rocm::constants::one>(compute_type_);
    } else {
        beta_ = &rocm::NumericConst<rocm::constants::zero>(compute_type_);
    }
    OPENVINO_ASSERT(m_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(k_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(n_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_a_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_b_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(ld_c_ != 0, "Node name: ", GetName());
    OPENVINO_ASSERT(batch_count_ != 0, "Node name: ", GetName());

    // Optional tuned rocMLIR GEMM for plain 2D fp16, single batch, no A-transpose,
    // beta==0. compile_rocmlir_gemm models C[M,N]=A[M,K]*(transB?Bᵀ:B). Decided by
    // tuning-cache presence; a one-time numeric check vs rocBLAS runs in Execute.
    const auto& props = context.device().props();
    arch_ = props.gcnArchName;
    if (auto p = arch_.find(':'); p != std::string::npos) arch_ = arch_.substr(0, p);
    num_cu_ = props.multiProcessorCount;
    const bool beta_zero = (beta_ == &rocm::NumericConst<rocm::constants::zero>(compute_type_));
    if (data_type_ == HIP_R_16F && batch_count_ == 1 &&
        rocblas_transpose_a_ == rocblas_operation_none && beta_zero) {
        const bool transB = (rocblas_transpose_b_ == rocblas_operation_transpose);
        const std::string cfg = rocmlir_gemm::get_tuned_gemm_config(
            m_, n_, k_, transB, arch_, num_cu_, rocmlir_gemm::Epilogue::None);
        if (!cfg.empty()) {
            auto maybe = rocmlir_gemm::compile_rocmlir_gemm(
                m_, n_, k_, transB, arch_, num_cu_, rocmlir_gemm::Epilogue::None, cfg);
            if (maybe) rocmlir_kernel_ = std::make_shared<rocmlir_gemm::GemmKernel>(std::move(*maybe));
        }
    }
}
template MatMulOp::MatMulOp(const CreationContext& context,
                            const ov::op::v0::MatMul&,
                            IndexCollection&&,
                            IndexCollection&&);
template MatMulOp::MatMulOp(const CreationContext& context,
                            const nodes::FullyConnected&,
                            IndexCollection&&,
                            IndexCollection&&);

hipDataType MatMulOp::GetComputeType(const hipDataType abDataType, const hipDataType cDataType) {
    constexpr auto SwitchCase = [](hipDataType a, hipDataType b) constexpr { return (a << 16) + b; };
    /**
     * NOTE: This switch is an implementation of CuBlas table for available compute types:
     * @reference https://docs.rocm.com/rocm/cublas/index.html#cublas-GemmStridedBatchedEx
     */
    switch (SwitchCase(abDataType, cDataType)) {
        case SwitchCase(HIP_R_16F, HIP_R_16F): {
            return HIP_R_16F;
        }
        case SwitchCase(HIP_R_8I, HIP_R_32I): {
            return HIP_R_32I;
        }
#ifdef rocm_HAS_BF16_TYPE
        case SwitchCase(HIP_R_16BF, HIP_R_16BF):
        case SwitchCase(HIP_R_16BF, HIP_R_32F):
#endif
        case SwitchCase(HIP_R_8I, HIP_R_32F):
        case SwitchCase(HIP_R_16F, HIP_R_32F):
        case SwitchCase(HIP_R_32F, HIP_R_32F):
        case SwitchCase(HIP_C_8I, HIP_C_32F):
        case SwitchCase(HIP_C_32F, HIP_C_32F): {
            return HIP_R_32F;
        }
        case SwitchCase(HIP_R_64F, HIP_R_64F):
        case SwitchCase(HIP_C_64F, HIP_C_64F): {
            return HIP_R_64F;
        }
        default:
            throw_ov_exception(
                fmt::format("Not supported combination of A and B types [{}] "
                            "with C type [{}]",
                            abDataType,
                            cDataType));
    }
}

int MatMulOp::GetMatrixNumBatches(const ov::Shape& matrixShape) {
    return matrixShape.size() >= 2
               ? std::accumulate(matrixShape.begin(), matrixShape.end() - 2, 1, std::multiplies<size_t>())
               : 1;
}

void MatMulOp::BroadcastShapes(
    ov::Shape& matrixAShape, bool& transposeA, ov::Shape& matrixBShape, bool& transposeB, ov::Shape& matrixCShape) {
    /**
     * NOTE: See NGraph documentation for broadcasting:
     * @reference https://docs.openvinotoolkit.org/latest/openvino_docs_ops_matrix_MatMul_1.html
     */
    if (matrixAShape.size() == 1 && matrixBShape.size() == 1) {
        // 1D x 1D: [X] x [X] -> [1, X] x [X, 1] -> [1, 1] => [] (scalar)
        matrixAShape = ov::Shape{1, matrixAShape[0]};
        matrixBShape = ov::Shape{matrixBShape[0], 1};
        transposeA = false;
        transposeB = false;
    } else if (matrixAShape.size() == 1 && matrixBShape.size() > 1) {
        // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
        matrixAShape = ov::Shape{1, matrixAShape[0]};
        transposeA = false;
    } else if (matrixAShape.size() > 1 && matrixBShape.size() == 1) {
        // ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]
        matrixBShape = ov::Shape{matrixBShape[0], 1};
        transposeB = false;
    } else if (matrixAShape.size() > 1 && matrixBShape.size() > 1) {
        // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
        auto broadcastNdToMd = [](const auto& shapeToBroadcast, auto& broadcastShape) {
            OPENVINO_ASSERT(shapeToBroadcast.size() >= broadcastShape.size());
            std::vector<size_t> newAxies;
            newAxies.reserve(shapeToBroadcast.size());
            newAxies.insert(newAxies.end(), shapeToBroadcast.begin(), shapeToBroadcast.end() - 2);
            newAxies.insert(newAxies.end(), broadcastShape.end() - 2, broadcastShape.end());
            broadcastShape = ov::Shape{newAxies};
        };
        const size_t batchA = GetMatrixNumBatches(matrixAShape);
        const size_t batchB = GetMatrixNumBatches(matrixBShape);
        if (batchA > batchB) {
            broadcastNdToMd(matrixAShape, matrixBShape);
        } else if (batchA < batchB) {
            broadcastNdToMd(matrixBShape, matrixAShape);
        }
        OPENVINO_ASSERT(GetMatrixNumBatches(matrixAShape) == GetMatrixNumBatches(matrixBShape));
    }
    OPENVINO_ASSERT(*(matrixAShape.end() - transposeA - 1) == *(matrixBShape.end() - !transposeB - 1));
    if (matrixAShape.size() > matrixBShape.size()) {
        matrixCShape = matrixAShape;
    } else {
        matrixCShape = matrixBShape;
    }
    *(matrixCShape.end() - 2) = *(matrixAShape.end() - !transposeA - 1);
    *(matrixCShape.end() - 1) = *(matrixBShape.end() - transposeB - 1);
}

void MatMulOp::BroadcastToMatrix(ov::Shape& shape) {
    if (shape.size() < 2) {
        shape.insert(shape.begin(), 2 - shape.size(), 1);
    }
}

// NOTE: Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n), C is stored as row-major matrix
void MatMulOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    auto& RocBlasHandle = context.getThreadContext().rocBlasHandle();
    auto matrixA = inputs[0];
    auto matrixB = inputs[1];
    auto matrixC = outputs[0];
    hipStream_t hstream = context.getThreadContext().stream().get();

    // ── Tuned rocMLIR GEMM (plain 2D fp16) ───────────────────────────────────
    // One-time correctness check vs rocBLAS (timing-only selection could silently
    // accept a wrong kernel), then pure dispatch — no per-inference benchmark.
    if (rocmlir_kernel_ && !rocmlir_checked_) {
        auto* self = const_cast<MatMulOp*>(this);
        self->rocmlir_checked_ = true;
        const size_t cElems = (size_t)m_ * n_;
        const rocblas_datatype dtc = rocblas_datatype_f16_r, ctc = rocblas_datatype_f16_r;
        const void* one = &rocm::NumericConst<rocm::constants::one>(compute_type_);
        auto run_rb = [&]{
            rocblas_gemm_strided_batched_ex(RocBlasHandle.get(),
                rocblas_transpose_b_, rocblas_transpose_a_, n_, m_, k_, one,
                matrixB.get(), dtc, ld_b_, stride_b_, matrixA.get(), dtc, ld_a_, stride_a_,
                beta_, matrixC.get(), dtc, ld_c_, stride_c_, matrixC.get(), dtc, ld_c_, stride_c_,
                1, ctc, rocblas_gemm_algo_standard, 0, 0);
        };
        auto run_ml = [&]{
            rocmlir_gemm::launch_rocmlir_gemm(*rocmlir_kernel_, hstream,
                const_cast<void*>(matrixA.get()), const_cast<void*>(matrixB.get()), matrixC.get());
        };
        std::vector<uint16_t> ref(cElems), got(cElems);
        run_rb(); hipStreamSynchronize(hstream);
        hipMemcpy(ref.data(), matrixC.get(), cElems*2, hipMemcpyDeviceToHost);
        run_ml(); hipStreamSynchronize(hstream);
        hipMemcpy(got.data(), matrixC.get(), cElems*2, hipMemcpyDeviceToHost);
        auto h2f=[](uint16_t h)->float{ __half x; std::memcpy(&x,&h,2); return __half2float(x); };
        size_t nbad=0;
        for (size_t i=0;i<cElems;++i){ float a=h2f(ref[i]),b=h2f(got[i]);
            if (std::abs(a-b)/std::max(std::abs(a),1e-3f) > 0.02) ++nbad; }
        self->use_rocmlir_ = (nbad <= cElems/1000);
        if (!use_rocmlir_) { self->rocmlir_kernel_.reset();
            fprintf(stderr, "[MatMul] rocMLIR numeric mismatch %dx%dx%d -> rocBLAS\n", m_,n_,k_); }
        else fprintf(stderr, "[MatMul] using tuned rocMLIR GEMM %dx%dx%d\n", m_,n_,k_);
    }
    if (use_rocmlir_ && rocmlir_kernel_) {
        rocmlir_gemm::launch_rocmlir_gemm(*rocmlir_kernel_, hstream,
            const_cast<void*>(matrixA.get()), const_cast<void*>(matrixB.get()), matrixC.get());
        return;
    }

    // Map hipDataType → rocblas_datatype
    auto to_rocblas_dt = [](hipDataType dt) -> rocblas_datatype {
        switch (dt) {
            case HIP_R_16F: return rocblas_datatype_f16_r;
            case HIP_R_32F: return rocblas_datatype_f32_r;
            case HIP_R_64F: return rocblas_datatype_f64_r;
            case HIP_R_8I:  return rocblas_datatype_i8_r;
            default:        return rocblas_datatype_f32_r;
        }
    };
    const rocblas_datatype dt = to_rocblas_dt(data_type_);
    const rocblas_datatype ct = to_rocblas_dt(compute_type_);
    const void* alpha_ptr = &rocm::NumericConst<rocm::constants::one>(compute_type_);

    // rocBLAS uses column-major: compute Ct = Bt × At → C is row-major.
    // transa/transb are swapped relative to math notation.
    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    rocblas_int sol = 0;

    if (gemmTuningEnabled()) {
        GemmKey key{n_, m_, k_, batch_count_,
                    (int)rocblas_transpose_b_, (int)rocblas_transpose_a_,
                    (int)dt, (int)ct};
        sol = GemmCache::instance().get(key);
        if (sol < 0) {
            sol = tuneBatchedGemm(
                RocBlasHandle.get(),
                rocblas_transpose_b_, rocblas_transpose_a_,
                n_, m_, k_,
                alpha_ptr,
                matrixB.get(), dt, ld_b_, stride_b_,
                matrixA.get(), dt, ld_a_, stride_a_,
                beta_,
                matrixC.get(), dt, ld_c_, stride_c_,
                batch_count_, ct);
            GemmCache::instance().set(key, sol);
        }
        algo = rocblas_gemm_algo_solution_index;
    }

    throwIfError(rocblas_gemm_strided_batched_ex(
        RocBlasHandle.get(),
        rocblas_transpose_b_,   // transa for Bt
        rocblas_transpose_a_,   // transb for At
        n_,                     // m  (rows of Bt / Ct)
        m_,                     // n  (cols of At / Ct)
        k_,                     // k  (inner dimension)
        alpha_ptr,
        matrixB.get(), dt, ld_b_, stride_b_,  // B (acts as A in col-major call)
        matrixA.get(), dt, ld_a_, stride_a_,  // A (acts as B in col-major call)
        beta_,
        matrixC.get(), dt, ld_c_, stride_c_,  // C
        matrixC.get(), dt, ld_c_, stride_c_,  // D (same as C for in-place)
        batch_count_,
        ct,
        algo,
        sol,
        0)); // flags
}

rocmGraphCompatibility MatMulOp::GetrocmGraphCompatibility() const { return rocmGraphCompatibility::FULL; }

// See below (after closing namespace) for the attention-aware factory

}  // namespace rocm_gpu
}  // namespace ov

// ── Attention-aware MatMul factory (outside namespace) ───────────────────────
// When ROCM_FUSE_ATTENTION != "0" and a MatMul node has been tagged by
// RocmAttentionFusionPass (via rt_info["rocm_attn_kind"]), create
// RocmAttentionMatMulOp instead of standard rocBLAS MatMulOp.
// Set ROCM_FUSE_ATTENTION=0 to revert to standard rocBLAS (no regression).
#include "rocm_attention_matmul.hpp"
namespace {
ov::rocm_gpu::OperationBase::Ptr matmulAttentionFactory(
        const ov::rocm_gpu::CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        ov::rocm_gpu::OperationBase::IndexCollection&& inputIds,
        ov::rocm_gpu::OperationBase::IndexCollection&& outputIds)
{
    const auto& rt = node->get_rt_info();
    if (ov::rocm_gpu::RocmAttentionMatMulOp::isEnabled() && rt.count("rocm_attn_kind")) {
        try {
            return std::make_shared<ov::rocm_gpu::RocmAttentionMatMulOp>(
                context, *node,
                ov::rocm_gpu::OperationBase::IndexCollection{inputIds},
                ov::rocm_gpu::OperationBase::IndexCollection{outputIds});
        } catch (const std::exception& e) {
            std::cerr << "[AttnFusion] Fallback to rocBLAS MatMul: " << e.what() << "\n";
        }
    }
    // Cast to MatMul to access get_transpose_a/b (required by template)
    auto matmul_node = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node);
    OPENVINO_ASSERT(matmul_node, "matmulAttentionFactory: node is not MatMul");
    return std::make_shared<ov::rocm_gpu::MatMulOp>(
        context, *matmul_node,
        ov::rocm_gpu::OperationBase::IndexCollection{inputIds},
        ov::rocm_gpu::OperationBase::IndexCollection{outputIds});
}
}  // anonymous namespace
namespace ov { namespace rocm_gpu {
OPERATION_REGISTER_FACTORY(matmulAttentionFactory, MatMul);
}}  // namespace ov::rocm_gpu
