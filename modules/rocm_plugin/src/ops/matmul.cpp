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
#include <openvino/op/result.hpp>
#include <openvino/core/model.hpp>
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

// Cast int32 → float16 for rocBLAS (which doesn't support i32×i32→i32 GEMM).
// INT8-derived MatMul operands (A−zpA ∈ [−255,255], B ∈ [−127,127]) are exactly
// representable in f16; the GEMM accumulates in f32, so this is numerically
// identical to the f32-input path but runs on the much faster f16 tensor cores.
static __global__ void cast_i32_to_f16(const int* __restrict__ src, __half* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(static_cast<float>(src[i]));
}
static __global__ void cast_i32_to_f32(const int* __restrict__ src, float* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<float>(src[i]);
}
static __global__ void cast_f32_to_i32(const float* __restrict__ src, int* __restrict__ dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int>(rintf(src[i]));
}

// Verification/fallback switch: ROCM_I32_GEMM_PREC=f32 forces the legacy f32-input
// GEMM path so the f16 path can be A/B compared. Default (unset) = f16.
static bool i32GemmUseF16() {
    static const bool v = [] {
        const char* p = std::getenv("ROCM_I32_GEMM_PREC");
        return !(p && std::string(p) == "f32");
    }();
    return v;
}

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
    bool dirty_ = false;  // written since last save

    static GemmCache& instance() {
        static GemmCache g;
        static bool loaded = false;
        if (!loaded) { loaded = true; g.load(); }
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
        dirty_ = true;
    }

    // Load cache from file at ROCM_TUNE_GEMM_CACHE path (binary: N × (8 ints + 1 int sol))
    void load() {
        const char* path = std::getenv("ROCM_TUNE_GEMM_CACHE");
        if (!path) return;
        FILE* f = fopen(path, "rb");
        if (!f) return;
        GemmKey k;
        rocblas_int sol;
        while (fread(&k, sizeof(k), 1, f) == 1 && fread(&sol, sizeof(sol), 1, f) == 1)
            best[k] = sol;
        fclose(f);
        fprintf(stderr, "[ROCM_TUNE_GEMM] Loaded %zu solutions from %s\n", best.size(), path);
    }

    void save() {
        const char* path = std::getenv("ROCM_TUNE_GEMM_CACHE");
        if (!path || !dirty_) return;
        FILE* f = fopen(path, "wb");
        if (!f) return;
        for (const auto& [k, sol] : best) {
            fwrite(&k, sizeof(k), 1, f);
            fwrite(&sol, sizeof(sol), 1, f);
        }
        fclose(f);
        dirty_ = false;
        fprintf(stderr, "[ROCM_TUNE_GEMM] Saved %zu solutions to %s\n", best.size(), path);
    }

    ~GemmCache() { save(); }
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
    auto out_dt = convertDataType<hipDataType>(op.get_output_element_type(0));
    // rocBLAS does not support i32×i32→i32 GEMM. INT8-derived operands are small
    // integers (A−zpA ∈ [−255,255], B ∈ [−127,127]) and exactly representable in
    // f16. Cast A,B to f16 and accumulate in f32 — same result as f32 inputs but
    // on the much faster f16 tensor cores. Output cast back to i32 at Execute time.
    if (data_type_ == HIP_R_32I) {
        needs_i32_cast_ = true;
        data_type_ = i32GemmUseF16() ? HIP_R_16F : HIP_R_32F;
        // If MatMulDequantConvertPass already retyped this MatMul's output to f32
        // (eliminating the downstream Convert(i32->f32)), write the f32 GEMM result
        // straight to the op output — skip the cast_f32_to_i32 round-trip entirely.
        i32_out_is_f32_ = (op.get_output_element_type(0) == ov::element::f32);
        out_dt = HIP_R_32F;
    }
    compute_type_ = GetComputeType(data_type_, out_dt);
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
    if (needs_i32_cast_) {
        i32_use_f16_ = i32GemmUseF16();
        const size_t ab_elem = i32_use_f16_ ? sizeof(__half) : sizeof(float);
        a_elems_ = ov::shape_size(op.get_input_shape(0));
        b_elems_ = ov::shape_size(op.get_input_shape(1));
        c_elems_ = ov::shape_size(op.get_output_shape(0));
        auto err_a = hipMalloc(&d_a_ab_, a_elems_ * ab_elem);
        OPENVINO_ASSERT(err_a == hipSuccess && d_a_ab_ != nullptr,
            "hipMalloc failed for i32 GEMM A buffer: ", hipGetErrorString(err_a));
        // When the output stays i32, the GEMM accumulates into a private f32 buffer
        // then casts to i32. When the Convert was eliminated (output is f32), the
        // GEMM writes straight to the op output and no scratch buffer is needed.
        if (!i32_out_is_f32_) {
            auto err_c = hipMalloc(&d_c_f32_, c_elems_ * sizeof(float));
            OPENVINO_ASSERT(err_c == hipSuccess && d_c_f32_ != nullptr,
                "hipMalloc failed for i32 GEMM f32 C buffer: ", hipGetErrorString(err_c));
        }

        // Try to pre-cast constant B (weight) at compile time — avoids cast kernel per inference.
        // B is constant if its upstream subgraph consists only of Constant/Parameter-free ops.
        // Quantized weight subgraphs (DequantizeLinear → Convert/Subtract/Multiply/...)
        // can be deeper than a handful of hops, so use a generous depth limit. Reaching
        // a Parameter means genuinely dynamic → not const. Empty-input nodes that aren't
        // Constant (shouldn't happen for a pure const subgraph) → treat as non-const.
        auto is_const_subgraph = [](const auto& self, const ov::Node* n, int depth) -> bool {
            if (n->get_type_name() == std::string("Constant")) return true;
            if (n->get_type_name() == std::string("Parameter"))  return false;
            if (depth > 32) return false;
            if (n->get_input_size() == 0) return false;
            for (size_t i = 0; i < n->get_input_size(); ++i)
                if (!self(self, n->input(i).get_source_output().get_node_shared_ptr().get(), depth + 1))
                    return false;
            return true;
        };
        const auto* b_node = op.get_input_node_ptr(1);
        b_is_const_ = is_const_subgraph(is_const_subgraph, b_node, 0);
        if (std::getenv("ROCM_I32_NO_CONST_B")) b_is_const_ = false;  // bisect: force runtime cast
        if (std::getenv("ROCM_I32_DEBUG")) {
            auto bshape = op.get_input_shape(1);
            std::string sh; for (auto d : bshape) sh += std::to_string(d) + ",";
            fprintf(stderr, "[Bsrc] %s B_node=%s shape=[%s] b_elems=%zu bConst=%d\n",
                    GetName().c_str(), b_node->get_type_name(), sh.c_str(), b_elems_, (int)b_is_const_);
            fflush(stderr);
        }

        hipMalloc(&d_b_ab_, b_elems_ * ab_elem);
        if (b_is_const_) {
            // Materialize B's i32 values on host, cast to f16/f32, upload once (permanent).
            // PREFER reading a direct Constant node via cast_vector (no evaluate()):
            // these weights are commonly a folded Constant, and Model::evaluate() can
            // THROW on this build (missing CPU-reference for some op) — that throw was
            // silently turning b_is_const_ off and forcing a runtime cast that read B
            // out of bounds → GPU fault/hang. Fall back to evaluate() only for deeper
            // const subgraphs; verify size == b_elems_ before use.
            bool precast_ok = false;
            try {
                std::vector<int32_t> bvals;
                if (auto bc = dynamic_cast<const ov::op::v0::Constant*>(b_node)) {
                    bvals = bc->cast_vector<int32_t>();   // direct constant, no evaluate
                } else {
                    ov::TensorVector out_tensors;  // empty — evaluate allocates
                    auto b_model_output = std::make_shared<ov::op::v0::Result>(op.input_value(1));
                    auto b_subgraph = std::make_shared<ov::Model>(ov::ResultVector{b_model_output},
                                                                  ov::ParameterVector{});
                    if (b_subgraph->evaluate(out_tensors, {}) && out_tensors.size() == 1 &&
                        out_tensors[0].get_element_type() == ov::element::i32) {
                        const int32_t* p = out_tensors[0].data<int32_t>();
                        bvals.assign(p, p + out_tensors[0].get_size());
                    }
                }
                if (std::getenv("ROCM_I32_DEBUG")) {
                    fprintf(stderr, "[precast] %s got=%zu expect=%zu\n",
                            GetName().c_str(), bvals.size(), b_elems_); fflush(stderr);
                }
                if (bvals.size() == b_elems_) {
                    if (i32_use_f16_) {
                        std::vector<__half> b_f16(b_elems_);
                        for (size_t i = 0; i < b_elems_; ++i) b_f16[i] = __float2half(static_cast<float>(bvals[i]));
                        precast_ok = (hipMemcpy(d_b_ab_, b_f16.data(), b_elems_ * sizeof(__half),
                                                hipMemcpyHostToDevice) == hipSuccess);
                    } else {
                        std::vector<float> b_f32(b_elems_);
                        for (size_t i = 0; i < b_elems_; ++i) b_f32[i] = static_cast<float>(bvals[i]);
                        precast_ok = (hipMemcpy(d_b_ab_, b_f32.data(), b_elems_ * sizeof(float),
                                                hipMemcpyHostToDevice) == hipSuccess);
                    }
                }
            } catch (...) { precast_ok = false; }
            if (!precast_ok) b_is_const_ = false;  // fall back: cast B at Execute time
        }
        // else: B is dynamic — buffer allocated, cast at Execute time
    }

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

    // INT8 path: for the i32 GEMM (cast to f16), route through hipBLASLt to avoid a
    // rocBLAS gemm_strided_batched_ex GPU hang on gfx1201 for certain shapes.
    if (needs_i32_cast_ && i32_use_f16_)
        setup_hipblaslt_i32();

    // fp16 path: optional tuned rocMLIR GEMM for plain 2D fp16, single batch, no
    // A-transpose, beta==0. Decided by tuning-cache presence; one-time numeric check
    // vs rocBLAS in Execute. Mutually exclusive with the INT8 i32 path (gated on dtype).
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

// Set up a hipBLASLt matmul for the i32 (cast-to-f16) GEMM. Mirrors the col-major
// convention used elsewhere: rocBLAS computes Ct[N,M] = Bt[N,K] × At[K,M], i.e. the
// hipBLASLt "A" operand is our B (f16, ldB, transB), "B" operand is our A (f16, ldA,
// transA), output C is f32 (ldC). Batched dims use BATCH_COUNT + STRIDED_BATCH_OFFSET.
void MatMulOp::setup_hipblaslt_i32() {
    do {
        if (hipblasLtCreate(&lt_handle_) != HIPBLAS_STATUS_SUCCESS) break;
        if (hipblasLtMatmulDescCreate(&lt_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                != HIPBLAS_STATUS_SUCCESS) break;

        // transa applies to our B operand, transb to our A operand (col-major swap).
        hipblasOperation_t ta = (rocblas_transpose_b_ == rocblas_operation_transpose)
                                    ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        hipblasOperation_t tb = (rocblas_transpose_a_ == rocblas_operation_transpose)
                                    ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        hipblasLtMatmulDescSetAttribute(lt_desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));

        // Layouts: B(f16) as "A": rows=n_, cols=k_, ld=ld_b_; A(f16) as "B": rows=k_,
        // cols=m_, ld=ld_a_; C(f32): rows=n_, cols=m_, ld=ld_c_. For transposed
        // operands the layout's (rows,cols) is the physical storage, ld stays.
        auto mk = [](hipblasLtMatrixLayout_t& L, hipDataType dt, int rows, int cols, int ld,
                     hipblasOperation_t tr) -> bool {
            int r = (tr == HIPBLAS_OP_T) ? cols : rows;
            int c = (tr == HIPBLAS_OP_T) ? rows : cols;
            return hipblasLtMatrixLayoutCreate(&L, dt, r, c, ld) == HIPBLAS_STATUS_SUCCESS;
        };
        if (!mk(lt_lb_, HIP_R_16F, n_, k_, ld_b_, ta)) break;
        if (!mk(lt_la_, HIP_R_16F, k_, m_, ld_a_, tb)) break;
        if (hipblasLtMatrixLayoutCreate(&lt_lc_, HIP_R_32F, n_, m_, ld_c_)
                != HIPBLAS_STATUS_SUCCESS) break;

        if (batch_count_ > 1) {
            int32_t bc = batch_count_;
            int64_t sB = stride_b_, sA = stride_a_, sC = stride_c_;
            for (auto L : {lt_lb_, lt_la_, lt_lc_})
                hipblasLtMatrixLayoutSetAttribute(L, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &bc, sizeof(bc));
            hipblasLtMatrixLayoutSetAttribute(lt_lb_,
                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sB, sizeof(sB));
            hipblasLtMatrixLayoutSetAttribute(lt_la_,
                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA, sizeof(sA));
            hipblasLtMatrixLayoutSetAttribute(lt_lc_,
                HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC, sizeof(sC));
        }

        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);
        lt_workspace_bytes_ = 32 * 1024 * 1024;
        hipblasLtMatmulPreferenceSetAttribute(pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &lt_workspace_bytes_, sizeof(lt_workspace_bytes_));

        hipblasLtMatmulHeuristicResult_t result;
        int returned = 0;
        auto st = hipblasLtMatmulAlgoGetHeuristic(lt_handle_, lt_desc_,
            lt_lb_, lt_la_, lt_lc_, lt_lc_, pref, 1, &result, &returned);
        hipblasLtMatmulPreferenceDestroy(pref);
        if (st != HIPBLAS_STATUS_SUCCESS || returned == 0) break;
        lt_algo_ = result.algo;
        if (lt_workspace_bytes_ > 0) hipMalloc(&lt_workspace_, lt_workspace_bytes_);

        use_lt_i32_ = true;
        if (std::getenv("ROCM_I32_DEBUG"))
            fprintf(stderr, "[i32 hipBLASLt] %s m=%d n=%d k=%d batch=%d OK\n",
                    GetName().c_str(), m_, n_, k_, batch_count_);
    } while (false);

    if (!use_lt_i32_ && std::getenv("ROCM_I32_DEBUG"))
        fprintf(stderr, "[i32 hipBLASLt] %s setup FAILED — falling back to rocBLAS\n",
                GetName().c_str());
}

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
        case SwitchCase(HIP_R_8I, HIP_R_32I):
        case SwitchCase(HIP_R_32I, HIP_R_32I): {
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
    const auto& stream = context.getThreadContext().stream();

    // ── Tuned rocMLIR GEMM (plain 2D fp16; mutually exclusive with INT8 i32) ──
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
            rocmlir_gemm::launch_rocmlir_gemm(*rocmlir_kernel_, stream.get(),
                const_cast<void*>(matrixA.get()), const_cast<void*>(matrixB.get()), matrixC.get());
        };
        std::vector<uint16_t> ref(cElems), got(cElems);
        run_rb(); hipStreamSynchronize(stream.get());
        hipMemcpy(ref.data(), matrixC.get(), cElems*2, hipMemcpyDeviceToHost);
        run_ml(); hipStreamSynchronize(stream.get());
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
        rocmlir_gemm::launch_rocmlir_gemm(*rocmlir_kernel_, stream.get(),
            const_cast<void*>(matrixA.get()), const_cast<void*>(matrixB.get()), matrixC.get());
        return;
    }

    if (needs_i32_cast_ && std::getenv("ROCM_I32_DEBUG")) {
        static int idx = 0;
        fprintf(stderr, "[i32MM #%d] %s m=%d n=%d k=%d batch=%d ldA=%d ldB=%d ldC=%d "
                "strideA=%lld strideB=%lld strideC=%lld a_elems=%zu b_elems=%zu c_elems=%zu f16=%d bConst=%d\n",
                idx++, GetName().c_str(), m_, n_, k_, batch_count_, ld_a_, ld_b_, ld_c_,
                (long long)stride_a_, (long long)stride_b_, (long long)stride_c_,
                a_elems_, b_elems_, c_elems_, (int)i32_use_f16_, (int)b_is_const_);
        hipStreamSynchronize(stream.get());
    }

    // For i32 GEMM: cast i32→f16/f32 (A,B) using pre-allocated device buffers, run an
    // f32-accumulate GEMM into d_c_f32_, then cast the f32 result back to i32.
    if (needs_i32_cast_) {
        constexpr unsigned kBlock = 256;
        auto cast_a = [&](size_t n, const void* src, void* dst) {
            const unsigned g = (n + kBlock - 1) / kBlock;
            if (i32_use_f16_)
                cast_i32_to_f16<<<g, kBlock, 0, stream.get()>>>((const int*)src, (__half*)dst, n);
            else
                cast_i32_to_f32<<<g, kBlock, 0, stream.get()>>>((const int*)src, (float*)dst, n);
        };
        if (std::getenv("ROCM_I32_DEBUG")) {
            fprintf(stderr, "[i32 pre-castA] %s A=%p a_elems=%zu d_a=%p\n",
                    GetName().c_str(), matrixA.get(), a_elems_, d_a_ab_); fflush(stderr);
        }
        cast_a(a_elems_, matrixA.get(), d_a_ab_);
        if (std::getenv("ROCM_I32_DEBUG")) {
            auto e = hipStreamSynchronize(stream.get());
            fprintf(stderr, "[i32 castA done] %s %s\n", GetName().c_str(),
                    e==hipSuccess?"OK":hipGetErrorString(e)); fflush(stderr);
        }
        if (!b_is_const_) cast_a(b_elems_, matrixB.get(), d_b_ab_);
        // b_is_const_: d_b_ab_ pre-loaded at compile time, skip cast
    }

    if (needs_i32_cast_ && std::getenv("ROCM_I32_DEBUG")) {
        auto e = hipStreamSynchronize(stream.get());
        fprintf(stderr, "[i32 cast-AB done] %s %s\n", GetName().c_str(),
                e == hipSuccess ? "OK" : hipGetErrorString(e)); fflush(stderr);
    }

    const void* pA = needs_i32_cast_ ? (const void*)d_a_ab_ : matrixA.get();
    const void* pB = needs_i32_cast_ ? (const void*)d_b_ab_ : matrixB.get();
    // i32 path normally accumulates into d_c_f32_ then casts to i32. If the output
    // was retyped to f32 (Convert eliminated), write the f32 result to matrixC directly.
    void*       pC = (needs_i32_cast_ && !i32_out_is_f32_) ? (void*)d_c_f32_ : matrixC.get();

    // Map hipDataType → rocblas_datatype
    auto to_rocblas_dt = [](hipDataType dt) -> rocblas_datatype {
        switch (dt) {
            case HIP_R_16F: return rocblas_datatype_f16_r;
            case HIP_R_32F: return rocblas_datatype_f32_r;
            case HIP_R_64F: return rocblas_datatype_f64_r;
            case HIP_R_8I:  return rocblas_datatype_i8_r;
            case HIP_R_32I: return rocblas_datatype_i32_r;
            default:        return rocblas_datatype_f32_r;
        }
    };
    const rocblas_datatype dt = to_rocblas_dt(data_type_);
    // C/D datatype: for the i32 path A,B are f16 but C accumulates in f32; otherwise C == A,B type.
    const rocblas_datatype dt_c = needs_i32_cast_ ? rocblas_datatype_f32_r : dt;
    const rocblas_datatype ct = to_rocblas_dt(compute_type_);
    const void* alpha_ptr = &rocm::NumericConst<rocm::constants::one>(compute_type_);

    // rocBLAS uses column-major: compute Ct = Bt × At → C is row-major.
    // transa/transb are swapped relative to math notation.
    rocblas_gemm_algo algo = rocblas_gemm_algo_standard;
    rocblas_int sol = 0;

    if (gemmTuningEnabled() && !needs_i32_cast_) {
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

    if (use_lt_i32_) {
        // hipBLASLt path for the i32 (cast-to-f16) GEMM — avoids the rocBLAS hang.
        // f32 alpha/beta (compute type is f32); C accumulates in f32 (d_c_f32_).
        const float lt_alpha = 1.0f, lt_beta = 0.0f;
        auto st = hipblasLtMatmul(lt_handle_, lt_desc_, &lt_alpha,
            pB, lt_lb_, pA, lt_la_, &lt_beta,
            pC, lt_lc_, pC, lt_lc_,
            &lt_algo_, lt_workspace_, lt_workspace_bytes_, stream.get());
        OPENVINO_ASSERT(st == HIPBLAS_STATUS_SUCCESS,
            "hipBLASLt i32 GEMM failed, status=", (int)st, ", node: ", GetName());
    } else {
        throwIfError(rocblas_gemm_strided_batched_ex(
            RocBlasHandle.get(),
            rocblas_transpose_b_,   // transa for Bt
            rocblas_transpose_a_,   // transb for At
            n_,                     // m  (rows of Bt / Ct)
            m_,                     // n  (cols of At / Ct)
            k_,                     // k  (inner dimension)
            alpha_ptr,
            pB, dt, ld_b_, stride_b_,  // B (acts as A in col-major call)
            pA, dt, ld_a_, stride_a_,  // A (acts as B in col-major call)
            beta_,
            pC, dt_c, ld_c_, stride_c_,  // C
            pC, dt_c, ld_c_, stride_c_,  // D (same as C for in-place)
            batch_count_,
            ct,
            algo,
            sol,
            0)); // flags
    }

    if (needs_i32_cast_ && !i32_out_is_f32_) {
        constexpr unsigned kBlock = 256;
        cast_f32_to_i32<<<(c_elems_ + kBlock - 1) / kBlock, kBlock, 0, stream.get()>>>(
            d_c_f32_, (int*)matrixC.get(), c_elems_);
    }

    if (needs_i32_cast_ && std::getenv("ROCM_I32_DEBUG")) {
        auto e = hipStreamSynchronize(stream.get());
        fprintf(stderr, "[i32MM DONE] %s sync=%s\n", GetName().c_str(),
                e == hipSuccess ? "OK" : hipGetErrorString(e));
    }
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
