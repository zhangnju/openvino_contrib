// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// rocMLIR rock.gemm kernel compilation and backend auto-selection for FC layers.
// Benchmarks rocMLIR vs rocBLAS at model load time and caches the faster backend.
// Supports epilogue fusion: GEMM-only, GEMM+bias, GEMM+bias+GELU.
#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace ov {
namespace rocm_gpu {
namespace rocmlir_gemm {

// Epilogue operations fused into the GEMM kernel
enum class Epilogue {
    None,     // C = A * B
    Bias,     // C = A * B + bias[n]   (bias broadcast over M rows)
    BiasGELU, // C = GELU(A * B + bias[n])
};

enum class Backend { ROCBLAS, ROCMLIR };

struct GemmKernel {
    std::vector<char> hsaco;
    std::string       func_name;
    hipModule_t       module{nullptr};
    hipFunction_t     func{nullptr};
    unsigned          grid_x{0};
    unsigned          block_x{0};
    Epilogue          epilogue{Epilogue::None};

    ~GemmKernel() { if (module) hipModuleUnload(module); }
    GemmKernel() = default;
    GemmKernel(const GemmKernel&) = delete;
    GemmKernel& operator=(const GemmKernel&) = delete;
    GemmKernel(GemmKernel&& o)
        : hsaco(std::move(o.hsaco)), func_name(std::move(o.func_name))
        , module(o.module), func(o.func)
        , grid_x(o.grid_x), block_x(o.block_x), epilogue(o.epilogue) {
        o.module = nullptr;
    }
};

struct SelectResult {
    Backend     backend;
    std::shared_ptr<GemmKernel> kernel;  // non-null only when backend == ROCMLIR
};

// Compile a rock.gemm f16 kernel with optional epilogue:
//   None:     args = (A[M,K], B[N,K], C[M,N])
//   Bias:     args = (A[M,K], B[N,K], bias[N], C[M,N])
//   BiasGELU: args = (A[M,K], B[N,K], bias[N], C[M,N])
// transB=true means B is stored [N,K] (transposed FC weights).
// Returns nullopt on failure (falls back to rocBLAS + separate bias kernel).
std::optional<GemmKernel> compile_rocmlir_gemm(
    int M, int N, int K, bool transB,
    const std::string& arch, int num_cu,
    Epilogue epilogue = Epilogue::None);

// Benchmark rocBLAS vs rocMLIR (no-epilogue) and return the faster backend.
// Results are cached globally per shape.
SelectResult select_backend(
    rocblas_handle handle, hipStream_t stream,
    int M, int N, int K, bool transB,
    const void* A, const void* B, void* C,
    const std::string& arch, int num_cu);

// Launch a rocMLIR kernel.
// For Epilogue::None:     args = (A, B, C)
// For Epilogue::Bias/BiasGELU: args = (A, B, bias, C)
void launch_rocmlir_gemm(
    const GemmKernel& kernel, hipStream_t stream,
    void* A, void* B, void* C);

void launch_rocmlir_gemm_bias(
    const GemmKernel& kernel, hipStream_t stream,
    void* A, void* B, void* bias, void* C);

}  // namespace rocmlir_gemm
}  // namespace rocm_gpu
}  // namespace ov
