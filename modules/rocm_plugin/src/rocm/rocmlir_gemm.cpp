// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rocmlir_gemm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <fmt/format.h>

namespace ov {
namespace rocm_gpu {
namespace rocmlir_gemm {

namespace {

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string find_tool(const char* name) {
    for (const char* dir : {"/home/rocmlir_install/bin/", "/opt/rocm/bin/"}) {
        std::string p = std::string(dir) + name;
        if (access(p.c_str(), X_OK) == 0) return p;
    }
    return name;
}

static std::string run_cmd(const std::string& cmd, int& ec) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) { ec = -1; return ""; }
    std::string out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    ec = pclose(pipe);
    return out;
}

static std::vector<char> extract_hsaco(const std::string& compiled) {
    const std::string marker = "bin = \"";
    auto pos = compiled.find(marker);
    if (pos == std::string::npos) return {};
    std::vector<char> bytes;
    size_t i = pos + marker.size();
    while (i < compiled.size()) {
        char c = compiled[i];
        if (c == '"') break;
        if (c == '\\' && i + 1 < compiled.size()) {
            char nx = compiled[i+1];
            if (nx == 'n')  { bytes.push_back('\n'); i += 2; continue; }
            if (nx == '\\') { bytes.push_back('\\'); i += 2; continue; }
            if (i+2 < compiled.size() &&
                std::isxdigit((unsigned char)compiled[i+1]) &&
                std::isxdigit((unsigned char)compiled[i+2])) {
                char h[3] = {compiled[i+1], compiled[i+2], 0};
                bytes.push_back((char)std::strtol(h, nullptr, 16));
                i += 3; continue;
            }
        }
        bytes.push_back(c); ++i;
    }
    return bytes;
}

static unsigned parse_uint(const std::string& s, const std::string& marker) {
    auto p = s.find(marker);
    return (p != std::string::npos) ? (unsigned)std::stoi(s.substr(p + marker.size(), 20)) : 0;
}

// Median of N kernel timings (us)
template<typename Fn>
static double median_us(hipStream_t stream, Fn fn, int warmup=5, int reps=20) {
    for (int i = 0; i < warmup; ++i) fn();
    hipStreamSynchronize(stream);
    std::vector<float> ms(reps);
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i = 0; i < reps; ++i) {
        hipEventRecord(t0, stream);
        fn();
        hipEventRecord(t1, stream);
        hipStreamSynchronize(stream);
        hipEventElapsedTime(&ms[i], t0, t1);
    }
    hipEventDestroy(t0); hipEventDestroy(t1);
    std::sort(ms.begin(), ms.end());
    return (double)ms[reps/2] * 1000.0;
}

// ── Cache key ─────────────────────────────────────────────────────────────────

struct ShapeKey {
    int M, N, K;
    bool transB;
    std::string arch;
    bool operator==(const ShapeKey& o) const {
        return M==o.M && N==o.N && K==o.K && transB==o.transB && arch==o.arch;
    }
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const {
        size_t h = std::hash<std::string>{}(k.arch);
        auto mix = [&](int v) {
            h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h<<6) + (h>>2);
        };
        mix(k.M); mix(k.N); mix(k.K); mix((int)k.transB);
        return h;
    }
};

// In-process compile cache: shape → shared_ptr<GemmKernel>
// Ensures each unique shape is compiled only once per process.
static std::mutex g_compile_mu;
static std::unordered_map<ShapeKey, std::shared_ptr<GemmKernel>, ShapeKeyHash> g_compile_cache;

// In-process backend selection cache: shape → SelectResult
static std::mutex g_select_mu;
static std::unordered_map<ShapeKey, SelectResult, ShapeKeyHash> g_select_cache;

// ── Disk cache ────────────────────────────────────────────────────────────────
// Layout: ~/.cache/ov_rocmlir_gemm_<arch>/gemm_<M>_<N>_<K>_tB<0|1>.bin
//         + ~/.cache/ov_rocmlir_gemm_<arch>/gemm_<M>_<N>_<K>_tB<0|1>.meta
//
// meta file format (text):
//   func_name=<name>
//   grid_x=<n>
//   block_x=<n>
//   winner=rocmlir|rocblas   (backend that won the benchmark)

static std::string cache_dir(const std::string& arch) {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "/tmp") +
           "/.cache/ov_rocmlir_gemm_" + arch;
}

static std::string hsaco_path(const std::string& dir, int M, int N, int K, bool tB) {
    return fmt::format("{}/gemm_{}_{}_{}_tB{}.bin", dir, M, N, K, (int)tB);
}
static std::string meta_path(const std::string& dir, int M, int N, int K, bool tB) {
    return fmt::format("{}/gemm_{}_{}_{}_tB{}.meta", dir, M, N, K, (int)tB);
}

static void ensure_dir(const std::string& dir) {
    mkdir(dir.c_str(), 0755);
}

// Save compiled kernel + winner to disk
static void disk_save(const std::string& arch, int M, int N, int K, bool tB,
                      const GemmKernel& kernel, Backend winner) {
    const std::string dir = cache_dir(arch);
    ensure_dir(dir);
    // Save HSACO binary
    std::ofstream bin(hsaco_path(dir, M, N, K, tB), std::ios::binary);
    if (bin) bin.write(kernel.hsaco.data(), kernel.hsaco.size());
    // Save metadata
    std::ofstream meta(meta_path(dir, M, N, K, tB));
    if (meta) {
        meta << "func_name=" << kernel.func_name << "\n";
        meta << "grid_x=" << kernel.grid_x << "\n";
        meta << "block_x=" << kernel.block_x << "\n";
        meta << "winner=" << (winner == Backend::ROCMLIR ? "rocmlir" : "rocblas") << "\n";
    }
}

// Try to load from disk. Returns the loaded kernel (with HIP module loaded) or nullopt.
// winner_out is set to the cached backend selection.
static std::optional<GemmKernel> disk_load(const std::string& arch, int M, int N, int K, bool tB,
                                            Backend& winner_out) {
    const std::string dir = cache_dir(arch);
    const std::string bin_p  = hsaco_path(dir, M, N, K, tB);
    const std::string meta_p = meta_path(dir, M, N, K, tB);

    // Read HSACO
    std::ifstream bin(bin_p, std::ios::binary | std::ios::ate);
    if (!bin) return std::nullopt;
    auto sz = bin.tellg(); bin.seekg(0);
    if (sz < 100) return std::nullopt;
    std::vector<char> hsaco(sz);
    bin.read(hsaco.data(), sz);

    // Read metadata
    std::ifstream meta(meta_p);
    if (!meta) return std::nullopt;
    std::string func_name;
    unsigned grid_x = 0, block_x = 128;
    Backend winner = Backend::ROCMLIR;
    std::string line;
    while (std::getline(meta, line)) {
        if (line.substr(0, 10) == "func_name=") func_name = line.substr(10);
        else if (line.substr(0, 7) == "grid_x=") grid_x = std::stoi(line.substr(7));
        else if (line.substr(0, 8) == "block_x=") block_x = std::stoi(line.substr(8));
        else if (line.substr(0, 7) == "winner=") winner = (line.substr(7) == "rocmlir") ? Backend::ROCMLIR : Backend::ROCBLAS;
    }
    if (func_name.empty() || grid_x == 0) return std::nullopt;

    // Load into HIP module
    GemmKernel kernel;
    kernel.hsaco     = std::move(hsaco);
    kernel.func_name = func_name;
    kernel.grid_x    = grid_x;
    kernel.block_x   = block_x;

    hipError_t herr = hipModuleLoadData(&kernel.module, kernel.hsaco.data());
    if (herr != hipSuccess) return std::nullopt;
    herr = hipModuleGetFunction(&kernel.func, kernel.module, kernel.func_name.c_str());
    if (herr != hipSuccess) {
        hipModuleUnload(kernel.module);
        kernel.module = nullptr;
        return std::nullopt;
    }

    winner_out = winner;
    fprintf(stderr, "[rocmlir_gemm] disk-cache hit %dx%dx%d tB=%d winner=%s\n",
            M, N, K, (int)tB, winner == Backend::ROCMLIR ? "rocmlir" : "rocblas");
    return kernel;
}

}  // namespace

// ── MLIR generation ───────────────────────────────────────────────────────────

// Generate MLIR for rock.gemm with optional epilogue (bias add / bias+GELU).
// The epilogue is expressed as a linalg.generic immediately following rock.gemm
// in the same function — rocmlir-driver compiles this to a single GPU kernel.
static std::string make_gemm_mlir(int M, int N, int K, bool transB,
                                   const std::string& arch, Epilogue epilogue) {
    std::ostringstream s;

    // Affine maps for the epilogue linalg.generic
    s << "#map_bias = affine_map<(d0, d1) -> (d1)>\n";
    s << "#map_out  = affine_map<(d0, d1) -> (d0, d1)>\n";
    s << fmt::format("module attributes {{mhal.arch = \"amdgcn-amd-amdhsa:{}\"}} {{\n", arch);

    // Function signature depends on epilogue
    const std::string func_base = epilogue == Epilogue::None
        ? fmt::format("  func.func @rock_gemm(%A: memref<{}x{}xf16>, %B: memref<{}x{}xf16>, "
                      "%C: memref<{}x{}xf16>)",
                      M, K, transB ? N : K, transB ? K : N, M, N)
        : fmt::format("  func.func @rock_gemm_bias(%A: memref<{}x{}xf16>, %B: memref<{}x{}xf16>, "
                      "%bias: memref<{}xf16>, %C: memref<{}x{}xf16>)",
                      M, K, transB ? N : K, transB ? K : N, N, M, N);

    s << func_base;
    s << fmt::format(" attributes {{kernel, mhal.arch = \"amdgcn-amd-amdhsa:{}\", num_cu = 64 : i32}} {{\n", arch);

    // rock.gemm: C = A * B (or A * tr(B) when transB)
    const std::string B_type = transB
        ? fmt::format("memref<{}x{}xf16>", N, K)
        : fmt::format("memref<{}x{}xf16>", K, N);
    const std::string tr_op = transB ? "tr %B" : "%B";
    s << fmt::format("    rock.gemm %C = %A * {}\n", tr_op);
    s << "      features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>\n";
    s << "      storeMethod = #rock<StoreMethod set>\n";
    s << fmt::format("      : memref<{}x{}xf16> = memref<{}x{}xf16> * {}\n", M, N, M, K, B_type);

    if (epilogue == Epilogue::Bias) {
        // Fused bias add: C[i,j] = C[i,j] + bias[j]
        s << fmt::format("    linalg.generic {{\n");
        s << "      indexing_maps = [#map_bias, #map_out],\n";
        s << "      iterator_types = [\"parallel\", \"parallel\"]\n";
        s << fmt::format("    }} ins(%bias : memref<{}xf16>) outs(%C : memref<{}x{}xf16>) {{\n", N, M, N);
        s << "    ^bb0(%b: f16, %c: f16):\n";
        s << "      %r = arith.addf %b, %c : f16\n";
        s << "      linalg.yield %r : f16\n";
        s << "    }\n";

    } else if (epilogue == Epilogue::BiasGELU) {
        // Fused bias + GELU: C[i,j] = GELU(C[i,j] + bias[j])
        s << fmt::format("    linalg.generic {{\n");
        s << "      indexing_maps = [#map_bias, #map_out],\n";
        s << "      iterator_types = [\"parallel\", \"parallel\"]\n";
        s << fmt::format("    }} ins(%bias : memref<{}xf16>) outs(%C : memref<{}x{}xf16>) {{\n", N, M, N);
        s << "    ^bb0(%b: f16, %c: f16):\n";
        s << "      %x    = arith.addf %b, %c : f16\n";
        s << "      %x32  = arith.extf %x : f16 to f32\n";
        s << "      %k    = arith.constant 0.7071067811865476 : f32\n";
        s << "      %half = arith.constant 0.5 : f32\n";
        s << "      %one  = arith.constant 1.0 : f32\n";
        s << "      %xn   = arith.mulf %x32, %k : f32\n";
        s << "      %e    = math.erf %xn : f32\n";
        s << "      %ep   = arith.addf %e, %one : f32\n";
        s << "      %t    = arith.mulf %half, %ep : f32\n";
        s << "      %g    = arith.mulf %x32, %t : f32\n";
        s << "      %r    = arith.truncf %g : f32 to f16\n";
        s << "      linalg.yield %r : f16\n";
        s << "    }\n";
    }

    s << "    return\n  }\n}\n";
    return s.str();
}

// ── Public API ────────────────────────────────────────────────────────────────

std::optional<GemmKernel> compile_rocmlir_gemm(
        int M, int N, int K, bool transB,
        const std::string& arch, int /*num_cu*/,
        Epilogue epilogue) {

    // Encode epilogue into cache key via arch suffix
    const std::string arch_key = arch + "_ep" + std::to_string((int)epilogue);
    const ShapeKey key{M, N, K, transB, arch_key};

    // ── In-process compile cache ───────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(g_compile_mu);
        auto it = g_compile_cache.find(key);
        if (it != g_compile_cache.end()) {
            if (!it->second) return std::nullopt;
            GemmKernel copy;
            copy.hsaco     = it->second->hsaco;
            copy.func_name = it->second->func_name;
            copy.grid_x    = it->second->grid_x;
            copy.block_x   = it->second->block_x;
            copy.epilogue  = it->second->epilogue;
            hipModuleLoadData(&copy.module, copy.hsaco.data());
            hipModuleGetFunction(&copy.func, copy.module, copy.func_name.c_str());
            return copy;
        }
    }

    // ── Disk cache ─────────────────────────────────────────────────────────────
    {
        Backend dummy;
        auto from_disk = disk_load(arch_key, M, N, K, transB, dummy);
        if (from_disk) {
            from_disk->epilogue = epilogue;
            auto ptr = std::make_shared<GemmKernel>(std::move(*from_disk));
            {
                std::lock_guard<std::mutex> lk(g_compile_mu);
                g_compile_cache[key] = ptr;
            }
            GemmKernel copy;
            copy.hsaco     = ptr->hsaco;
            copy.func_name = ptr->func_name;
            copy.grid_x    = ptr->grid_x;
            copy.block_x   = ptr->block_x;
            copy.epilogue  = epilogue;
            hipModuleLoadData(&copy.module, copy.hsaco.data());
            hipModuleGetFunction(&copy.func, copy.module, copy.func_name.c_str());
            return copy;
        }
    }

    // ── Compile via rocmlir-driver ─────────────────────────────────────────────
    // For Epilogue::None: use rocmlir-gen (fast, produces tuned tile config)
    // For Bias/BiasGELU: write MLIR manually with rock.gemm + linalg.generic epilogue
    const std::string rocmlir_driver = find_tool("rocmlir-driver");
    std::string mlir_src;
    int ec = 0;

    if (epilogue == Epilogue::None) {
        const std::string rocmlir_gen = find_tool("rocmlir-gen");
        const std::string gen_cmd = fmt::format(
            "{} --arch {} --operation gemm -m {} -n {} -k {} {} -t f16 2>&1",
            rocmlir_gen, arch, M, N, K, transB ? "--transB" : "");
        mlir_src = run_cmd(gen_cmd, ec);
        if (ec != 0 || mlir_src.find("func.func") == std::string::npos) {
            fprintf(stderr, "[rocmlir_gemm] gen failed ec=%d %dx%dx%d tB=%d\n", ec, M, N, K, (int)transB);
            std::lock_guard<std::mutex> lk(g_compile_mu);
            g_compile_cache[key] = nullptr;
            return std::nullopt;
        }
    } else {
        mlir_src = make_gemm_mlir(M, N, K, transB, arch, epilogue);
    }

    const std::string mlir_path = fmt::format("/tmp/ov_gemm_{}_{}_{}_{}_ep{}.mlir",
                                               M, N, K, (int)transB, (int)epilogue);
    { FILE* f = fopen(mlir_path.c_str(), "w"); if (f) { fputs(mlir_src.c_str(), f); fclose(f); } }

    const std::string compile_cmd = fmt::format(
        "{} --arch {} --kernel-pipeline=full {} 2>&1", rocmlir_driver, arch, mlir_path);
    const std::string compiled = run_cmd(compile_cmd, ec);
    std::remove(mlir_path.c_str());

    if (ec != 0 || compiled.size() < 100) {
        fprintf(stderr, "[rocmlir_gemm] compile failed ec=%d %dx%dx%d\n", ec, M, N, K);
        std::lock_guard<std::mutex> lk(g_compile_mu);
        g_compile_cache[key] = nullptr;
        return std::nullopt;
    }

    auto hsaco = extract_hsaco(compiled);
    if (hsaco.size() < 100) {
        std::lock_guard<std::mutex> lk(g_compile_mu);
        g_compile_cache[key] = nullptr;
        return std::nullopt;
    }

    GemmKernel kernel;
    kernel.hsaco = std::move(hsaco);
    {
        const std::string km = "kernel_metadata<\"";
        auto kp = compiled.find(km);
        if (kp != std::string::npos) {
            size_t ks = kp + km.size(), ke = compiled.find('"', ks);
            if (ke != std::string::npos) kernel.func_name = compiled.substr(ks, ke - ks);
        }
        if (kernel.func_name.empty()) {
            const std::string sn = "symbol_name=";
            auto kp2 = compiled.find(sn);
            if (kp2 != std::string::npos) {
                size_t ks = kp2 + sn.size(), ke = compiled.find('"', ks);
                if (ke != std::string::npos) kernel.func_name = compiled.substr(ks, ke - ks);
            }
        }
    }
    kernel.grid_x  = parse_uint(compiled, "grid_size = ");
    kernel.block_x = parse_uint(compiled, "block_size = ");
    if (kernel.block_x == 0) kernel.block_x = 128;
    if (kernel.grid_x  == 0) kernel.grid_x  = ((size_t)M * N + kernel.block_x - 1) / kernel.block_x;

    hipError_t herr = hipModuleLoadData(&kernel.module, kernel.hsaco.data());
    if (herr != hipSuccess) {
        std::lock_guard<std::mutex> lk(g_compile_mu);
        g_compile_cache[key] = nullptr;
        return std::nullopt;
    }
    herr = hipModuleGetFunction(&kernel.func, kernel.module, kernel.func_name.c_str());
    if (herr != hipSuccess) {
        hipModuleUnload(kernel.module); kernel.module = nullptr;
        std::lock_guard<std::mutex> lk(g_compile_mu);
        g_compile_cache[key] = nullptr;
        return std::nullopt;
    }

    kernel.epilogue = epilogue;
    fprintf(stderr, "[rocmlir_gemm] compiled %dx%dx%d tB=%d ep=%d: grid=%u blk=%u HSACO=%zu B\n",
            M, N, K, (int)transB, (int)epilogue, kernel.grid_x, kernel.block_x, kernel.hsaco.size());

    // Persist to disk and store in compile cache
    disk_save(arch_key, M, N, K, transB, kernel, Backend::ROCMLIR);
    auto canonical = std::make_shared<GemmKernel>(std::move(kernel));
    {
        std::lock_guard<std::mutex> lk(g_compile_mu);
        g_compile_cache[key] = canonical;
    }

    GemmKernel copy;
    copy.hsaco     = canonical->hsaco;
    copy.func_name = canonical->func_name;
    copy.grid_x    = canonical->grid_x;
    copy.block_x   = canonical->block_x;
    copy.epilogue  = canonical->epilogue;
    hipModuleLoadData(&copy.module, copy.hsaco.data());
    hipModuleGetFunction(&copy.func, copy.module, copy.func_name.c_str());
    return copy;
}

void launch_rocmlir_gemm(const GemmKernel& kernel, hipStream_t stream,
                          void* A, void* B, void* C) {
    void* args[] = {&A, &B, &C};
    hipModuleLaunchKernel(kernel.func,
        kernel.grid_x, 1, 1, kernel.block_x, 1, 1,
        0, stream, args, nullptr);
}

void launch_rocmlir_gemm_bias(const GemmKernel& kernel, hipStream_t stream,
                               void* A, void* B, void* bias, void* C) {
    void* args[] = {&A, &B, &bias, &C};
    hipModuleLaunchKernel(kernel.func,
        kernel.grid_x, 1, 1, kernel.block_x, 1, 1,
        0, stream, args, nullptr);
}

SelectResult select_backend(
        rocblas_handle handle, hipStream_t stream,
        int M, int N, int K, bool transB,
        const void* A, const void* B, void* C,
        const std::string& arch, int /*num_cu*/) {

    const ShapeKey key{M, N, K, transB, arch};

    // ── In-process selection cache ─────────────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(g_select_mu);
        auto it = g_select_cache.find(key);
        if (it != g_select_cache.end()) return it->second;
    }

    // ── Disk cache: winner already known? ─────────────────────────────────────
    {
        Backend cached_winner;
        auto from_disk = disk_load(arch, M, N, K, transB, cached_winner);
        if (from_disk) {
            SelectResult res;
            res.backend = cached_winner;
            if (cached_winner == Backend::ROCMLIR) {
                res.kernel = std::make_shared<GemmKernel>(std::move(*from_disk));
            }
            std::lock_guard<std::mutex> lk(g_select_mu);
            g_select_cache[key] = res;
            return res;
        }
    }

    // ── Compile kernel for benchmark ──────────────────────────────────────────
    auto maybe_kernel = compile_rocmlir_gemm(M, N, K, transB, arch, 0);
    if (!maybe_kernel) {
        SelectResult res{Backend::ROCBLAS, nullptr};
        std::lock_guard<std::mutex> lk(g_select_mu);
        g_select_cache[key] = res;
        return res;
    }
    auto kernel_ptr = std::make_shared<GemmKernel>(std::move(*maybe_kernel));

    // ── Benchmark rocBLAS (5 warmup + 20 reps, ~0.3s) ────────────────────────
    const float alpha = 1.f, beta = 0.f;
    auto blas_fn = [&]() {
        rocblas_gemm_ex(handle,
            transB ? rocblas_operation_none : rocblas_operation_transpose,
            rocblas_operation_none,
            N, M, K, &alpha,
            B, rocblas_datatype_f16_r, transB ? N : K,
            A, rocblas_datatype_f16_r, K,
            &beta,
            C, rocblas_datatype_f16_r, N,
            C, rocblas_datatype_f16_r, N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0);
    };
    const double blas_us = median_us(stream, blas_fn);

    // ── Benchmark rocMLIR ─────────────────────────────────────────────────────
    void* A_nc = const_cast<void*>(A);
    void* B_nc = const_cast<void*>(B);
    auto mlir_fn = [&]() { launch_rocmlir_gemm(*kernel_ptr, stream, A_nc, B_nc, C); };
    const double mlir_us = median_us(stream, mlir_fn);

    const bool mlir_wins = mlir_us < blas_us;
    fprintf(stderr, "[rocmlir_gemm] %dx%dx%d tB=%d: rocBLAS=%.2fus rocMLIR=%.2fus → %s\n",
            M, N, K, (int)transB, blas_us, mlir_us, mlir_wins ? "rocMLIR ✓" : "rocBLAS ✓");

    // ── Persist to disk ───────────────────────────────────────────────────────
    disk_save(arch, M, N, K, transB, *kernel_ptr,
              mlir_wins ? Backend::ROCMLIR : Backend::ROCBLAS);

    SelectResult res;
    res.backend = mlir_wins ? Backend::ROCMLIR : Backend::ROCBLAS;
    res.kernel  = mlir_wins ? kernel_ptr : nullptr;

    std::lock_guard<std::mutex> lk(g_select_mu);
    g_select_cache[key] = res;
    return res;
}

}  // namespace rocmlir_gemm
}  // namespace rocm_gpu
}  // namespace ov
