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
#include <sstream>
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

// Generate the high-level fused-GEMM IR (rocMLIR "highlevel" dot/add/exp dialect)
// for GEMM + optional epilogue. Compiled via the rocMLIR
// "migraphx,highlevel,gpu,rocdl,binary" pipeline (pipeline name is fixed by the
// rocMLIR tool, unrelated to the MIGraphX framework), which fuses the whole
// dot+bias+activation chain into ONE kernel and honors an injected perf_config.
//
// X = A[M,K] row-major; W = B. transB=true → W stored [N,K] (OV FC weights),
// transposed to [K,N] for the dot. bias[N] broadcast over M rows. GELU uses the
// sigmoid approximation y = x/(1+exp(-1.702x)) — pure elementwise, fuses cleanly.
// Kernel args (match launch order): X, W, (bias); C is the RETURN value.
static std::string make_gemm_mlir(int M, int N, int K, bool transB,
                                   const std::string& arch, Epilogue epilogue,
                                   const std::string& perf_config = "", int num_cu = 64) {
    std::ostringstream s;
    const std::string archTriple = "amdgcn-amd-amdhsa:" + arch;
    const bool fused = (epilogue != Epilogue::None);
    const int wD0 = transB ? N : K, wD1 = transB ? K : N;
    const std::string dotcfg = perf_config.empty() ? "" :
        fmt::format(" {{perf_config = \"{}\"}}", perf_config);
    auto sh = [&](const std::string& dims, const std::string& st) {
        return fmt::format("!migraphx.shaped<{}xf16, {}>", dims, st);
    };

    s << "module {\n";
    s << "  func.func @rock_gemm(";
    s << "%arg0: " << sh(fmt::format("{}x{}", M, K), fmt::format("{}x1", K));
    s << ", %arg1: " << sh(fmt::format("{}x{}", wD0, wD1), fmt::format("{}x1", wD1));
    if (fused) s << ", %arg2: " << sh(std::to_string(N), "1");
    s << ") -> " << sh(fmt::format("{}x{}", M, N), fmt::format("{}x1", N));
    s << fmt::format(" attributes {{arch = \"{}\", kernel = \"mixr\", num_cu = {} : i64}} {{\n",
                     archTriple, num_cu);

    const std::string Xmk = sh(fmt::format("{}x{}", M, K), fmt::format("{}x1", K));
    const std::string Cmn = sh(fmt::format("{}x{}", M, N), fmt::format("{}x1", N));
    int v = 0;
    std::string wref = "%arg1";
    std::string Wkn;
    if (transB) {
        s << fmt::format("    %{} = migraphx.transpose {} {{permutation = [1, 0]}} : {} -> {}\n",
                         v, wref, sh(fmt::format("{}x{}", N, K), fmt::format("{}x1", K)),
                         sh(fmt::format("{}x{}", K, N), fmt::format("1x{}", K)));
        wref = fmt::format("%{}", v); ++v;
        Wkn = sh(fmt::format("{}x{}", K, N), fmt::format("1x{}", K));
    } else {
        Wkn = sh(fmt::format("{}x{}", K, N), fmt::format("{}x1", N));
    }
    s << fmt::format("    %{} = migraphx.dot %arg0, {}{} : {}, {} -> {}\n",
                     v, wref, dotcfg, Xmk, Wkn, Cmn);
    std::string cur = fmt::format("%{}", v); ++v;

    if (fused) {
        s << fmt::format("    %{} = migraphx.multibroadcast %arg2 {{out_lens = [{}, {}]}} : {} -> {}\n",
                         v, M, N, sh(std::to_string(N), "1"), sh(fmt::format("{}x{}", M, N), "0x1"));
        int bb = v; ++v;
        s << fmt::format("    %{} = migraphx.add {}, %{} : {}, {} -> {}\n",
                         v, cur, bb, Cmn, sh(fmt::format("{}x{}", M, N), "0x1"), Cmn);
        cur = fmt::format("%{}", v); ++v;
        if (epilogue == Epilogue::BiasGELU) {
            auto bconst = [&](double val) {
                s << fmt::format("    %{} = migraphx.literal(dense<{:.6e}> : tensor<1xf16>) : <1xf16, 0>\n", v, val);
                int lit = v; ++v;
                s << fmt::format("    %{} = migraphx.multibroadcast %{} {{out_lens = [{}, {}]}} : <1xf16, 0> -> {}\n",
                                 v, lit, M, N, sh(fmt::format("{}x{}", M, N), "0x0"));
                int b = v; ++v; return b;
            };
            int c = bconst(-1.702);
            s << fmt::format("    %{} = migraphx.mul {}, %{} : {}, {} -> {}\n",
                             v, cur, c, Cmn, sh(fmt::format("{}x{}", M, N), "0x0"), Cmn);
            int mx = v; ++v;
            s << fmt::format("    %{} = migraphx.exp %{} : {} -> {}\n", v, mx, Cmn, Cmn);
            int ex = v; ++v;
            int c1 = bconst(1.0);
            s << fmt::format("    %{} = migraphx.add %{}, %{} : {}, {} -> {}\n",
                             v, ex, c1, Cmn, sh(fmt::format("{}x{}", M, N), "0x0"), Cmn);
            int den = v; ++v;
            s << fmt::format("    %{} = migraphx.div {}, %{} : {}, {} -> {}\n", v, cur, den, Cmn, Cmn, Cmn);
            cur = fmt::format("%{}", v); ++v;
        }
    }
    s << fmt::format("    return {} : {}\n", cur, Cmn);
    s << "  }\n}\n";
    return s.str();
}

// ── Public API ────────────────────────────────────────────────────────────────

std::optional<GemmKernel> compile_rocmlir_gemm(
        int M, int N, int K, bool transB,
        const std::string& arch, int num_cu,
        Epilogue epilogue, const std::string& perf_config) {

    std::string arch_key = arch + "_ep" + std::to_string((int)epilogue);
    if (!perf_config.empty()) {
        std::string pc = perf_config;
        for (char& c : pc) if (c == ':' || c == ',') c = '_';
        arch_key += "_pc" + pc;
    }
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
    // High-level fused IR through the rocMLIR "migraphx,highlevel,gpu,rocdl,binary"
    // pipeline (name fixed by the tool; unrelated to the MIGraphX framework). Fuses
    // the whole chain into one kernel and honors the injected perf_config tile.
    const std::string rocmlir_driver = find_tool("rocmlir-driver");
    int ec = 0;
    const std::string mlir_src = make_gemm_mlir(M, N, K, transB, arch, epilogue, perf_config, num_cu);

    const std::string mlir_path = fmt::format("/tmp/ov_gemm_{}_{}_{}_{}_ep{}.mlir",
                                               M, N, K, (int)transB, (int)epilogue);
    { FILE* f = fopen(mlir_path.c_str(), "w"); if (f) { fputs(mlir_src.c_str(), f); fclose(f); } }

    const std::string compile_cmd = fmt::format(
        "{} --arch {} --kernel-pipeline=migraphx,highlevel,gpu,rocdl,binary {} 2>&1",
        rocmlir_driver, arch, mlir_path);
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

// ── GEMM perf_config tuning (persistent JSON, reuses conv cache file) ───────────
namespace {
std::string gemm_tune_cache_path(const std::string& arch) {
    const char* env = std::getenv("ROCMLIR_TUNING_CACHE");
    if (env && *env) return env;
    const char* home = std::getenv("HOME");
    std::string base = home ? std::string(home) : "/tmp";
    return base + "/.cache/ov_rocmlir_tuning_" + arch.substr(0, arch.find(':')) + ".json";
}
size_t gemm_tune_key(int M,int N,int K,bool tB,int ep){
    size_t h=std::hash<std::string>{}("gemm");
    auto mix=[&](long v){h^=std::hash<long>{}(v)+0x9e3779b9+(h<<6)+(h>>2);};
    mix(M);mix(N);mix(K);mix(tB?1:0);mix(ep);return h;
}
std::unordered_map<size_t,std::string>& gemm_tune_map(const std::string& arch){
    static std::mutex mu; static std::unordered_map<size_t,std::string> m; static bool loaded=false;
    std::lock_guard<std::mutex> lk(mu);
    if(!loaded){loaded=true; std::ifstream f(gemm_tune_cache_path(arch)); std::string ln;
        while(std::getline(f,ln)){auto q1=ln.find('"');if(q1==std::string::npos)continue;
            auto q2=ln.find('"',q1+1);auto q3=ln.find('"',q2+1);auto q4=ln.find('"',q3+1);
            if(q2==std::string::npos||q3==std::string::npos||q4==std::string::npos)continue;
            try{m[std::stoull(ln.substr(q1+1,q2-q1-1),nullptr,16)]=ln.substr(q3+1,q4-q3-1);}catch(...){}}}
    return m;
}
void gemm_tune_save(const std::string& arch){
    auto& m=gemm_tune_map(arch); std::string path=gemm_tune_cache_path(arch);
    auto sl=path.rfind('/'); if(sl!=std::string::npos)::mkdir(path.substr(0,sl).c_str(),0755);
    std::ofstream f(path,std::ios::trunc); if(!f)return;
    f<<"{\n"; bool first=true;
    for(auto&[k,vv]:m){if(!first)f<<",\n";char hx[32];snprintf(hx,sizeof(hx),"%016zx",k);
        f<<"  \""<<hx<<"\": \""<<vv<<"\"";first=false;}
    f<<"\n}\n";
}
double time_gemm_cfg(int M,int N,int K,bool tB,const std::string& arch,int num_cu,Epilogue ep,const std::string& cfg){
    auto k=compile_rocmlir_gemm(M,N,K,tB,arch,num_cu,ep,cfg);
    if(!k||!k->func)return 1e9;
    bool fused=(ep!=Epilogue::None);
    void *A=nullptr,*W=nullptr,*Bs=nullptr,*C=nullptr;
    if(hipMalloc(&A,(size_t)M*K*2))return 1e9;
    if(hipMalloc(&W,(size_t)N*K*2)){hipFree(A);return 1e9;}
    if(hipMalloc(&C,(size_t)M*N*2)){hipFree(A);hipFree(W);return 1e9;}
    if(fused&&hipMalloc(&Bs,(size_t)N*2)){hipFree(A);hipFree(W);hipFree(C);return 1e9;}
    hipMemset(A,0,(size_t)M*K*2);hipMemset(W,0,(size_t)N*K*2);if(fused)hipMemset(Bs,0,(size_t)N*2);
    auto run=[&]{if(fused)launch_rocmlir_gemm_bias(*k,nullptr,A,W,Bs,C);else launch_rocmlir_gemm(*k,nullptr,A,W,C);};
    double us=1e9; for(int i=0;i<5;i++)run();
    if(hipDeviceSynchronize()==hipSuccess)us=median_us(nullptr,run,3,15);
    hipFree(A);hipFree(W);hipFree(C);if(fused)hipFree(Bs);
    return us;
}
} // namespace

std::string get_tuned_gemm_config(int M,int N,int K,bool transB,
                                  const std::string& arch,int num_cu,Epilogue epilogue){
    const char* env_cfg=std::getenv("ROCMLIR_PERF_CONFIG");
    if(env_cfg&&*env_cfg)return env_cfg;
    const size_t key=gemm_tune_key(M,N,K,transB,(int)epilogue);
    auto& m=gemm_tune_map(arch);
    { static std::mutex lk; std::lock_guard<std::mutex> g(lk);
      auto it=m.find(key); if(it!=m.end())return it->second; }
    const char* en=std::getenv("ROCMLIR_ENABLE_TUNING");
    if(!en||std::string(en)!="1")return "";
    const std::string gen=find_tool("rocmlir-gen"); int ec=0;
    std::string space=run_cmd(fmt::format(
        "{} --arch {} --operation gemm -m {} -n {} -k {} {} -t f16 --emit-tuning-space=quick 2>/dev/null",
        gen,arch,M,N,K,transB?"--transB":""),ec);
    std::vector<std::string> cands;
    { std::istringstream is(space); std::string ln;
      while(std::getline(is,ln))if(!ln.empty()&&ln[0]=='v')cands.push_back(ln); }
    double best=time_gemm_cfg(M,N,K,transB,arch,num_cu,epilogue,"");
    std::string best_cfg="";
    for(auto&c:cands){double t=time_gemm_cfg(M,N,K,transB,arch,num_cu,epilogue,c);if(t<best){best=t;best_cfg=c;}}
    fprintf(stderr,"[gemm-tune] %dx%dx%d tB=%d ep=%d best=%.2fus cfg=%s (%zu cands)\n",
            M,N,K,(int)transB,(int)epilogue,best,best_cfg.empty()?"<default>":best_cfg.c_str(),cands.size());
    { static std::mutex lk; std::lock_guard<std::mutex> g(lk); m[key]=best_cfg; gemm_tune_save(arch); }
    // Mark "tuned" even if heuristic won, so normal runs know this shape was tuned:
    // store a sentinel when best_cfg empty so the entry exists (caller treats non-missing as tuned).
    return best_cfg;
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
