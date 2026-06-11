// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RocmAttentionMatMulOp: compiles attention GEMM kernels via rocmlir-gen.
// The factory compiles the kernel on first use; Execute() dispatches to it.

#include "rocm_attention_matmul.hpp"
#include <rocm_operation_registry.hpp>
#include <error.hpp>
#include <fmt/format.h>
#include <cstdlib>
#include <fstream>
#include <atomic>
#include <unistd.h>
#include <cmath>
#include "kernels/bias_add.hpp"

static std::string attn_run_cmd(const std::string& cmd, int& exit_code) {
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (!pipe) { exit_code = -1; return ""; }
    std::string out;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    exit_code = ::pclose(pipe);
    return out;
}

namespace ov {
namespace rocm_gpu {

bool RocmAttentionMatMulOp::isEnabled() {
    const char* env = std::getenv("ROCM_FUSE_ATTENTION");
    return !(env && std::string(env) == "0");
}

// ── Global kernel cache ───────────────────────────────────────────────────────
struct AttnKernelEntry {
    std::vector<char> hsaco;
    std::string kernel_name;
    int grid_x = 1, block_x = 256;
    hipModule_t   module   = nullptr;
    hipFunction_t function = nullptr;
};

static std::mutex g_mu;
static std::unordered_map<std::string, AttnKernelEntry> g_cache;

// ── Generate pe(V) conv-only MLIR ─────────────────────────────────────────────
// Computes ONLY: depthwise_conv3x3(V) → pe_work (workspace)
// Takes V directly (already sliced by VariadicSplit, flat [K*H*W] layout).
// The bias add and attn_out add are done in Execute via launch_pe_add_fp16.
//
// Kernel signature: 3 args (v_flat, filter_flat, pe_work_output_flat)
// This matches MIGraphX's @mlir_reshape_slice_reshape_convolution_broadcast_reshape_add_add
// but simplified: V is pre-sliced, so no Slice transform needed.
static std::string generate_pe_conv_mlir(
        int nh, int dq, int dv, int H, int W,
        const std::string& perf_config, const std::string& arch_triple, int num_cu) {
    const int N           = H * W;
    const int K           = nh * dv;
    const int v_flat      = K * N;          // input V flat: [nh*dv*H*W]
    const int filter_flat = K * 9;          // filter flat: [K*1*3*3]
    const int output_flat = K * N;          // output flat: [K*H*W]

    std::ostringstream ir;

    // 3-arg kernel: (v_flat, filter_flat, pe_work_output)
    // V is already VariadicSplit output — flat contiguous layout [K, H*W] (or [nh, dv, H, W])
    ir << "module {\n"
       << "  func.func @pe_conv_v2("
       << "%arg0: memref<" << v_flat << "xf16>, "
       << "%arg1: memref<" << filter_flat << "xf16>, "
       << "%arg2: memref<" << output_flat << "xf16>)\n"
       << "    attributes {arch = \"" << arch_triple << "\", kernel = \"mixr\", num_cu = " << num_cu << " : i64} {\n";

    // Filter: flat [K*1*3*3] → [K, 1, 3, 3]
    ir << "    %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> ((d0 * 3 + d2) * 3 + d3)>"
       << " by [<Unmerge{" << K << ", 3, 3} [\"exp0\", \"exp2\", \"exp3\"] at [0, 2, 3] -> [\"dim0\"] at [0]>,"
       << " <AddDim{1} [\"unit1\"] at [1] -> [] at []>]"
       << " bounds = [" << K << ", 1, 3, 3] -> [" << filter_flat << "]>"
       << " : memref<" << filter_flat << "xf16> to memref<" << K << "x1x3x3xf16>\n";

    // V: flat [K*H*W] → [1, K, H, W] (V from VariadicSplit is contiguous K×H×W)
    // V layout from AV GEMM: A is [nh, dv, N] = [nh, dv, H*W], flattened = [K, H*W]
    // rock.transform: flat index i → (k, h, w) where k=i/(H*W), h=(i/W)%H, w=i%W
    ir << "    %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> ((d1 * " << H << " + d2) * " << W << " + d3)>"
       << " by [<Unmerge{" << K << ", " << H << ", " << W << "} [\"exp1\", \"exp2\", \"exp3\"] at [1, 2, 3] -> [\"dim0\"] at [0]>,"
       << " <AddDim{1} [\"unit0\"] at [0] -> [] at []>]"
       << " bounds = [1, " << K << ", " << H << ", " << W << "] -> [" << v_flat << "]>"
       << " : memref<" << v_flat << "xf16> to memref<1x" << K << "x" << H << "x" << W << "xf16>\n";

    // V [1,K,H,W] → [1,K,1,H,W] for depthwise conv (add unit c-dim per group)
    ir << "    %5 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d2, d3, d4)>"
       << " by [<PassThrough [\"n\", \"h\", \"w\"] at [0, 3, 4] -> [\"n\", \"h\", \"w\"] at [0, 2, 3]>,"
       << " <Unmerge{" << K << ", 1} [\"g\", \"c\"] at [1, 2] -> [\"c\"] at [1]>]"
       << " bounds = [1, " << K << ", 1, " << H << ", " << W << "] -> [1, " << K << ", " << H << ", " << W << "]>"
       << " : memref<1x" << K << "x" << H << "x" << W << "xf16> to memref<1x" << K << "x1x" << H << "x" << W << "xf16>\n";

    // Filter: [K,1,3,3] → [K,1,1,3,3] (add group dim)
    ir << "    %6 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 + d1, d2, d3, d4)>"
       << " by [<PassThrough [\"c\", \"y\", \"x\"] at [2, 3, 4] -> [\"c\", \"y\", \"x\"] at [1, 2, 3]>,"
       << " <Unmerge{" << K << ", 1} [\"g\", \"k\"] at [0, 1] -> [\"k\"] at [0]>]"
       << " bounds = [" << K << ", 1, 1, 3, 3] -> [" << K << ", 1, 3, 3]>"
       << " : memref<" << K << "x1x3x3xf16> to memref<" << K << "x1x1x3x3xf16>\n";

    // Internal alloc for conv output (pe fusion disabled; kept for future use).
    // Direct write to arg2 via transforms causes GPU page faults on gfx1201 due to
    // invalid AGPR count in compiled kernel. %alloc + memref.copy is used instead.
    ir << "    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x" << K << "x" << H << "x" << W << "xf16>\n";
    ir << "    %7 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d2, d3, d4)>"
       << " by [<PassThrough [\"n\", \"h\", \"w\"] at [0, 3, 4] -> [\"n\", \"h\", \"w\"] at [0, 2, 3]>,"
       << " <Unmerge{" << K << ", 1} [\"g\", \"k\"] at [1, 2] -> [\"k\"] at [1]>]"
       << " bounds = [1, " << K << ", 1, " << H << ", " << W << "] -> [1, " << K << ", " << H << ", " << W << "]>"
       << " : memref<1x" << K << "x" << H << "x" << W << "xf16> to memref<1x" << K << "x1x" << H << "x" << W << "xf16>\n";

    // Permute filter [K,1,1,3,3] → [K,3,1,1,3] (g,y,k,c,x layout)
    ir << "    %8 = rock.transform %6 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1, d4)>"
       << " by [<PassThrough [\"dim0\", \"dim3\", \"dim1\", \"dim2\", \"dim4\"] at [0, 1, 2, 3, 4]"
       << " -> [\"dim0\", \"dim3\", \"dim1\", \"dim2\", \"dim4\"] at [0, 3, 1, 2, 4]>]"
       << " bounds = [" << K << ", 3, 1, 1, 3] -> [" << K << ", 1, 1, 3, 3]>"
       << " : memref<" << K << "x1x1x3x3xf16> to memref<" << K << "x3x1x1x3xf16>\n";

    // Permute input [1,K,1,H,W] → [K,H,1,1,W] (gi,hi,ni,ci,wi layout)
    ir << "    %9 = rock.transform %5 by <affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d1, d4)>"
       << " by [<PassThrough [\"dim1\", \"dim3\", \"dim0\", \"dim2\", \"dim4\"] at [0, 1, 2, 3, 4]"
       << " -> [\"dim1\", \"dim3\", \"dim0\", \"dim2\", \"dim4\"] at [1, 3, 0, 2, 4]>]"
       << " bounds = [" << K << ", " << H << ", 1, 1, " << W << "] -> [1, " << K << ", 1, " << H << ", " << W << "]>"
       << " : memref<1x" << K << "x1x" << H << "x" << W << "xf16> to memref<" << K << "x" << H << "x1x1x" << W << "xf16>\n";

    // Depthwise 3×3 conv → %alloc, then copy to arg2
    ir << "    rock.conv(%8, %9, %7) {dilations = [1 : index, 1 : index],"
       << " filter_layout = [\"g\", \"y\", \"k\", \"c\", \"x\"],"
       << " input_layout = [\"gi\", \"hi\", \"ni\", \"ci\", \"wi\"],"
       << " output_layout = [\"no\", \"go\", \"ko\", \"ho\", \"wo\"],"
       << " padding = [1 : index, 1 : index, 1 : index, 1 : index],"
       << " perf_config = \"" << perf_config << "\","
       << " strides = [1 : index, 1 : index]}"
       << " : memref<" << K << "x3x1x1x3xf16>, memref<" << K << "x" << H << "x1x1x" << W << "xf16>,"
       << " memref<1x" << K << "x1x" << H << "x" << W << "xf16>\n";

    ir << "    %alloc_flat = rock.transform %alloc by <affine_map<(d0) -> (0, d0 floordiv " << (H*W) << ", "
       << "(d0 mod " << (H*W) << ") floordiv " << W << ", d0 mod " << W << ")>"
       << " by [<Merge{1, " << K << ", " << H << ", " << W << "} [\"dim0\"] at [0] -> [\"col0\", \"col1\", \"col2\", \"col3\"] at [0, 1, 2, 3]>]"
       << " bounds = [" << output_flat << "] -> [1, " << K << ", " << H << ", " << W << "]>"
       << " : memref<1x" << K << "x" << H << "x" << W << "xf16> to memref<" << output_flat << "xf16>\n";
    ir << "    memref.copy %alloc_flat, %arg2 : memref<" << output_flat << "xf16> to memref<" << output_flat << "xf16>\n";

    ir << "    return\n  }\n}\n";
    return ir.str();
}

// ── Compile Attention GEMM via rocmlir-gen + rocmlir-driver ──────────────────
// Uses rocmlir-gen to generate a batched GEMM MLIR directly (no external deps).
// QKT kernel: (Q^T × K) → [nh, N, N], params: M=N, N=N, K=dq, transA=true
// AV  kernel: (V  × A ) → [nh, dv, N], params: M=dv, N=N, K=N
static AttnKernelEntry compile_kernel(
        const std::string& sym,
        int nh, int dq, int dv, int H, int W,
        const std::string& arch) {

    static std::atomic<int> seq{0};
    const std::string base = "/tmp/ov_attn_" + std::to_string(getpid())
                           + "_" + std::to_string(seq.fetch_add(1));

    const int N = H * W;

    // Find rocmlir-driver: env var → common install paths → PATH
    auto find_driver = []() -> std::string {
        if (auto e = std::getenv("ROCMLIR_DRIVER")) return e;
        for (const char* p : {"/root/rocmlir_install/bin/rocmlir-driver",
                               "/home/rocmlir_install/bin/rocmlir-driver",
                               "/home/openvino/rocmlir_install/bin/rocmlir-driver",
                               "/home/openvino/rocmlir-driver",
                               "/opt/rocmlir/bin/rocmlir-driver",
                               "/opt/rocm/bin/rocmlir-driver"}) {
            if (::access(p, X_OK) == 0) return p;
        }
        return "rocmlir-driver";
    };
    const std::string driver = find_driver();

    // Derive rocmlir-gen from driver path
    std::string gen_path = driver;
    {
        auto slash = driver.rfind('/');
        gen_path = (slash != std::string::npos)
            ? driver.substr(0, slash + 1) + "rocmlir-gen"
            : "rocmlir-gen";
    }

    // QKT: Q^T [nh, dq, N] × K [nh, dq, N] → [nh, N, N], transA
    // AV:  V   [nh, dv, N] × A [nh, N,  N] → [nh, dv, N], no transpose
    const bool is_qkt = (sym.find("transpose") != std::string::npos
                      || sym.find("qkt") != std::string::npos
                      || sym == "attn_qkt");

    int M_dim = is_qkt ? N  : dv;
    int N_dim = N;
    int K_dim = is_qkt ? dq : N;

    // perf_config: let rocmlir-gen auto-select for this arch.
    // For gfx1201 (RDNA4, 32 CUs, wavefront=32 WMMA), auto-selection gives:
    //   QKT: grid~52 block=128 (optimal for 32-CU utilization)
    //   AV:  grid~52 block=64
    const std::string perf_cfg = "";

    // Pass ROCMLIR_ENABLE_TUNING to rocmlir-gen if set (enables exhaustive search)
    const char* attn_tune = std::getenv("ROCMLIR_ENABLE_TUNING");
    const std::string attn_tune_prefix = (attn_tune && std::string(attn_tune) == "1")
        ? "ROCMLIR_ENABLE_TUNING=1 " : "";
    const std::string gen_cmd =
        attn_tune_prefix + gen_path
        + " -t f16 --operation gemm"
        + " -m " + std::to_string(M_dim)
        + " -n " + std::to_string(N_dim)
        + " -k " + std::to_string(K_dim)
        + " --batchsize " + std::to_string(nh)
        + (is_qkt ? " --transA" : "")
        + " --arch " + arch
        + (perf_cfg.empty() ? "" : " --perf_config \"" + perf_cfg + "\"")
        + " 2>&1";

    int ec = 0;
    const std::string gen_ir = attn_run_cmd(gen_cmd, ec);
    if (ec != 0 || gen_ir.size() < 10)
        OPENVINO_THROW("RocmAttention: rocmlir-gen failed (ec=", ec,
                       "): ", gen_ir.substr(0, 200));

    const std::string ir_file = base + ".mlir";
    { std::ofstream f(ir_file); f << gen_ir; }
    const std::string drv_cmd = driver + " --arch " + arch + " --kernel-pipeline=full " + ir_file;
    const std::string drv_out = attn_run_cmd(drv_cmd, ec);
    std::remove(ir_file.c_str());

    AttnKernelEntry k;
    // Extract kernel metadata - rocmlir-driver output format:
    //   #gpu.kernel_metadata<"name", !llvm.func<...>, ..., metadata = {block_size = N : i32, grid_size = M : i32, ...}>
    // followed by:
    //   gpu.binary @module [#gpu.object<..., bin = "\7FELF...">]
    auto km_pos = drv_out.find("kernel_metadata<\"");
    auto bn_pos = drv_out.find("bin = \"");
    if (km_pos == std::string::npos || bn_pos == std::string::npos)
        OPENVINO_THROW("RocmAttention: no kernel in rocmlir-driver output");

    size_t kns = km_pos + 17, kne = drv_out.find('"', kns);
    k.kernel_name = drv_out.substr(kns, kne - kns);

    // grid_size and block_size are inside the metadata = {...} block
    auto gs_pos = drv_out.find("grid_size = ", km_pos);
    auto bs_pos = drv_out.find("block_size = ", km_pos);
    if (gs_pos != std::string::npos) k.grid_x  = std::stoi(drv_out.substr(gs_pos + 12, 20));
    if (bs_pos != std::string::npos) k.block_x = std::stoi(drv_out.substr(bs_pos + 13, 20));

    // Extract binary
    size_t i = bn_pos + 7;
    while (i < drv_out.size()) {
        char c = drv_out[i];
        if (c == '"') break;
        if (c == '\\' && i+1 < drv_out.size()) {
            char nx = drv_out[i+1];
            if (nx == 'n') { k.hsaco.push_back('\n'); i+=2; continue; }
            if (nx == '\\') { k.hsaco.push_back('\\'); i+=2; continue; }
            if (i+2 < drv_out.size() && std::isxdigit(drv_out[i+1]) && std::isxdigit(drv_out[i+2])) {
                char h[3]={drv_out[i+1],drv_out[i+2],0};
                k.hsaco.push_back((char)std::strtol(h,nullptr,16)); i+=3; continue;
            }
        }
        k.hsaco.push_back(c); i++;
    }
    if (k.hsaco.size() < 4 || (unsigned char)k.hsaco[0] != 0x7F)
        OPENVINO_THROW("RocmAttention: invalid ELF (size=", k.hsaco.size(), ")");

    auto err = hipModuleLoadData(&k.module, k.hsaco.data());
    if (err != hipSuccess)
        OPENVINO_THROW("RocmAttention: hipModuleLoadData: ", hipGetErrorString(err));
    err = hipModuleGetFunction(&k.function, k.module, k.kernel_name.c_str());
    if (err != hipSuccess)
        OPENVINO_THROW("RocmAttention: hipModuleGetFunction(", k.kernel_name, "): ", hipGetErrorString(err));

    std::cerr << "[AttnKernel] " << sym << " grid=" << k.grid_x << " block=" << k.block_x << "\n";
    return k;
}

// ── Compile pe(V) depthwise conv kernel via rocmlir-gen ───────────────────────
// Uses rocmlir-gen to auto-generate a depthwise 3×3 grouped conv kernel.
// This avoids hand-written MLIR with architecture-specific perf_config issues
// (e.g., v3 WMMA perf_config causes AGPR metadata bugs on gfx1201).
// rocmlir-gen auto-selects the correct perf_config and avoids GPU heap malloc.
//
// Kernel signature (rocmlir-gen grouped conv):
//   arg0: filter [K*1*3*3 = 3456 x f16]  (filter first!)
//   arg1: input V [K*H*W = 153600 x f16]
//   arg2: output pe_work [K*H*W = 153600 x f16]
//
// Execute(): pass (filter, v_flat, pe_work) in this order.
static AttnKernelEntry compile_pe_kernel(int nh, int dq, int dv, int H, int W,
                                          const std::string& arch, const std::string& driver,
                                          int num_cu) {
    static std::atomic<int> seq{0};
    const std::string base = "/tmp/ov_pe_" + std::to_string(getpid())
                           + "_" + std::to_string(seq.fetch_add(1));

    const int K_ch = nh * dv;  // total depthwise channels (K=384)

    // Find rocmlir-gen from driver path
    std::string gen_path;
    {
        auto slash = driver.rfind('/');
        gen_path = (slash != std::string::npos)
            ? driver.substr(0, slash + 1) + "rocmlir-gen"
            : "rocmlir-gen";
    }

    // Generate depthwise conv MLIR via rocmlir-gen (groups=K = depthwise).
    // ROCMLIR_ENABLE_TUNING=1 enables exhaustive perf_config search for this shape.
    const char* tune_env = std::getenv("ROCMLIR_ENABLE_TUNING");
    const std::string tune_prefix = (tune_env && std::string(tune_env) == "1")
        ? "ROCMLIR_ENABLE_TUNING=1 " : "";
    const std::string gen_cmd =
        tune_prefix + gen_path
        + " -ph -t f16"
        + " -batchsize 1"
        + " -in_channels " + std::to_string(K_ch)
        + " -out_channels " + std::to_string(K_ch)
        + " -g " + std::to_string(K_ch)    // groups=K → depthwise conv
        + " -fil_h 3 -fil_w 3"
        + " -in_h " + std::to_string(H)
        + " -in_w " + std::to_string(W)
        + " --padding_h 1 --padding_w 1"
        + " --arch " + arch
        + " 2>&1";

    int ec = 0;
    const std::string gen_ir = attn_run_cmd(gen_cmd, ec);
    if (ec != 0 || gen_ir.size() < 10)
        OPENVINO_THROW("RocmAttn pe: rocmlir-gen failed (ec=", ec, "): ", gen_ir.substr(0, 200));

    // Compile MLIR → HSACO
    const std::string ir_file = base + ".mlir";
    { std::ofstream f(ir_file); f << gen_ir; }
    const std::string drv_cmd = driver + " --arch " + arch + " --kernel-pipeline=full " + ir_file;
    const std::string drv_out = attn_run_cmd(drv_cmd, ec);
    std::remove(ir_file.c_str());

    if (ec != 0)
        OPENVINO_THROW("RocmAttention pe: rocmlir-driver failed (ec=", ec, ")");

    AttnKernelEntry k;
    auto km_pos = drv_out.find("kernel_metadata<\"");
    auto bn_pos = drv_out.find("bin = \"");
    if (km_pos == std::string::npos || bn_pos == std::string::npos)
        OPENVINO_THROW("RocmAttention pe: no kernel in driver output");

    size_t kns = km_pos + 17, kne = drv_out.find('"', kns);
    k.kernel_name = drv_out.substr(kns, kne - kns);
    auto gs_pos = drv_out.find("grid_size = ", km_pos);
    auto bs_pos = drv_out.find("block_size = ", km_pos);
    if (gs_pos != std::string::npos) k.grid_x  = std::stoi(drv_out.substr(gs_pos + 12, 20));
    if (bs_pos != std::string::npos) k.block_x = std::stoi(drv_out.substr(bs_pos + 13, 20));

    // Extract binary
    size_t i = bn_pos + 7;
    while (i < drv_out.size()) {
        char c = drv_out[i];
        if (c == '"') break;
        if (c == '\\' && i+1 < drv_out.size()) {
            char nx = drv_out[i+1];
            if (nx == 'n') { k.hsaco.push_back('\n'); i+=2; continue; }
            if (nx == '\\') { k.hsaco.push_back('\\'); i+=2; continue; }
            if (i+2 < drv_out.size() && std::isxdigit(drv_out[i+1]) && std::isxdigit(drv_out[i+2])) {
                char h[3]={drv_out[i+1],drv_out[i+2],0};
                k.hsaco.push_back((char)std::strtol(h,nullptr,16)); i+=3; continue;
            }
        }
        k.hsaco.push_back(c); i++;
    }
    if (k.hsaco.size() < 4 || (unsigned char)k.hsaco[0] != 0x7F)
        OPENVINO_THROW("RocmAttention pe: invalid ELF");

    auto err = hipModuleLoadData(&k.module, k.hsaco.data());
    if (err != hipSuccess)
        OPENVINO_THROW("RocmAttention pe: hipModuleLoadData: ", hipGetErrorString(err));
    err = hipModuleGetFunction(&k.function, k.module, k.kernel_name.c_str());
    if (err != hipSuccess)
        OPENVINO_THROW("RocmAttention pe: hipModuleGetFunction: ", hipGetErrorString(err));

    const int K_log = nh * dv;
    std::cerr << "[AttnKernel] pe_conv nh=" << nh << " dv=" << dv << " K=" << K_log
              << " H=" << H << " W=" << W
              << " grid=" << k.grid_x << " block=" << k.block_x << "\n";
    return k;
}

// ── Constructor ───────────────────────────────────────────────────────────────
RocmAttentionMatMulOp::RocmAttentionMatMulOp(
        const CreationContext& ctx,
        const ov::Node& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(ctx, node, std::move(inputIds), std::move(outputIds))
{
    const auto& rt = node.get_rt_info();
    auto get_rt = [&](const std::string& k) -> std::string {
        auto it = rt.find(k);
        return it != rt.end() ? it->second.as<std::string>() : "";
    };

    kind_          = get_rt("rocm_attn_kind");
    const int nh   = std::stoi(get_rt("rocm_attn_nheads"));
    const int dq   = std::stoi(get_rt("rocm_attn_dim_q"));
    const int dv   = std::stoi(get_rt("rocm_attn_dim_v"));
    const int H    = std::stoi(get_rt("rocm_attn_H"));
    const int W    = std::stoi(get_rt("rocm_attn_W"));

    OPENVINO_ASSERT(!kind_.empty(), "RocmAttentionMatMulOp: missing rt_info rocm_attn_kind");

    const std::string sym = (kind_ == "qkt") ? "attn_qkt" : "attn_av";

    // Get GPU arch
    std::string arch = ctx.device().props().gcnArchName;
    auto c = arch.find(':');
    if (c != std::string::npos) arch = arch.substr(0, c);

    // Find rocmlir-driver using same search as compile_kernel
    auto find_driver2 = []() -> std::string {
        if (auto e = std::getenv("ROCMLIR_DRIVER")) return e;
        for (const char* p : {"/root/rocmlir_install/bin/rocmlir-driver",
                               "/home/rocmlir_install/bin/rocmlir-driver",
                               "/home/openvino/rocmlir_install/bin/rocmlir-driver",
                               "/home/openvino/rocmlir-driver",
                               "/opt/rocmlir/bin/rocmlir-driver",
                               "/opt/rocm/bin/rocmlir-driver"}) {
            if (::access(p, X_OK) == 0) return p;
        }
        return "rocmlir-driver";
    };
    const std::string driver = find_driver2();
    std::string gen_path = driver;
    {
        auto slash = driver.rfind('/');
        gen_path = (slash != std::string::npos)
            ? driver.substr(0, slash + 1) + "rocmlir-gen"
            : "rocmlir-gen";
    }

    const std::string cache_key = sym + "_" + std::to_string(nh) + "_"
        + std::to_string(dq) + "_" + std::to_string(dv)
        + "_" + std::to_string(H) + "x" + std::to_string(W) + "_" + arch;

    {
        std::lock_guard<std::mutex> lk(g_mu);
        auto it = g_cache.find(cache_key);
        if (it == g_cache.end())
            it = g_cache.emplace(cache_key, compile_kernel(sym, nh, dq, dv, H, W, arch)).first;

        hip_module_   = it->second.module;
        hip_func_     = it->second.function;
        grid_x_       = it->second.grid_x;
        block_x_      = it->second.block_x;
    }

    // Compile pe(V)+AttnOutput fused kernel when pe is enabled (default ON).
    // Set ROCM_FUSE_PE=0 to disable.
    // Compile pe(V) depthwise conv kernel via rocmlir-gen.
    // rocmlir-gen handles V's non-contiguous layout (VariadicSplitAlias stride).
    const char* fuse_pe_env = std::getenv("ROCM_FUSE_PE");
    const bool pe_enabled = !fuse_pe_env || std::string(fuse_pe_env) != "0";
    if (pe_enabled && kind_ == "av" && rt.count("rocm_attn_pe_add")) {
        const int K_pe = nh * dv;
        const int num_cu = ctx.device().props().multiProcessorCount;
        const std::string pe_key = "pe_gen_nh" + std::to_string(nh)
            + "_dq" + std::to_string(dq) + "_dv" + std::to_string(dv)
            + "_H" + std::to_string(H) + "W" + std::to_string(W) + "_" + arch;
        try {
            std::lock_guard<std::mutex> lk(g_mu);
            auto pe_it = g_cache.find(pe_key);
            if (pe_it == g_cache.end())
                pe_it = g_cache.emplace(pe_key,
                    compile_pe_kernel(nh, dq, dv, H, W, arch, driver, num_cu)).first;

            pe_module_   = pe_it->second.module;
            pe_func_     = pe_it->second.function;
            pe_grid_x_   = pe_it->second.grid_x;
            pe_block_x_  = pe_it->second.block_x;
            has_pe_conv_ = true;
            pe_out_elems_       = K_pe * H * W;
            pe_workspace_bytes_ = (size_t)pe_out_elems_ * 2;  // pe_work: K*H*W FP16
            H_pe_ = H;
            W_pe_ = W;
            std::cerr << "[AttnKernel] pe_conv nh=" << nh << " dv=" << dv << " K=" << K_pe
                      << " H=" << H << " W=" << W
                      << " grid=" << pe_it->second.grid_x << " block=" << pe_it->second.block_x << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[AttnKernel] pe_conv compile failed: " << e.what() << "\n";
            has_pe_conv_ = false;
        }
    }
}

// ── Execute ───────────────────────────────────────────────────────────────────
void RocmAttentionMatMulOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers& wbs) const {
    // outputs[0] = AV GEMM result; outputs[1] = FGC output (pe+attn result, when pe enabled)
    OPENVINO_ASSERT(!inputs.empty() && outputs.size() >= 1);
    auto& stream = ctx.getThreadContext().stream();
    void* out = outputs[0].get();

    // Dispatch batched GEMM kernel compiled by rocmlir-gen.
    // Input layout:
    //   inputs[0]   = A tensor (Q for QKT, V for AV)
    //   inputs[1]   = B tensor (K×scale for QKT, softmax_attn for AV)
    //   inputs[2]   = full QKV tensor (injected by registerAttentionExtraInputs)
    //   inputs[3,4] = pe filter, pe bias (injected if has_pe_conv_, AV only)
    OPENVINO_ASSERT(inputs.size() >= 2, "RocmAttentionMatMulOp: need at least 2 inputs");
    void* A = const_cast<void*>(inputs[0].get());
    void* B = const_cast<void*>(inputs[1].get());

    {
        void* args[] = { &A, &B, &out };
        auto err = hipModuleLaunchKernel(hip_func_, grid_x_,1,1, block_x_,1,1, 0,
                                          stream.get(), args, nullptr);
        if (err != hipSuccess)
            throw_ov_exception(fmt::format("RocmAttn {}: {}", kind_, hipGetErrorString(err)));
    }

    // pe(V) fusion: Form B pe computation
    // inputs[0]=V, inputs[1]=softmax, inputs[2]=QKV(extra), inputs[3]=filter, inputs[4]=bias
    // outputs[0]=AV_out, outputs[1]=fgc_out (pe+attn result, pre-allocated buffer)
    //
    // FGC output buffer is pre-allocated (lifespan_start=0) so it has a valid GPU address
    // at AV Execute time despite being produced logically by FGC (which runs later).
    // pe_add3 computes fgc_out = AV_out + pe_conv(V) + bias in a single memory pass.
    // pe(V) fusion: rocmlir-gen pe_conv (handles non-contiguous V) + 3-way add
    // pe_conv uses rock.transform to correctly handle V's non-contiguous layout
    // (V from VariadicSplitAlias has head stride = C_head×H×W ≠ dv×H×W).
    // pe_add3: fgc_out = AV_out + pe_work + bias (single-pass, no extra D2D copy)
    if (has_pe_conv_ && inputs.size() >= 5 && outputs.size() >= 2 && !wbs.mutable_buffers.empty()) {
        void* v_flat  = const_cast<void*>(inputs[0].get());  // V [K*H*W] (via VariadicSplitAlias)
        void* filter  = const_cast<void*>(inputs[3].get());  // pe filter [K×1×3×3]
        void* bias_pe = const_cast<void*>(inputs[4].get());  // pe bias [K]
        void* fgc_out = outputs[1].get();                    // FGC output buffer (pre-allocated)
        void* pe_work = wbs.mutable_buffers[0].get();        // conv intermediate workspace

        const int K_pe = pe_out_elems_ / (H_pe_ * W_pe_);

        // Step 1: rocmlir-gen depthwise conv(V, filter) → pe_work
        // (filter-first arg order for rocmlir-gen kernel)
        {
            void* pe_args[] = { &filter, &v_flat, &pe_work };
            auto err2 = hipModuleLaunchKernel(pe_func_, pe_grid_x_,1,1, pe_block_x_,1,1, 0,
                                               stream.get(), pe_args, nullptr);
            if (err2 != hipSuccess)
                throw_ov_exception(fmt::format("RocmAttn pe_conv: {}", hipGetErrorString(err2)));
        }

        // Step 2: fgc_out = AV_out + pe_work + bias (3-way add, no D2D copy)
        kernel::launch_pe_add3_fp16(fgc_out, out, pe_work, bias_pe, K_pe, H_pe_ * W_pe_, stream.get());
    }
}

// Note: The factory registration is done in rocm_attention_fusion.cpp
// via a custom factory function since this op reuses ov::op::v0::MatMul nodes.

}  // namespace rocm_gpu
}  // namespace ov
