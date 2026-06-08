// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// RocmAttentionSoftmaxOp: MIGraphX-compiled fused scale+softmax kernel for attention.
// Replaces: QKT_output → Mul(scale) → Softmax(axis=-1)
// MIGraphX fuses this into: convert_mul_reduce_max_sub_exp_reduce_sum_div_convert

#include "rocm_attention_softmax.hpp"
#include <rocm_operation_registry.hpp>
#include <error.hpp>
#include <fmt/format.h>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <unistd.h>

static std::string softmax_run_cmd(const std::string& cmd, int& exit_code) {
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

bool RocmAttentionSoftmaxOp::isEnabled() {
    // Attention softmax fusion disabled: MIOpen already handles this efficiently
    // with FULL HIP graph compatibility. External tool approach removed.
    return false;
}

// ── Global kernel cache ───────────────────────────────────────────────────────
struct SoftmaxKernelEntry {
    std::vector<char> hsaco;
    std::string kernel_name;
    int grid_x = 1, block_x = 256;
    hipModule_t   module   = nullptr;
    hipFunction_t function = nullptr;
};

static std::mutex g_mu;
static std::unordered_map<std::string, SoftmaxKernelEntry> g_cache;

// Softmax kernel compilation — currently disabled (isEnabled() returns false).
// MIOpen already provides efficient softmax with FULL HIP graph compatibility.
static SoftmaxKernelEntry compile_softmax_kernel(
        int nh, int N, float scale, const std::string& arch) {

    OPENVINO_THROW("RocmAttentionSoftmaxOp: disabled (use MIOpen softmax)");

    (void)nh; (void)N; (void)scale; (void)arch;
    static std::atomic<int> seq{0};
    const std::string base = "/tmp/ov_softmax_" + std::to_string(getpid())
                           + "_" + std::to_string(seq.fetch_add(1));

    const char* py_env = std::getenv("ROCMLIR_PYTHON");
    const std::string python = py_env ? py_env : "python3";
    const std::string onnx_file = base + ".onnx";
    const std::string py_file   = base + ".py";

    // Generate ONNX: scale_mul + softmax on [1, nh, N, N]
    {
        std::ofstream f(py_file);
        f << "import onnx\n"
          << "from onnx import helper, TensorProto, numpy_helper\n"
          << "import numpy as np\n"
          << "B,nh,N=" << 1 << "," << nh << "," << N << "\n"
          << "sc=" << scale << "\n"
          << "dt=TensorProto.FLOAT16\n"
          << "xi=helper.make_tensor_value_info('x',dt,[B,nh,N,N])\n"
          << "yo=helper.make_tensor_value_info('y',dt,[B,nh,N,N])\n"
          << "sv=numpy_helper.from_array(np.array([sc],dtype=np.float16).reshape([1,1,1,1]),'scale_val')\n"
          << "nodes=[\n"
          << "  helper.make_node('Mul',['x','scale_val'],['xs']),\n"
          << "  helper.make_node('Softmax',['xs'],['y'],axis=-1),\n"
          << "]\n"
          << "g=helper.make_graph(nodes,'softmax',[xi],[yo],[sv])\n"
          << "m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)])\n"
          << "onnx.save(m,'" << onnx_file << "')\nprint('OK')\n";
    }

    int ec = 0;
    auto res = softmax_run_cmd(python + " " + py_file + " 2>/dev/null", ec);
    std::remove(py_file.c_str());
    if (ec != 0 || res.find("OK") == std::string::npos)
        OPENVINO_THROW("RocmSoftmax: failed to generate ONNX (ec=", ec, ")");

    // (dead code - this path is never reached when isEnabled() returns false)
    std::string mig_out;
    std::remove(onnx_file.c_str());

    // Find the fused softmax MLIR module (contains "reduce_max" or "softmax")
    const std::string size_marker = std::to_string((size_t)nh * N * N) + "xf16";
    std::string mlir_module;
    size_t start = 0;
    while (true) {
        auto pos = mig_out.find("module {", start);
        if (pos == std::string::npos) break;
        auto nxt = mig_out.find("module {", pos + 8);
        const std::string cand = mig_out.substr(pos, (nxt != std::string::npos ? nxt : mig_out.size()) - pos);
        if ((cand.find("reduce_max") != std::string::npos || cand.find("reduce_sum") != std::string::npos)
            && cand.find(size_marker) != std::string::npos) {
            mlir_module = cand; break;
        }
        start = pos + 8;
    }
    if (mlir_module.empty())
        OPENVINO_THROW("RocmSoftmax: MLIR not found for softmax (nh=", nh, " N=", N, ")");

    const char* drv_env = std::getenv("ROCMLIR_DRIVER");
    const std::string driver = drv_env ? drv_env : "/root/rocmlir_install/bin/rocmlir-driver";
    const std::string ir_file = base + ".mlir";
    { std::ofstream f(ir_file); f << mlir_module; }
    const std::string drv_cmd = driver + " --arch " + arch + " --kernel-pipeline=full " + ir_file;
    const std::string drv_out = softmax_run_cmd(drv_cmd, ec);
    std::remove(ir_file.c_str());

    SoftmaxKernelEntry k;
    auto km_pos = drv_out.find("kernel_metadata<\"");
    auto bn_pos = drv_out.find("bin = \"");
    if (km_pos == std::string::npos || bn_pos == std::string::npos)
        OPENVINO_THROW("RocmSoftmax: no kernel in rocmlir-driver output");

    size_t kns = km_pos + 17, kne = drv_out.find('"', kns);
    k.kernel_name = drv_out.substr(kns, kne - kns);
    // grid/block sizes are inside metadata = {...} after kernel_metadata
    auto gs_pos = drv_out.find("grid_size = ", km_pos);
    auto bs_pos = drv_out.find("block_size = ", km_pos);
    if (gs_pos != std::string::npos) k.grid_x  = std::stoi(drv_out.substr(gs_pos + 12, 20));
    if (bs_pos != std::string::npos) k.block_x = std::stoi(drv_out.substr(bs_pos + 13, 20));

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
        OPENVINO_THROW("RocmSoftmax: invalid ELF (size=", k.hsaco.size(), ")");

    auto err = hipModuleLoadData(&k.module, k.hsaco.data());
    if (err != hipSuccess)
        OPENVINO_THROW("RocmSoftmax: hipModuleLoadData: ", hipGetErrorString(err));
    err = hipModuleGetFunction(&k.function, k.module, k.kernel_name.c_str());
    if (err != hipSuccess)
        OPENVINO_THROW("RocmSoftmax: hipModuleGetFunction(", k.kernel_name, "): ", hipGetErrorString(err));

    std::cerr << "[AttnSoftmax] Compiled scale+softmax kernel: " << k.kernel_name
              << " nh=" << nh << " N=" << N << " grid=" << k.grid_x << "\n";
    return k;
}

// ── Constructor ───────────────────────────────────────────────────────────────
RocmAttentionSoftmaxOp::RocmAttentionSoftmaxOp(
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

    const int nh    = std::stoi(get_rt("rocm_attn_nheads"));
    const int N     = std::stoi(get_rt("rocm_attn_seq"));
    const float scale = std::stof(get_rt("rocm_attn_scale"));

    std::string arch = ctx.device().props().gcnArchName;
    auto c = arch.find(':');
    if (c != std::string::npos) arch = arch.substr(0, c);

    const std::string cache_key = "softmax_nh" + std::to_string(nh)
        + "_N" + std::to_string(N)
        + "_sc" + std::to_string(scale)
        + "_" + arch;

    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_cache.find(cache_key);
    if (it == g_cache.end())
        it = g_cache.emplace(cache_key, compile_softmax_kernel(nh, N, scale, arch)).first;

    hip_module_ = it->second.module;
    hip_func_   = it->second.function;
    grid_x_     = it->second.grid_x;
    block_x_    = it->second.block_x;
}

// ── Execute ───────────────────────────────────────────────────────────────────
void RocmAttentionSoftmaxOp::Execute(
        const InferenceRequestContext& ctx,
        Inputs inputs, Outputs outputs,
        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() >= 1 && outputs.size() == 1);
    auto& stream = ctx.getThreadContext().stream();
    void* in  = const_cast<void*>(inputs[0].get());
    void* out = outputs[0].get();

    // mlir_softmax_fused(x_in, x_out)
    void* args[] = { &in, &out };
    auto err = hipModuleLaunchKernel(hip_func_, grid_x_,1,1, block_x_,1,1, 0,
                                      stream.get(), args, nullptr);
    if (err != hipSuccess)
        throw_ov_exception(fmt::format("RocmSoftmax: {}", hipGetErrorString(err)));
}

}  // namespace rocm_gpu
}  // namespace ov
