// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bert_self_attention.hpp"
#include <rocm_operation_registry.hpp>
#include <rocm/runtime.hpp>
#include <openvino/core/except.hpp>
#include <fmt/format.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>

namespace ov {
namespace rocm_gpu {

namespace {

static std::string find_tool(const std::string& name) {
    for (const auto& dir : std::vector<std::string>{
            "/home/rocmlir_install/bin/", "/opt/rocm/bin/", "/usr/local/bin/"}) {
        std::string p = dir + name;
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

static std::string make_mlir(int64_t seq, int64_t heads, int64_t dim,
                              int64_t num_cu, const std::string& arch) {
    int64_t hidden   = heads * dim;
    int64_t qkv_flat = heads * seq * dim;

    std::ostringstream s;

    // Bias: BERT attn_mask is [1, 1, seq, seq] — only seq*seq elements.
    // We pass the flat [seq*seq] buffer. Inside rock.attention elementwise block,
    // the bias memref is also [seq, seq] and we broadcast by indexing only (sq, sk).
    int64_t bias_flat = seq * seq;

    // Affine maps for Q/K/V: interleaved [seq, heads*dim] → [heads, seq, dim]
    s << fmt::format("#map_qv = affine_map<(d0, d1, d2) -> (d1 * {} + d0 * {} + d2)>\n", hidden, dim);
    // K: [seq, heads*dim] → [heads, dim, sk]  (transposed in head-dim)
    s << fmt::format("#map_k  = affine_map<(d0, d1, d2) -> (d2 * {} + d0 * {} + d1)>\n", hidden, dim);
    // Out: [heads, sq, dim] → [seq, heads*dim] interleaved  (same layout as qv)
    s << fmt::format("#map_out = affine_map<(d0, d1, d2) -> ((d0 * {} + d1) * {} + d2)>\n", seq, dim);
    s << "#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>\n";
    // Broadcast bias over heads: (h, sq, sk) → sq*seq + sk  (ignores head dim h)
    s << fmt::format("#map_bias_bc = affine_map<(d0, d1, d2) -> (d1 * {} + d2)>\n", seq);

    s << fmt::format("#xq   = #rock.transform_map<#map_qv  by [<Unmerge{{{},{},{}}} [\"g\",\"sq\",\"hd\"] at [0,1,2] -> [\"raw\"] at [0]>] bounds = [{},{},{}] -> [{}]>\n",
                     heads,seq,dim, heads,seq,dim, qkv_flat);
    s << fmt::format("#xkt  = #rock.transform_map<#map_k   by [<Unmerge{{{},{},{}}} [\"g\",\"hd\",\"sk\"] at [0,1,2] -> [\"raw\"] at [0]>] bounds = [{},{},{}] -> [{}]>\n",
                     heads,dim,seq, heads,dim,seq, qkv_flat);
    s << fmt::format("#xv   = #rock.transform_map<#map_qv  by [<Unmerge{{{},{},{}}} [\"g\",\"sk\",\"hv\"] at [0,1,2] -> [\"raw\"] at [0]>] bounds = [{},{},{}] -> [{}]>\n",
                     heads,seq,dim, heads,seq,dim, qkv_flat);
    s << fmt::format("#xout = #rock.transform_map<#map_out by [<Unmerge{{{},{},{}}} [\"g\",\"sq\",\"hv\"] at [0,1,2] -> [\"raw\"] at [0]>] bounds = [{},{},{}] -> [{}]>\n",
                     heads,seq,dim, heads,seq,dim, qkv_flat);
    // Bias: broadcast [seq*seq] flat buffer to [heads,seq,seq] by ignoring the head dim.
    // map_bias_bc: (h, sq, sk) → sq*seq + sk  —  h is ignored, so head dim broadcasts.
    s << fmt::format("#xbias = #rock.transform_map<#map_bias_bc by [<Unmerge{{{},{},{}}} [\"g\",\"sq\",\"sk\"] at [0,1,2] -> [\"raw\"] at [0]>] bounds = [{},{},{}] -> [{}]>\n",
                     heads,seq,seq, heads,seq,seq, bias_flat);

    s << fmt::format("module attributes {{mhal.arch = \"amdgcn-amd-amdhsa:{}\"}} {{\n", arch);
    s << fmt::format("  func.func @rock_attention(%a0: memref<{}xf16>, %a1: memref<{}xf16>, %a2: memref<{}xf16>, %a3: memref<{}xf16>, %a4: memref<{}xf16>) attributes {{kernel, mhal.arch = \"amdgcn-amd-amdhsa:{}\", num_cu = {} : i32}} {{\n",
                     qkv_flat,qkv_flat,qkv_flat,bias_flat,qkv_flat, arch, num_cu);
    s << fmt::format("    %Q = rock.transform %a0 by #xq   : memref<{}xf16> to memref<{}x{}x{}xf16>\n", qkv_flat,heads,seq,dim);
    s << fmt::format("    %K = rock.transform %a1 by #xkt  : memref<{}xf16> to memref<{}x{}x{}xf16>\n", qkv_flat,heads,dim,seq);
    s << fmt::format("    %V = rock.transform %a2 by #xv   : memref<{}xf16> to memref<{}x{}x{}xf16>\n", qkv_flat,heads,seq,dim);
    // Bias: pass as [heads,seq,seq] via broadcast affine_map from [seq*seq] flat buffer
    s << fmt::format("    %B = rock.transform %a3 by #xbias: memref<{}xf16> to memref<{}x{}x{}xf16>\n", bias_flat,heads,seq,seq);
    s << fmt::format("    %O = rock.transform %a4 by #xout : memref<{}xf16> to memref<{}x{}x{}xf16>\n", qkv_flat,heads,seq,dim);
    s << "    rock.attention{\n";
    s << fmt::format("     qk = %Q * %K : memref<{}x{}x{}xf16>, memref<{}x{}x{}xf16>\n", heads,seq,dim, heads,dim,seq);
    // elementwise block: add attention bias (broadcast [seq,seq] → [heads,seq,seq] via map)
    s << fmt::format("     qk = elementwise otherIns(%B : memref<{}x{}x{}xf16>) {{\n", heads,seq,seq);
    s << fmt::format("    ^bb0(%b0: memref<{}x{}x{}xf16>, %b1: memref<{}x{}x{}xf16>, %b2: memref<{}x{}x{}xf16>):\n",
                     heads,seq,seq, heads,seq,seq, heads,seq,seq);
    s << fmt::format("      %tmp = memref.alloc() {{alignment = 64 : i64}} : memref<{}x{}x{}xf16>\n", heads,seq,seq);
    s << fmt::format("      linalg.generic {{indexing_maps = [#id3, #id3, #id3], iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}} ins(%b0, %b1 : memref<{}x{}x{}xf16>, memref<{}x{}x{}xf16>) outs(%tmp : memref<{}x{}x{}xf16>) {{\n",
                     heads,seq,seq, heads,seq,seq, heads,seq,seq);
    s << "      ^bb1(%x: f16, %y: f16, %z: f16):\n";
    s << "        %s = arith.addf %x, %y : f16\n";
    s << "        linalg.yield %s : f16\n";
    s << "      }\n";
    s << fmt::format("      memref.copy %tmp, %b2 : memref<{}x{}x{}xf16> to memref<{}x{}x{}xf16>\n",
                     heads,seq,seq, heads,seq,seq);
    s << "      rock.yield\n";
    s << "    }\n";
    s << fmt::format("     %O = softmax(qk) * %V : memref<{}x{}x{}xf16> -> memref<{}x{}x{}xf16>\n",
                     heads,seq,dim, heads,seq,dim);
    s << fmt::format("    }} {{features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>, firstGemmIndices = array<i64: 0>, numHeadsKV = {} : i32, numHeadsQ = {} : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>}}\n",
                     heads, heads);
    s << "    return\n";
    s << "  }\n";
    s << "}\n";
    return s.str();
}

}  // namespace

BertSelfAttentionOp::BertSelfAttentionOp(
        const CreationContext& context,
        const std::shared_ptr<ov::Node>& node,
        IndexCollection&& inputIds,
        IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto attn = std::dynamic_pointer_cast<nodes::BertSelfAttention>(node);
    OPENVINO_ASSERT(attn, "BertSelfAttentionOp: expected BertSelfAttention node");

    seq_len_   = attn->get_seq_len();
    num_heads_ = attn->get_num_heads();
    head_dim_  = attn->get_head_dim();
    fprintf(stderr, "[BertAttn] Constructor: seq=%lld heads=%lld dim=%lld bias_elems=%lld\n",
            (long long)seq_len_, (long long)num_heads_, (long long)head_dim_,
            (long long)ov::shape_size(node->get_input_partial_shape(3).to_shape()));

    const auto& props = context.device().props();
    std::string arch = props.gcnArchName;
    if (auto pos = arch.find(':'); pos != std::string::npos) arch = arch.substr(0, pos);
    int64_t num_cu = props.multiProcessorCount;

    // Generate and compile MLIR
    const std::string mlir_src = make_mlir(seq_len_, num_heads_, head_dim_, num_cu, arch);

    const std::string pid_str = std::to_string(getpid());
    const std::string mlir_path = "/tmp/ov_bert_attn_" + arch + "_" + pid_str + ".mlir";
    {
        std::ofstream f(mlir_path);
        OPENVINO_ASSERT(f.good(), "BertSelfAttentionOp: can't write MLIR to ", mlir_path);
        f << mlir_src;
    }

    const std::string driver = find_tool("rocmlir-driver");
    const std::string cmd = driver + " --arch " + arch + " --kernel-pipeline=full " +
                            mlir_path + " 2>&1";
    int ec = 0;
    const std::string compiled = run_cmd(cmd, ec);

    if (ec != 0 || compiled.size() < 100) {
        // Keep MLIR file for debugging, print it to stderr
        fprintf(stderr, "[BertAttn] MLIR compilation failed (ec=%d). Generated MLIR:\n%s\n",
                ec, mlir_src.c_str());
        fprintf(stderr, "[BertAttn] Compiler output:\n%s\n", compiled.c_str());
        OPENVINO_THROW("BertSelfAttentionOp: compilation failed (ec=", ec, "): ",
                       compiled.substr(0, 400));
    }
    std::remove(mlir_path.c_str());

    // Parse kernel name
    {
        const std::string kn_marker = "symbol_name=";
        auto kn_pos = compiled.find(kn_marker);
        if (kn_pos != std::string::npos) {
            size_t kns = kn_pos + kn_marker.size();
            size_t kne = compiled.find('"', kns);
            if (kne != std::string::npos && kns != kne)
                kernel_name_ = compiled.substr(kns, kne - kns);
        }
    }
    // Parse grid/block
    {
        auto find_int = [&](const std::string& marker) -> unsigned {
            size_t p = compiled.find(marker);
            return (p != std::string::npos) ? std::stoi(compiled.substr(p + marker.size(), 20)) : 0;
        };
        grid_x_  = find_int("grid_size = ");
        block_x_ = find_int("block_size = ");
    }
    if (block_x_ == 0) block_x_ = 32;
    if (grid_x_  == 0) grid_x_  = static_cast<unsigned>(num_heads_) * 8;

    // Extract HSACO binary from compiled output (same logic as RocmAttentionMatMulOp)
    auto extract_bin = [&]() -> std::vector<char> {
        const std::string marker = "bin = \"";
        auto bn_pos = compiled.find(marker);
        if (bn_pos == std::string::npos) return {};
        std::vector<char> bytes;
        size_t i = bn_pos + marker.size();
        while (i < compiled.size()) {
            char c = compiled[i];
            if (c == '"') break;
            if (c == '\\' && i + 1 < compiled.size()) {
                char nx = compiled[i+1];
                if (nx == 'n') { bytes.push_back('\n'); i += 2; continue; }
                if (nx == '\\') { bytes.push_back('\\'); i += 2; continue; }
                if (i + 2 < compiled.size() && std::isxdigit(compiled[i+1]) && std::isxdigit(compiled[i+2])) {
                    char h[3] = {compiled[i+1], compiled[i+2], 0};
                    bytes.push_back(static_cast<char>(std::strtol(h, nullptr, 16)));
                    i += 3;
                    continue;
                }
            }
            bytes.push_back(c);
            ++i;
        }
        return bytes;
    };

    hsaco_ = extract_bin();
    OPENVINO_ASSERT(hsaco_.size() > 100,
        "BertSelfAttentionOp: HSACO extraction failed (size=", hsaco_.size(), ")");

    // Load HIP module
    hipError_t err = hipModuleLoadData(&module_, hsaco_.data());
    OPENVINO_ASSERT(err == hipSuccess,
        "BertSelfAttentionOp: hipModuleLoadData failed: ", hipGetErrorString(err));
    err = hipModuleGetFunction(&func_, module_, kernel_name_.c_str());
    OPENVINO_ASSERT(err == hipSuccess,
        "BertSelfAttentionOp: hipModuleGetFunction '", kernel_name_, "' failed: ",
        hipGetErrorString(err));

    fprintf(stderr, "[BertAttn] Attention kernel: arch=%s grid=%u block=%u HSACO=%zu B\n",
            arch.c_str(), grid_x_, block_x_, hsaco_.size());
}

void BertSelfAttentionOp::Execute(const InferenceRequestContext& context,
                                   Inputs inputs,
                                   Outputs outputs,
                                   const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 4 && outputs.size() == 1);
    // Kernel args: a0=q[seq*heads*dim], a1=k[seq*heads*dim], a2=v[seq*heads*dim],
    //              a3=bias[seq*seq] (flat, broadcast over heads via affine_map), a4=out
    void* q_ptr    = const_cast<void*>(inputs[0].get());
    void* k_ptr    = const_cast<void*>(inputs[1].get());
    void* v_ptr    = const_cast<void*>(inputs[2].get());
    void* bias_ptr = const_cast<void*>(inputs[3].get());
    void* out_ptr  = outputs[0].get();

    void* args[] = { &q_ptr, &k_ptr, &v_ptr, &bias_ptr, &out_ptr };
    hipError_t err = hipModuleLaunchKernel(
        func_,
        grid_x_, 1, 1,
        block_x_, 1, 1,
        0,
        context.getThreadContext().stream().get(),
        args, nullptr);
    OPENVINO_ASSERT(err == hipSuccess,
        "BertSelfAttentionOp: launch failed: ", hipGetErrorString(err));
}

OPERATION_REGISTER(BertSelfAttentionOp, BertSelfAttention);

}  // namespace rocm_gpu
}  // namespace ov
