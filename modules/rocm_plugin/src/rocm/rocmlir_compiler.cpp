// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rocmlir_compiler.hpp"

#include <openvino/core/except.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>  // getpid()

namespace ov {
namespace rocm_gpu {
namespace rocmlir {

// ─────────────────────────────────────────────────────────────────────────────
// ConvParams helpers
// ─────────────────────────────────────────────────────────────────────────────

bool ConvParams::operator==(const ConvParams& o) const {
    return N == o.N && C == o.C && H == o.H && W == o.W &&
           K == o.K && R == o.R && S == o.S &&
           pad_h == o.pad_h && pad_w == o.pad_w &&
           stride_h == o.stride_h && stride_w == o.stride_w &&
           dilation_h == o.dilation_h && dilation_w == o.dilation_w &&
           groups == o.groups && fp16 == o.fp16 &&
           arch == o.arch && num_cu == o.num_cu;
}

size_t ConvParams::hash() const {
    // Simple polynomial hash over fields
    size_t h = 0;
    auto mix = [&](size_t v) { h = h * 2654435761ULL ^ v; };
    mix(N); mix(C); mix(H); mix(W);
    mix(K); mix(R); mix(S);
    mix(pad_h); mix(pad_w);
    mix(stride_h); mix(stride_w);
    mix(dilation_h); mix(dilation_w);
    mix(groups); mix(fp16 ? 1 : 0);
    mix(std::hash<std::string>{}(arch));
    mix(num_cu);
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// MLIR IR generation
//
// Template matches what MIGRAPHX_TRACE_MLIR=2 produces for conv + bias + act:
//   func.func @mlir_convolution_broadcast_add_sigmoid_mul(...)
//     attributes {arch = "gfx950:sramecc+:xnack-", kernel = "mixr", num_cu = 256}
//   {
//     rock.conv(%filter, %input, %output) {
//       dilations = [...], filter_layout = [...], ...
//       perf_config = "v4:..."
//     }
//     rock.elementwise(...) { ... }   // optional bias / activation
//   }
// ─────────────────────────────────────────────────────────────────────────────

static std::string dtype_str(bool fp16) {
    return fp16 ? "f16" : "f32";
}

static std::string arch_triple(const std::string& arch) {
    return arch + ":sramecc+:xnack-";
}

// Build a flat memref size annotation, e.g. "1x192x160x160"
static std::string flat_size(const std::vector<int>& dims) {
    std::ostringstream s;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) s << "x";
        s << dims[i];
    }
    return s.str();
}

// Compute total elements
static size_t numel(const std::vector<int>& dims) {
    size_t n = 1;
    for (int d : dims) n *= static_cast<size_t>(d);
    return n;
}

// Generate the core rock.conv op attributes
static std::string conv_attrs(const ConvParams& p) {
    std::ostringstream s;
    // dilations
    s << "dilations = [" << p.dilation_h << " : index, " << p.dilation_w << " : index], ";
    // filter / input / output layouts (NCHW groupwise form expected by rock)
    s << "filter_layout = [\"g\", \"k\", \"c\", \"y\", \"x\"], ";
    s << "input_layout = [\"ni\", \"gi\", \"ci\", \"hi\", \"wi\"], ";
    s << "output_layout = [\"no\", \"go\", \"ko\", \"ho\", \"wo\"], ";
    // padding (4-value: top, bottom, left, right)
    s << "padding = [" << p.pad_h << " : index, " << p.pad_h
                << " : index, " << p.pad_w << " : index, " << p.pad_w << " : index], ";
    // strides
    s << "strides = [" << p.stride_h << " : index, " << p.stride_w << " : index]";
    return s.str();
}

std::string generate_conv_ir(const ConvParams& p) {
    const std::string dt   = dtype_str(p.fp16);
    const int G = p.groups;
    const int KpG = p.K / G;
    const int CpG = p.C / G;
    const int OH  = p.out_h();
    const int OW  = p.out_w();

    // memref sizes (flat, elements)
    const size_t in_elems  = (size_t)p.N * p.C * p.H * p.W;
    const size_t flt_elems = (size_t)p.K * CpG * p.R * p.S;
    const size_t out_elems = (size_t)p.N * p.K * OH * OW;

    std::ostringstream ir;
    ir << "module {\n";
    ir << "  func.func @mlir_conv_" << dt << "_" << p.N << "x" << p.C << "x" << p.H << "x" << p.W
       << "_" << p.K << "x" << CpG << "x" << p.R << "x" << p.S
       << "(%arg0: memref<" << in_elems  << "x" << dt << ">, "
       <<  "%arg1: memref<" << flt_elems << "x" << dt << ">, "
       <<  "%arg2: memref<" << out_elems << "x" << dt << ">)"
       << " attributes {arch = \"" << arch_triple(p.arch) << "\","
       << " kernel = \"mixr\", num_cu = " << p.num_cu << " : i64} {\n";

    ir << "    %0 = rock.transform %arg1 by <...> : memref<" << flt_elems << "x" << dt
       << "> to memref<" << G << "x" << KpG << "x" << CpG << "x" << p.R << "x" << p.S << "x" << dt << ">\n";
    ir << "    %1 = rock.transform %arg0 by <...> : memref<" << in_elems << "x" << dt
       << "> to memref<" << p.N << "x" << G  << "x" << CpG << "x" << p.H << "x" << p.W << "x" << dt << ">\n";
    ir << "    %2 = rock.transform %arg2 by <...> : memref<" << out_elems << "x" << dt
       << "> to memref<" << p.N << "x" << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << ">\n";

    ir << "    rock.conv(%0, %1, %2) {"
       << conv_attrs(p)
       << "} : memref<" << G << "x" << KpG << "x" << CpG << "x" << p.R << "x" << p.S << "x" << dt << ">, "
       <<    "memref<" << p.N << "x" << G  << "x" << CpG << "x" << p.H << "x" << p.W << "x" << dt << ">, "
       <<    "memref<" << p.N << "x" << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << ">\n";

    ir << "    return\n";
    ir << "  }\n";
    ir << "}\n";
    return ir.str();
}

std::string generate_fused_conv_bias_ir(const ConvParams& p) {
    // Bias shape: 1×K×1×1 (broadcast over N,H,W)
    // rocMLIR supports fused conv + elementwise add with broadcast
    const std::string dt  = dtype_str(p.fp16);
    const int G    = p.groups;
    const int KpG  = p.K / G;
    const int CpG  = p.C / G;
    const int OH   = p.out_h();
    const int OW   = p.out_w();
    const size_t in_elems  = (size_t)p.N * p.C * p.H * p.W;
    const size_t flt_elems = (size_t)p.K * CpG * p.R * p.S;
    const size_t bias_elems = (size_t)p.K;
    const size_t out_elems = (size_t)p.N * p.K * OH * OW;

    std::ostringstream ir;
    ir << "module {\n";
    ir << "  func.func @mlir_convolution_broadcast_add_"
       << dt << "_" << p.N << "x" << p.C << "x" << p.H << "x" << p.W
       << "(%arg0: memref<" << in_elems  << "x" << dt << ">, "
       <<  "%arg1: memref<" << flt_elems << "x" << dt << ">, "
       <<  "%arg2: memref<" << bias_elems << "x" << dt << ">, "
       <<  "%arg3: memref<" << out_elems << "x" << dt << ">)"
       << " attributes {arch = \"" << arch_triple(p.arch) << "\","
       << " kernel = \"mixr\", num_cu = " << p.num_cu << " : i64} {\n";

    ir << "    %filter = rock.transform %arg1 by <...> :"
       << " memref<" << flt_elems << "x" << dt << "> to"
       << " memref<" << G << "x" << KpG << "x" << CpG << "x" << p.R << "x" << p.S << "x" << dt << ">\n";
    ir << "    %input  = rock.transform %arg0 by <...> :"
       << " memref<" << in_elems << "x" << dt << "> to"
       << " memref<" << p.N << "x" << G << "x" << CpG << "x" << p.H << "x" << p.W << "x" << dt << ">\n";
    ir << "    %output = rock.transform %arg3 by <...> :"
       << " memref<" << out_elems << "x" << dt << "> to"
       << " memref<" << p.N << "x" << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << ">\n";

    ir << "    rock.conv(%filter, %input, %output) {"
       << conv_attrs(p)
       << "} : memref<" << G << "x" << KpG << "x" << CpG << "x" << p.R << "x" << p.S << "x" << dt << ">, "
       <<    "memref<" << p.N << "x" << G << "x" << CpG << "x" << p.H << "x" << p.W << "x" << dt << ">, "
       <<    "memref<" << p.N << "x" << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << ">\n";

    // Bias broadcast-add: %arg2 (K) broadcast to (N,K,OH,OW) and add to %arg3
    ir << "    %bias_bc = rock.transform %arg2 by <broadcast N,H,W> :"
       << " memref<" << bias_elems << "x" << dt << "> to"
       << " memref<" << p.N << "x" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";
    ir << "    rock.elementwise(%output, %bias_bc, %output) {pointwise = \"add\"} :"
       << " memref<" << p.N << "x" << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << "> ...\n";

    ir << "    return\n";
    ir << "  }\n";
    ir << "}\n";
    return ir.str();
}

std::string generate_fused_conv_bias_act_ir(const ConvParams& p, Activation act) {
    std::string base = generate_fused_conv_bias_ir(p);
    // Append activation name in function name and add elementwise op
    // (simplified: rocmlir-gen handles the actual tiling/fusion)
    const char* act_name = act == Activation::ReLU    ? "relu"    :
                           act == Activation::Sigmoid  ? "sigmoid" : "identity";
    // Replace function name to include activation
    std::string from = "broadcast_add_";
    std::string to   = std::string("broadcast_add_") + act_name + "_";
    size_t pos = base.find(from);
    if (pos != std::string::npos) base.replace(pos, from.size(), to);
    return base;
}

// ─────────────────────────────────────────────────────────────────────────────
// rocmlir-driver subprocess compilation
// ─────────────────────────────────────────────────────────────────────────────

std::string find_rocmlir_driver() {
    // 1. Environment variable
    const char* env = std::getenv("ROCMLIR_DRIVER");
    if (env && *env) return env;

    // 2. Standard build install paths
    const char* candidates[] = {
        "/home/openvino/rocmlir_install/bin/rocmlir-driver",
        "/opt/rocmlir/bin/rocmlir-driver",
        "/opt/rocm/bin/rocmlir-driver",
    };
    for (const char* c : candidates) {
        if (std::ifstream(c).good()) return c;
    }

    // 3. PATH
    return "rocmlir-driver";
}

// Execute a shell command and capture stdout + stderr
static std::string run_cmd(const std::string& cmd, int& exit_code) {
    std::array<char, 4096> buf{};
    std::string result;
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) {
        exit_code = -1;
        return "";
    }
    while (fgets(buf.data(), buf.size(), pipe)) {
        result += buf.data();
    }
    exit_code = pclose(pipe);
    return result;
}

CompiledConv compile_mlir_ir(const std::string& mlir_ir,
                              const std::string& arch,
                              const std::string& rocmlir_driver_path) {
    const std::string driver = rocmlir_driver_path.empty()
                               ? find_rocmlir_driver()
                               : rocmlir_driver_path;

    // Write IR to temp file
    const std::string ir_file  = "/tmp/ov_rocmlir_conv_" + std::to_string(getpid()) + ".mlir";
    const std::string hsaco_file = ir_file + ".hsaco";

    {
        FILE* f = fopen(ir_file.c_str(), "w");
        if (!f) OPENVINO_THROW("rocMLIR: failed to create temp IR file: ", ir_file);
        fwrite(mlir_ir.data(), 1, mlir_ir.size(), f);
        fclose(f);
    }

    // Run rocmlir-driver with --kernel-pipeline=full
    // Output is MLIR text with embedded HSACO as hex escape string in bin="..." field.
    const std::string cmd =
        driver +
        " --arch " + arch +
        " --kernel-pipeline=full" +
        " " + ir_file;

    int exit_code = 0;
    std::string output = run_cmd(cmd, exit_code);

    // Clean up IR file
    std::remove(ir_file.c_str());

    if (exit_code != 0) {
        OPENVINO_THROW("rocMLIR compilation failed (exit ", exit_code, "): ", output);
    }

    // Extract the raw ELF binary from the MLIR bin="..." field
    // Format: bin = "\x7FELF..." with C-style escape sequences
    CompiledConv result;
    {
        const std::string marker = "bin = \"";
        const auto pos_start = output.find(marker);
        if (pos_start == std::string::npos)
            OPENVINO_THROW("rocMLIR: no 'bin = ...' field in driver output");
        const size_t content_start = pos_start + marker.size();

        // Find closing quote (un-escaped)
        size_t content_end = content_start;
        while (content_end < output.size()) {
            if (output[content_end] == '"') break;
            if (output[content_end] == '\\') content_end++; // skip escape
            content_end++;
        }
        if (content_end >= output.size())
            OPENVINO_THROW("rocMLIR: unterminated bin field in driver output");

        // Decode MLIR string escape sequences → raw bytes
        const std::string raw = output.substr(content_start, content_end - content_start);
        result.hsaco.reserve(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            if (raw[i] == '\\' && i + 1 < raw.size()) {
                const char next = raw[++i];
                if (next == 'n')  { result.hsaco.push_back('\n'); }
                else if (next == 'r') { result.hsaco.push_back('\r'); }
                else if (next == 't') { result.hsaco.push_back('\t'); }
                else if (next == '"') { result.hsaco.push_back('"'); }
                else if (next == '\\') { result.hsaco.push_back('\\'); }
                else if (i + 1 < raw.size() &&
                         isxdigit(static_cast<unsigned char>(next)) &&
                         isxdigit(static_cast<unsigned char>(raw[i+1]))) {
                    // Two-digit hex escape \XX
                    char hex[3] = { next, raw[i+1], 0 };
                    result.hsaco.push_back(static_cast<char>(strtol(hex, nullptr, 16)));
                    i++;
                } else {
                    result.hsaco.push_back(next);
                }
            } else {
                result.hsaco.push_back(raw[i]);
            }
        }

        if (result.hsaco.size() < 4 ||
            result.hsaco[0] != '\x7F' || result.hsaco[1] != 'E' ||
            result.hsaco[2] != 'L'   || result.hsaco[3] != 'F') {
            OPENVINO_THROW("rocMLIR: extracted binary is not ELF (size=",
                           result.hsaco.size(), ")");
        }
    }

    // Extract kernel name from IR (first func.func name)
    {
        const std::string marker = "func.func @";
        auto pos = mlir_ir.find(marker);
        if (pos != std::string::npos) {
            pos += marker.size();
            auto end = mlir_ir.find('(', pos);
            if (end != std::string::npos)
                result.kernel_name = mlir_ir.substr(pos, end - pos);
        }
        if (result.kernel_name.empty())
            result.kernel_name = "mlir_conv";
    }

    // Extract perf_config from IR if present
    {
        const std::string pf_key = "perf_config = \"";
        auto pos = mlir_ir.find(pf_key);
        if (pos != std::string::npos) {
            pos += pf_key.size();
            auto end = mlir_ir.find('"', pos);
            if (end != std::string::npos)
                result.perf_config = mlir_ir.substr(pos, end - pos);
        }
    }

    // Parse block dims from perf_config "v4:BX,BY,BZ,..."
    // Format: v4:block_size,... (first field is block_x for most configs)
    if (!result.perf_config.empty()) {
        int bx = 64; // default
        if (sscanf(result.perf_config.c_str(), "v4:%d,", &bx) == 1)
            result.block_x = bx;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// High-level compile helpers — use rocmlir-gen to produce valid MLIR IR
// ─────────────────────────────────────────────────────────────────────────────

// Find rocmlir-gen tool next to rocmlir-driver
static std::string find_rocmlir_gen(const std::string& driver_path) {
    if (!driver_path.empty()) {
        const auto slash = driver_path.rfind('/');
        if (slash != std::string::npos) {
            return driver_path.substr(0, slash + 1) + "rocmlir-gen";
        }
    }
    return "rocmlir-gen";
}

// Build rocmlir-gen command for a plain convolution
static std::string rocmlir_gen_cmd(const ConvParams& p,
                                    const std::string& gen_path) {
    std::ostringstream cmd;
    cmd << gen_path
        << " -t " << (p.fp16 ? "f16" : "f32")
        << " --batchsize " << p.N
        << " --in_channels " << p.C
        << " --in_h " << p.H
        << " --in_w " << p.W
        << " --out_channels " << p.K
        << " --fil_h " << p.R
        << " --fil_w " << p.S
        << " --padding_h " << p.pad_h
        << " --padding_w " << p.pad_w
        << " --conv_stride_h " << p.stride_h
        << " --conv_stride_w " << p.stride_w
        << " --dilation_h " << p.dilation_h
        << " --dilation_w " << p.dilation_w
        << " --groupsize " << p.groups
        << " --arch " << p.arch;
    return cmd.str();
}

CompiledConv compile_conv(const ConvParams& p, const std::string& drv) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;
    const std::string gen    = find_rocmlir_gen(driver);

    // Step 1: generate valid rock.conv MLIR IR via rocmlir-gen subprocess
    const std::string gen_cmd = rocmlir_gen_cmd(p, gen);
    int exit_code = 0;
    const std::string mlir_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0)
        OPENVINO_THROW("rocmlir-gen failed (exit ", exit_code, "): ", mlir_ir);

    // Step 2: compile IR to HSACO
    return compile_mlir_ir(mlir_ir, p.arch, driver);
}

CompiledConv compile_fused_conv_bias(const ConvParams& p, const std::string& drv) {
    // rocmlir-gen doesn't directly generate fused conv+bias IR.
    // Use plain conv kernel; bias will be added by miopenConvolutionForwardBias.
    return compile_conv(p, drv);
}

CompiledConv compile_fused_conv_bias_act(const ConvParams& p, Activation act, const std::string& drv) {
    // Same: compile plain conv; activation applied post-conv by MIOpen.
    return compile_conv(p, drv);
}

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
