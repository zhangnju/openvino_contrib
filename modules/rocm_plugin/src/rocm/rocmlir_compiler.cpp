// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rocmlir_compiler.hpp"

#include <openvino/core/except.hpp>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <unordered_set>
#include <sys/stat.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>  // getpid()
#include <hip/hip_runtime.h>

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
           arch == o.arch && num_cu == o.num_cu &&
           C_full == o.C_full && c_start == o.c_start;
}

size_t ConvParams::hash() const {
    // FNV-1a inspired: better avalanche than simple polynomial
    size_t h = 14695981039346656037ULL;
    auto mix = [&](size_t v) {
        h ^= v;
        h *= 1099511628211ULL;
        // Extra rotation to prevent low-bit aliasing between similar values
        h ^= (h >> 33);
    };
    mix(static_cast<size_t>(N));
    mix(static_cast<size_t>(C));
    mix(static_cast<size_t>(H));
    mix(static_cast<size_t>(W));
    mix(static_cast<size_t>(K));
    mix(static_cast<size_t>(R));
    mix(static_cast<size_t>(S));
    mix(static_cast<size_t>(pad_h));
    mix(static_cast<size_t>(pad_w));
    mix(static_cast<size_t>(stride_h));
    mix(static_cast<size_t>(stride_w));
    mix(static_cast<size_t>(dilation_h));
    mix(static_cast<size_t>(dilation_w));
    mix(static_cast<size_t>(groups));
    mix(fp16 ? 1ULL : 0ULL);
    mix(static_cast<size_t>(num_cu));
    mix(std::hash<std::string>{}(arch));
    mix(static_cast<size_t>(C_full));
    mix(static_cast<size_t>(c_start));
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
#ifdef ROCMLIR_DRIVER_PATH
        ROCMLIR_DRIVER_PATH,  // CMake-configured path (highest priority)
#endif
        "/root/rocmlir_install/bin/rocmlir-driver",
        "/home/rocmlir_install/bin/rocmlir-driver",
        "/home/openvino/rocmlir_install/bin/rocmlir-driver",
        "/home/openvino/rocmlir-driver",
        "/opt/rocmlir/bin/rocmlir-driver",
        "/opt/rocm/bin/rocmlir-driver",
        nullptr
    };
    for (const char* c : candidates) {
        if (c && std::ifstream(c).good()) return c;
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

// Core compilation function: write MLIR to temp file, run rocmlir-driver with given pipeline,
// parse output (bin="..." ELF binary + kernel metadata). Used by both rock.conv and migraphx paths.
static CompiledConv compile_mlir_with_pipeline(const std::string& mlir_ir,
                                                const std::string& arch,
                                                const std::string& driver,
                                                const std::string& pipeline,
                                                const std::string& file_prefix) {
    static std::atomic<int> seq_core{0};
    const std::string ir_file = "/tmp/" + file_prefix + "_"
        + std::to_string(getpid()) + "_"
        + std::to_string(seq_core.fetch_add(1)) + ".mlir";

    {
        FILE* f = fopen(ir_file.c_str(), "w");
        if (!f) OPENVINO_THROW("rocMLIR: failed to create temp IR file: ", ir_file);
        fwrite(mlir_ir.data(), 1, mlir_ir.size(), f);
        fclose(f);
    }

    // Enable tuning fallback so injected perf_config gracefully degrades
    // to heuristic when the config is invalid for the given conv shape.
    const std::string cmd = driver
        + " --arch " + arch
        + " --tuning-fallback=true"
        + " --kernel-pipeline=" + pipeline
        + " " + ir_file;

    int exit_code = 0;
    std::string output = run_cmd(cmd, exit_code);
    std::remove(ir_file.c_str());

    if (exit_code != 0)
        OPENVINO_THROW("rocMLIR compilation failed (exit ", exit_code, "): ", output);

    // ── Parse output: extract HSACO binary + kernel metadata ──────────────
    CompiledConv result;

    // Extract ELF binary from bin="..." field
    {
        const std::string marker = "bin = \"";
        const auto pos_start = output.find(marker);
        if (pos_start == std::string::npos)
            OPENVINO_THROW("rocMLIR: no 'bin = ...' field in driver output");
        const size_t content_start = pos_start + marker.size();
        size_t content_end = content_start;
        while (content_end < output.size()) {
            if (output[content_end] == '"') break;
            if (output[content_end] == '\\') content_end++;
            content_end++;
        }
        if (content_end >= output.size())
            OPENVINO_THROW("rocMLIR: unterminated bin field in driver output");
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
        // Validate ELF magic
        if (result.hsaco.size() < 4 ||
            static_cast<unsigned char>(result.hsaco[0]) != 0x7F ||
            result.hsaco[1] != 'E' || result.hsaco[2] != 'L' || result.hsaco[3] != 'F') {
            OPENVINO_THROW("rocMLIR: decoded bin field is not a valid ELF (size=",
                           result.hsaco.size(), ")");
        }
    }

    // Extract kernel name from kernel_metadata<"name", ...>
    {
        const std::string km_marker = "#gpu.kernel_metadata<\"";
        auto pos = output.find(km_marker);
        if (pos != std::string::npos) {
            pos += km_marker.size();
            auto end = output.find('"', pos);
            if (end != std::string::npos)
                result.kernel_name = output.substr(pos, end - pos);
        }
        if (result.kernel_name.empty()) {
            const std::string gb_marker = "gpu.binary @";
            pos = output.find(gb_marker);
            if (pos != std::string::npos) {
                pos += gb_marker.size();
                auto end = output.find(' ', pos);
                if (end != std::string::npos) {
                    std::string mod = output.substr(pos, end - pos);
                    const std::string suffix = "_module";
                    if (mod.size() > suffix.size() &&
                        mod.substr(mod.size() - suffix.size()) == suffix)
                        mod = mod.substr(0, mod.size() - suffix.size());
                    result.kernel_name = mod;
                }
            }
        }
        if (result.kernel_name.empty()) result.kernel_name = "mlir_conv";
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

    // Extract block_size and grid_size from kernel metadata
    {
        int bsz = 256, gsz = 1;
        auto parse_int_field = [&](const char* key, int& out) {
            const std::string k = std::string(key) + " = ";
            auto p = output.find(k);
            if (p != std::string::npos) {
                p += k.size();
                out = std::stoi(output.substr(p, 20));
            }
        };
        parse_int_field("block_size", bsz);
        parse_int_field("grid_size",  gsz);
        result.block_x = bsz;
        result.grid_x  = gsz;
    }

    return result;
}

CompiledConv compile_mlir_ir(const std::string& mlir_ir,
                              const std::string& arch,
                              const std::string& rocmlir_driver_path) {
    const std::string driver = rocmlir_driver_path.empty()
                               ? find_rocmlir_driver()
                               : rocmlir_driver_path;
    return compile_mlir_with_pipeline(mlir_ir, arch, driver, "full", "ov_rocmlir_conv");
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

// ── Per-shape perf_config tuning cache (persistent across process restarts) ──
// On first compile for each unique conv shape, evaluates the tuning space
// from rocmlir-gen --emit-tuning-space and selects the best config via
// hipEvent timing. Results are persisted to a JSON file.
//
// Cache file location (in order of precedence):
//   1. ROCMLIR_TUNING_CACHE env var
//   2. ~/.cache/ov_rocmlir_tuning_<arch>.json
//
// File format: {"<shape_hash_hex>": "<perf_config_string>", ...}
// Empty string means "use rocmlir-gen default".

static std::string get_cache_path(const std::string& arch) {
    const char* env = std::getenv("ROCMLIR_TUNING_CACHE");
    if (env && *env) return env;
    const char* home = std::getenv("HOME");
    std::string base = home ? std::string(home) : "/tmp";
    // sanitize arch for filename (gfx1201:sramecc+:xnack- → gfx1201)
    std::string arch_clean = arch.substr(0, arch.find(':'));
    return base + "/.cache/ov_rocmlir_tuning_" + arch_clean + ".json";
}

static void ensure_dir(const std::string& path) {
    const auto slash = path.rfind('/');
    if (slash == std::string::npos) return;
    const std::string dir = path.substr(0, slash);
    ::mkdir(dir.c_str(), 0755);
}

struct PerfConfigCache {
    std::mutex mu;
    std::unordered_map<size_t, std::string> map;  // shape_hash → best perf_config
    bool loaded = false;
    std::string arch_for_file;  // arch string for file naming

    static PerfConfigCache& instance() { static PerfConfigCache c; return c; }

    // Load from persistent JSON file
    void load(const std::string& arch) {
        std::lock_guard<std::mutex> lk(mu);
        if (loaded) return;
        loaded = true;
        arch_for_file = arch;

        const std::string path = get_cache_path(arch);
        std::ifstream f(path);
        if (!f.is_open()) return;

        std::string line;
        while (std::getline(f, line)) {
            // Parse simple JSON: "hash_hex": "config"
            auto q1 = line.find('"');
            if (q1 == std::string::npos) continue;
            auto q2 = line.find('"', q1 + 1);
            if (q2 == std::string::npos) continue;
            auto q3 = line.find('"', q2 + 1);
            if (q3 == std::string::npos) continue;
            auto q4 = line.find('"', q3 + 1);
            if (q4 == std::string::npos) continue;
            const std::string hash_hex = line.substr(q1 + 1, q2 - q1 - 1);
            const std::string cfg = line.substr(q3 + 1, q4 - q3 - 1);
            try {
                size_t key = std::stoull(hash_hex, nullptr, 16);
                map[key] = cfg;
            } catch (...) {}
        }
        std::cerr << "[rocMLIR-tune] Loaded " << map.size() << " cached perf_configs from " << path << std::endl;
    }

    // Save updated cache to persistent JSON file
    void save() {
        const std::string path = get_cache_path(arch_for_file);
        ensure_dir(path);
        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open()) return;
        f << "{\n";
        bool first = true;
        for (const auto& [key, cfg] : map) {
            if (!first) f << ",\n";
            // Write hash as hex for readability
            char hex[32];
            snprintf(hex, sizeof(hex), "%016zx", key);
            f << "  \"" << hex << "\": \"" << cfg << "\"";
            first = false;
        }
        f << "\n}\n";
    }
};

// ── Persistent HSACO kernel cache (survives process restarts) ─────────────────
// Saves compiled HSACO binaries to disk so re-runs skip rocmlir-driver compilation.
// File: ~/.cache/ov_rocmlir_cache_<arch>/<hash_hex>.hsaco + <hash_hex>.meta
struct HsacoKernelCache {
    std::mutex mu;
    bool loaded = false;
    std::string arch_for_file;

    static HsacoKernelCache& instance() { static HsacoKernelCache c; return c; }

    static std::string get_dir(const std::string& arch) {
        const char* home = std::getenv("HOME");
        std::string base = home ? std::string(home) : "/tmp";
        std::string arch_clean = arch.substr(0, arch.find(':'));
        return base + "/.cache/ov_rocmlir_cache_" + arch_clean;
    }

    // Try to load a cached CompiledConv for the given shape hash.
    // Returns true and fills `out` if found, false otherwise.
    bool load(size_t key, const std::string& arch, CompiledConv& out) {
        std::lock_guard<std::mutex> lk(mu);
        const std::string dir = get_dir(arch);
        char hex[32]; snprintf(hex, sizeof(hex), "%016zx", key);
        const std::string hsaco_path = dir + "/" + hex + ".hsaco";
        const std::string meta_path  = dir + "/" + hex + ".meta";

        // Meta format: kernel_name\ngrid_x\nblock_x\nflags\n
        // flags = bias_fused | (silu_fused<<1) | (skip_add_fused<<2)
        std::ifstream mf(meta_path);
        if (!mf.is_open()) return false;
        std::string kname, sgrid, sblock, sflags;
        if (!std::getline(mf, kname) || !std::getline(mf, sgrid) || !std::getline(mf, sblock))
            return false;

        // Read HSACO binary
        std::ifstream hf(hsaco_path, std::ios::binary);
        if (!hf.is_open()) return false;
        out.hsaco.assign(std::istreambuf_iterator<char>(hf), {});
        if (out.hsaco.size() < 4 || out.hsaco[0] != '\x7F') return false;

        out.kernel_name = kname;
        out.grid_x  = std::stoi(sgrid);
        out.block_x = std::stoi(sblock);
        if (std::getline(mf, sflags) && !sflags.empty()) {
            int flags = std::stoi(sflags);
            out.bias_fused     = (flags & 1) != 0;
            out.silu_fused     = (flags & 2) != 0;
            out.skip_add_fused = (flags & 4) != 0;
        } else {
            // Legacy cache without flags line: assume bias+silu (SliceConv default)
            out.bias_fused = true;
            out.silu_fused = true;
            out.skip_add_fused = false;
        }
        return true;
    }

    // Save a compiled kernel to the persistent cache.
    void save(size_t key, const std::string& arch, const CompiledConv& c) {
        std::lock_guard<std::mutex> lk(mu);
        const std::string dir = get_dir(arch);
        ::mkdir(dir.c_str(), 0755);
        char hex[32]; snprintf(hex, sizeof(hex), "%016zx", key);

        // Write HSACO binary
        {
            std::ofstream hf(dir + "/" + hex + ".hsaco", std::ios::binary | std::ios::trunc);
            hf.write(c.hsaco.data(), (std::streamsize)c.hsaco.size());
        }
        // Write meta: kernel_name\ngrid_x\nblock_x\nflags\n
        {
            int flags = (c.bias_fused ? 1 : 0) | (c.silu_fused ? 2 : 0) | (c.skip_add_fused ? 4 : 0);
            std::ofstream mf(dir + "/" + hex + ".meta", std::ios::trunc);
            mf << c.kernel_name << "\n" << c.grid_x << "\n" << c.block_x << "\n" << flags << "\n";
        }
    }
};

// Time a single kernel compile+execute cycle using hipEvents.
// Returns average milliseconds over n_runs, or 1e9 on failure.
static float time_perf_config(const ConvParams& p,
                               const std::string& gen_path,
                               const std::string& driver_path,
                               const std::string& perf_cfg,
                               int n_warmup = 2, int n_runs = 5) {
    const std::string gen_cmd = gen_path
        + " -t " + (p.fp16 ? "f16" : "f32")
        + " --batchsize " + std::to_string(p.N)
        + " --in_channels " + std::to_string(p.C)
        + " --in_h " + std::to_string(p.H)
        + " --in_w " + std::to_string(p.W)
        + " --out_channels " + std::to_string(p.K)
        + " --fil_h " + std::to_string(p.R)
        + " --fil_w " + std::to_string(p.S)
        + " --padding_h " + std::to_string(p.pad_h)
        + " --padding_w " + std::to_string(p.pad_w)
        + " --conv_stride_h " + std::to_string(p.stride_h)
        + " --conv_stride_w " + std::to_string(p.stride_w)
        + " --dilation_h " + std::to_string(p.dilation_h)
        + " --dilation_w " + std::to_string(p.dilation_w)
        + " --groupsize " + std::to_string(p.groups)
        + " --arch " + p.arch
        + " --perf_config=" + perf_cfg;

    int exit_code = 0;
    const std::string mlir_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0) return 1e9f;

    // Compile IR
    CompiledConv compiled;
    try { compiled = compile_mlir_ir(mlir_ir, p.arch, driver_path); }
    catch (...) { return 1e9f; }

    // Allocate GPU buffers for timing
    const size_t in_bytes  = (size_t)p.N * p.C * p.H * p.W * (p.fp16 ? 2 : 4);
    const size_t flt_bytes = (size_t)p.K * (p.C / p.groups) * p.R * p.S * (p.fp16 ? 2 : 4);
    const size_t out_bytes = (size_t)p.N * p.K * compiled.grid_x / (compiled.block_x / compiled.block_x)
                             * (p.fp16 ? 2 : 4);
    const size_t out_b = (size_t)p.N * p.K * p.out_h() * p.out_w() * (p.fp16 ? 2 : 4);

    void *d_in = nullptr, *d_flt = nullptr, *d_out = nullptr;
    if (hipMalloc(&d_in, in_bytes) != hipSuccess) return 1e9f;
    if (hipMalloc(&d_flt, flt_bytes) != hipSuccess) { hipFree(d_in); return 1e9f; }
    if (hipMalloc(&d_out, out_b) != hipSuccess) { hipFree(d_in); hipFree(d_flt); return 1e9f; }
    hipMemset(d_in, 0, in_bytes);
    hipMemset(d_flt, 0, flt_bytes);

    // Load kernel
    hipModule_t module = nullptr;
    hipFunction_t func = nullptr;
    if (hipModuleLoadData(&module, compiled.hsaco.data()) != hipSuccess) {
        hipFree(d_in); hipFree(d_flt); hipFree(d_out);
        return 1e9f;
    }
    if (hipModuleGetFunction(&func, module, compiled.kernel_name.c_str()) != hipSuccess) {
        hipModuleUnload(module);
        hipFree(d_in); hipFree(d_flt); hipFree(d_out);
        return 1e9f;
    }

    // Run kernel (filter=arg0, input=arg1, output=arg2)
    void* args[] = { &d_flt, &d_in, &d_out };
    auto run_once = [&]() {
        hipModuleLaunchKernel(func, compiled.grid_x, 1, 1, compiled.block_x, 1, 1,
                              0, nullptr, args, nullptr);
    };

    // Warmup
    for (int i = 0; i < n_warmup; ++i) run_once();
    hipDeviceSynchronize();

    // Timed
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);
    hipEventRecord(ev0, nullptr);
    for (int i = 0; i < n_runs; ++i) run_once();
    hipEventRecord(ev1, nullptr);
    hipDeviceSynchronize();

    float ms = 0.f;
    hipEventElapsedTime(&ms, ev0, ev1);
    hipEventDestroy(ev0); hipEventDestroy(ev1);

    hipModuleUnload(module);
    hipFree(d_in); hipFree(d_flt); hipFree(d_out);

    return ms / n_runs;
}

// Get the optimal perf_config for a given conv shape.
// First call: runs tuning (tries all configs from --emit-tuning-space quick).
// Subsequent calls: returns cached result.
static std::string get_tuned_perf_config(const ConvParams& p,
                                          const std::string& gen_path,
                                          const std::string& driver_path) {
    // 1. User override
    const char* env_cfg = std::getenv("ROCMLIR_PERF_CONFIG");
    if (env_cfg && *env_cfg) return env_cfg;

    // 2. Tuning mode: enabled by ROCMLIR_ENABLE_TUNING=1.
    // When not set or set to 0, uses fast heuristic (RDNA3/RDNA4) or rocmlir-gen default.
    // NOTE: gfx1100-specific tuning configs must be validated via ROCMLIR_ENABLE_TUNING
    // before adding as heuristic — invalid thread block sizes cause rocMLIR compile failures.
    // gfx1151 (RDNA3.5 APU) uses rocmlir-gen default (emit-tuning-space) — v3: heuristic
    // causes "Lowering failed" on this arch; let ROCMLIR_ENABLE_TUNING=1 find valid configs.
    const char* enable_tuning = std::getenv("ROCMLIR_ENABLE_TUNING");
    if (!enable_tuning || std::string(enable_tuning) != "1") {
        // Fast heuristic for RDNA3 (gfx1100) and RDNA4 (gfx12xx) FP16 without full tuning.
        // gfx1151 (RDNA3.5) excluded: v3: configs cause Lowering failures on this arch.
        const bool is_gfx1151 = p.arch.find("gfx1151") != std::string::npos;
        const bool is_rdna = !is_gfx1151 &&
                              (p.arch.find("gfx12") != std::string::npos ||
                               p.arch.find("gfx11") != std::string::npos);
        if (is_rdna && p.fp16) {
            // Arch-specific fast heuristic (validated via ROCMLIR_ENABLE_TUNING profiling).
            // gfx1201 (RDNA4, 64 CU): configs from yolo26x tuning on actual R9700 hardware.
            // gfx1100 (RDNA3, 96 CU): same v3: base configs (similar wave32 tuning works).
            // Use ROCMLIR_ENABLE_TUNING=1 for per-shape optimal configs on each arch.
            if (p.R == 1 && p.S == 1) {
                const int out_elems = p.N * p.K * (p.H/p.stride_h) * (p.W/p.stride_w);
                if (out_elems >= 1000000) return "v3:128,256,8,64,32,8,1,1,2,1,1";
                if (out_elems >= 200000)  return "v3:64,128,4,32,64,8,1,1,2,1,1";
                if (out_elems >= 50000)   return "v3:64,32,8,16,32,8,1,2,2,1,1";
                return "v3:32,64,8,32,16,8,1,2,2,1,1";
            }
            if (p.R == 3 && p.S == 3) {
                if (p.stride_h == 1) return "v3:64,256,4,32,64,8,1,1,2,1,1";
                return "v3:128,128,8,32,16,8,1,2,2,1,1";
            }
        }
        return "";  // Use rocmlir-gen heuristic
    }
    // ROCMLIR_ENABLE_TUNING=1: fall through to exhaustive search + persistent cache.

    // 3. Load persistent cache on first call
    auto& cache = PerfConfigCache::instance();
    cache.load(p.arch);

    // 4. Check per-shape cache (memory + file)
    const size_t key = p.hash();
    {
        std::lock_guard<std::mutex> lk(cache.mu);
        auto it = cache.map.find(key);
        if (it != cache.map.end()) return it->second;
    }

    // 5. Curated v3: config candidates (RDNA4/wavefront-32 style)
    // rocmlir-gen --emit-tuning-space only returns v4: configs for f16+gfx1201, but
    // v3: configs (wavefront=32) are significantly faster on RDNA4 (gfx11xx/gfx12xx).
    // This set was derived by profiling typical YOLOv8 conv shapes on gfx1201 hardware.
    static const std::vector<std::string> rdna_v3_candidates = {
        // ── Core 29-entry search space (covers most conv shapes) ─────────────
        "v3:16,16,8,16,16,16,1,2,2,1,1",
        "v3:16,32,8,16,16,16,1,2,2,1,1",
        "v3:16,64,8,16,16,8,1,2,2,1,1",
        "v3:32,16,8,16,16,8,1,2,2,1,1",
        "v3:32,64,8,32,16,8,1,2,2,1,1",
        "v3:32,128,2,32,64,8,1,1,2,1,1",
        "v3:32,128,2,32,64,8,1,2,2,1,1",
        "v3:32,256,2,32,64,8,1,1,2,1,1",
        "v3:32,256,4,16,128,4,1,2,2,1,1",
        "v3:64,32,8,16,32,8,1,2,2,1,1",
        "v3:64,32,8,32,16,8,1,2,2,1,1",
        "v3:64,64,2,64,64,8,1,1,2,1,1",
        "v3:64,64,2,64,64,8,1,2,2,1,1",
        "v3:64,64,8,32,32,8,1,2,2,1,1",
        "v3:64,128,4,32,64,8,1,1,2,1,1",   // our fastest for 192x80x80 3x3 (60.7µs)
        "v3:64,128,4,64,32,8,1,2,2,1,1",
        "v3:64,128,8,32,32,4,1,2,2,1,1",
        "v3:64,128,8,64,32,8,1,2,2,1,1",
        "v3:64,256,4,32,64,8,1,1,2,1,1",
        "v3:64,256,8,32,32,8,1,1,2,1,1",
        "v3:64,256,8,64,32,8,1,1,2,1,1",
        "v3:128,64,2,32,64,8,1,2,2,1,1",
        "v3:128,64,4,64,64,4,1,2,2,1,1",
        "v3:128,64,8,32,64,8,1,1,2,1,1",
        "v3:128,128,8,32,16,8,1,2,2,1,1",
        "v3:128,128,8,64,64,8,1,1,2,1,1",
        "v3:128,256,4,128,64,4,1,2,2,1,1",
        "v3:128,256,8,64,32,8,1,1,2,1,1",
        "v3:256,128,8,32,32,8,1,1,2,1,1",
        // ── Additional candidates (extended search) ───────────────────────────
        "v3:64,64,4,64,64,8,1,1,2,1,1",
        "v3:64,64,8,64,64,8,1,1,2,1,1",
        "v3:128,64,8,64,32,8,1,1,2,1,1",   // ~65µs for 192x80x80
        "v3:128,128,4,32,64,8,1,1,2,1,1",
        "v3:128,128,8,32,64,8,1,1,2,1,1",
        "v3:256,64,8,64,32,8,1,1,2,1,1",
        "v3:256,128,4,128,64,4,1,2,2,1,1",
        "v3:256,256,8,128,32,8,1,1,2,1,1",
        // ── Depthwise/small convolution candidates ────────────────────────────
        "v3:16,16,8,16,16,16,1,1,2,1,1",
        "v3:16,32,8,16,32,16,1,1,2,1,1",
        "v3:32,16,8,16,16,16,1,2,2,1,1",
        "v3:32,32,8,32,16,16,1,2,2,1,1",
        "v3:64,64,8,64,32,8,1,2,2,1,1",
        // ── Stride-2 convolution candidates (yolo26x downsampling layers) ────
        // These shapes (768x80 stride=2, 384x160 stride=2 etc.) are slowest in yolo26x
        "v3:32,256,4,16,128,4,1,2,2,1,1",   // best for 768x80 stride=2
        "v3:16,32,8,16,16,16,1,2,2,1,1",    // best for 384x160 stride=2
        "v3:64,128,2,32,64,8,1,2,2,1,1",
        "v3:128,64,2,64,32,8,1,2,2,1,1",
        "v3:64,256,2,32,64,8,1,2,2,1,1",
        "v3:128,128,2,64,64,8,1,2,2,1,1",
        "v3:256,64,2,64,32,8,1,2,2,1,1",
        "v3:32,128,4,16,64,8,1,2,2,1,1",
        "v3:64,128,2,64,64,8,1,2,2,1,1",
        "v3:128,256,2,128,64,4,1,2,2,1,1",
        // ── gfx1201 profiling: configs seen in optimized traces ───────────────
        "v3:32,64,8,32,16,8,1,2,2,1,1",     // 1x1 conv (small output)
        "v3:32,16,8,16,16,8,1,2,2,1,1",     // depthwise / small 1x1
        "v3:64,256,8,64,32,8,1,1,2,1,1",    // 3x3 stride-1 large K
        "v3:32,256,2,32,64,8,1,1,2,1,1",    // 3x3 stride-2
        "v3:128,128,8,32,16,8,1,2,2,1,1",   // 3x3 stride-2 mid
        "v3:64,128,8,32,32,4,1,2,2,1,1",    // 3x3 stride-2 alt
        "v3:32,128,2,32,64,8,1,2,2,1,1",    // 1x1 wide
        // ── gfx1201 extended: BlockK=2/4/16 variants + nPerRepeat=2 variants ───
        // Expanding search space beyond curated 55 configs to find faster configs.
        // gfx1201 (64 CU, RDNA4) benefits from different occupancy than gfx1100 (96 CU).
        "v3:64,64,4,32,32,8,1,1,2,1,1",
        "v3:64,64,4,32,32,8,1,2,2,1,1",
        "v3:64,64,2,32,32,8,1,1,2,1,1",
        "v3:64,64,2,32,32,8,1,2,2,1,1",
        "v3:128,64,4,32,32,8,1,1,2,1,1",
        "v3:128,64,4,64,32,8,1,1,2,1,1",
        "v3:128,64,2,64,32,8,1,1,2,1,1",
        "v3:128,64,2,32,64,8,1,1,2,1,1",
        "v3:64,128,2,32,64,8,1,1,2,1,1",
        "v3:64,128,2,64,64,8,1,1,2,1,1",
        "v3:64,128,4,64,64,8,1,1,2,1,1",
        "v3:128,128,2,64,64,8,1,1,2,1,1",
        "v3:128,128,4,64,64,8,1,1,2,1,1",
        "v3:128,128,2,64,32,8,1,1,2,1,1",
        "v3:64,64,8,32,16,8,1,2,2,1,1",
        "v3:64,64,4,16,32,8,1,2,2,1,1",
        "v3:128,64,8,64,64,8,1,1,2,1,1",
        "v3:128,64,4,128,32,4,1,1,2,1,1",
        "v3:64,128,8,64,64,8,1,1,2,1,1",
        "v3:64,256,4,64,64,4,1,1,2,1,1",
        "v3:32,64,4,32,32,8,1,2,2,1,1",
        "v3:32,64,2,32,32,8,1,2,2,1,1",
        "v3:16,64,4,16,32,8,1,2,2,1,1",
        "v3:32,128,4,32,64,8,1,2,2,1,1",
        "v3:32,128,8,32,64,8,1,2,2,1,1",
        "v3:64,64,16,32,32,8,1,2,2,1,1",
        "v3:128,64,16,64,32,8,1,1,2,1,1",
        "v3:64,128,16,64,32,8,1,1,2,1,1",
    };
    // Add v3: candidates for RDNA3 (gfx1100) and RDNA4 (gfx12xx).
    // gfx1151 (RDNA3.5 APU) excluded: v3: configs cause Lowering failures on this arch.
    // gfx1151 uses emit-tuning-space to discover valid configs via ROCMLIR_ENABLE_TUNING=1.
    const bool is_gfx1151_tune = p.arch.find("gfx1151") != std::string::npos;
    const bool is_rdna = !is_gfx1151_tune &&
                         (p.arch.find("gfx12") != std::string::npos ||
                          p.arch.find("gfx11") != std::string::npos);

    // 6. Build candidate list
    // RDNA3 (gfx1100) and RDNA4 (gfx1201): curated v3: (wavefront-32) only.
    //   emit-tuning-space returns v4: configs (wavefront-64) which are SLOWER on RDNA
    //   because RDNA natively uses wavefront-32; v4: is emulated 2×wave32 with overhead.
    // gfx1151 and gfx950/CDNA3: emit-tuning-space (v3: not applicable or Lowering fails).
    const bool is_gfx12 = p.arch.find("gfx12") != std::string::npos;
    std::vector<std::string> configs;
    if (is_rdna && p.fp16) {
        // All RDNA (gfx11xx/gfx12xx): curated v3: (wave32) beats emit-tuning-space v4: (wave64)
        configs = rdna_v3_candidates;
    } else {
        const std::string tuning_cmd = gen_path
            + " -t " + (p.fp16 ? "f16" : "f32")
            + " --batchsize " + std::to_string(p.N)
            + " --in_channels " + std::to_string(p.C)
            + " --in_h " + std::to_string(p.H)
            + " --in_w " + std::to_string(p.W)
            + " --out_channels " + std::to_string(p.K)
            + " --fil_h " + std::to_string(p.R)
            + " --fil_w " + std::to_string(p.S)
            + " --padding_h " + std::to_string(p.pad_h)
            + " --padding_w " + std::to_string(p.pad_w)
            + " --conv_stride_h " + std::to_string(p.stride_h)
            + " --conv_stride_w " + std::to_string(p.stride_w)
            + " --dilation_h " + std::to_string(p.dilation_h)
            + " --dilation_w " + std::to_string(p.dilation_w)
            + " --groupsize " + std::to_string(p.groups)
            + " --arch " + p.arch
            + " --emit-tuning-space quick";

        int exit_code = 0;
        const std::string space = run_cmd(tuning_cmd, exit_code);
        if (exit_code != 0 || space.empty()) {
            std::lock_guard<std::mutex> lk(cache.mu);
            cache.map[key] = "";
            cache.save();
            return "";
        }

        std::istringstream ss(space);
        std::string line;
        while (std::getline(ss, line)) {
            if (!line.empty() && line[0] == 'v') configs.push_back(line);
        }
    }

    // Deduplicate
    {
        std::vector<std::string> deduped;
        std::unordered_set<std::string> seen;
        for (const auto& c : configs) {
            if (seen.insert(c).second) deduped.push_back(c);
        }
        configs = std::move(deduped);
    }

    // 7. Time each config, pick best
    std::string best_cfg;
    float best_ms = 1e9f;
    for (const auto& cfg : configs) {
        float ms = time_perf_config(p, gen_path, driver_path, cfg);
        if (ms < best_ms) { best_ms = ms; best_cfg = cfg; }
    }

    // Also time default (empty config)
    float default_ms = time_perf_config(p, gen_path, driver_path, "");
    if (default_ms <= best_ms) best_cfg = "";  // default is best

    std::cerr << "[rocMLIR-tune] N=" << p.N << " C=" << p.C << " H=" << p.H
              << " K=" << p.K << " R=" << p.R << " G=" << p.groups
              << " stride=" << p.stride_h << " fp16=" << p.fp16
              << " → best=" << (best_cfg.empty() ? "default" : best_cfg)
              << " (" << best_ms << "ms)" << std::endl;

    // Save to in-memory cache and persist to file
    {
        std::lock_guard<std::mutex> lk(cache.mu);
        cache.map[key] = best_cfg;
        cache.save();  // write updated cache to disk
    }
    return best_cfg;
}


// Forward declaration for fused_epilogue tuning
static std::string generate_fused_epilogue_ir(const ConvParams& p,
                                               bool with_skip,
                                               bool with_silu_add,
                                               const std::string& perf_cfg);

// ── fused_epilogue (migraphx dialect) perf_config tuning ─────────────────────
// Measures conv+bias+silu fused kernel time by compiling via generate_fused_epilogue_ir.
// Returns average ms over n_runs, or 1e9 on failure.
static float time_perf_config_fused_epilogue(const ConvParams& p,
                                              const std::string& driver_path,
                                              const std::string& cfg,
                                              int n_warmup = 2, int n_runs = 5) {
    // Compile fused_epilogue IR with the given perf_config
    std::string mlir_ir;
    try { mlir_ir = generate_fused_epilogue_ir(p, /*with_skip=*/false, /*with_silu_add=*/false, cfg); }
    catch (...) { return 1e9f; }

    CompiledConv compiled;
    try { compiled = compile_mlir_with_pipeline(mlir_ir, p.arch, driver_path, "migraphx,highlevel,gpu,rocdl,binary", "ov_fused_tune"); }
    catch (...) { return 1e9f; }
    if (compiled.hsaco.empty()) return 1e9f;

    const size_t elem_bytes = p.fp16 ? 2u : 4u;
    const size_t in_bytes   = (size_t)p.N * p.C * p.H * p.W * elem_bytes;
    const size_t flt_bytes  = (size_t)p.K * (p.C / p.groups) * p.R * p.S * elem_bytes;
    const size_t bias_bytes = (size_t)p.K * elem_bytes;
    const size_t out_bytes  = (size_t)p.N * p.K * p.out_h() * p.out_w() * elem_bytes;

    void *d_in = nullptr, *d_flt = nullptr, *d_bias = nullptr, *d_out = nullptr;
    auto cleanup = [&]() {
        if (d_in)   hipFree(d_in);
        if (d_flt)  hipFree(d_flt);
        if (d_bias) hipFree(d_bias);
        if (d_out)  hipFree(d_out);
    };
    if (hipMalloc(&d_in,   in_bytes)   != hipSuccess) { cleanup(); return 1e9f; }
    if (hipMalloc(&d_flt,  flt_bytes)  != hipSuccess) { cleanup(); return 1e9f; }
    if (hipMalloc(&d_bias, bias_bytes) != hipSuccess) { cleanup(); return 1e9f; }
    if (hipMalloc(&d_out,  out_bytes)  != hipSuccess) { cleanup(); return 1e9f; }
    hipMemset(d_in, 0, in_bytes); hipMemset(d_flt, 0, flt_bytes);
    hipMemset(d_bias, 0, bias_bytes);

    hipModule_t module = nullptr; hipFunction_t func = nullptr;
    if (hipModuleLoadData(&module, compiled.hsaco.data()) != hipSuccess) { cleanup(); return 1e9f; }
    if (hipModuleGetFunction(&func, module, compiled.kernel_name.c_str()) != hipSuccess) {
        hipModuleUnload(module); cleanup(); return 1e9f;
    }

    // fused_epilogue kernel: (input, filter, bias, output) - legacy arg order
    void* args[] = { &d_in, &d_flt, &d_bias, &d_out };
    auto run_once = [&]() {
        hipModuleLaunchKernel(func, compiled.grid_x, 1, 1, compiled.block_x, 1, 1,
                              0, nullptr, args, nullptr);
    };

    for (int i = 0; i < n_warmup; ++i) run_once();
    hipDeviceSynchronize();

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);
    hipEventRecord(ev0, nullptr);
    for (int i = 0; i < n_runs; ++i) run_once();
    hipEventRecord(ev1, nullptr);
    hipDeviceSynchronize();

    float ms = 0.f;
    hipEventElapsedTime(&ms, ev0, ev1);
    hipEventDestroy(ev0); hipEventDestroy(ev1);
    hipModuleUnload(module);
    cleanup();
    return ms / n_runs;
}

// Dedicated perf_config cache for fused_epilogue (conv+bias+silu) kernels.
// Stored separately from plain rock.conv cache because the optimal tile config
// may differ when SiLU epilogue is fused (changes compute/memory ratio).
struct FusedEpiloguePerfCache {
    std::mutex mu;
    std::unordered_map<size_t, std::string> map;
    bool loaded = false;
    std::string arch_for_file;

    static FusedEpiloguePerfCache& instance() { static FusedEpiloguePerfCache c; return c; }

    static std::string get_path(const std::string& arch) {
        const char* env = std::getenv("ROCMLIR_TUNING_CACHE_FUSED");
        if (env && *env) return env;
        const char* home = std::getenv("HOME");
        std::string base = home ? std::string(home) : "/tmp";
        std::string arch_clean = arch.substr(0, arch.find(':'));
        return base + "/.cache/ov_rocmlir_tuning_" + arch_clean + "_fused.json";
    }

    void load(const std::string& arch) {
        std::lock_guard<std::mutex> lk(mu);
        if (loaded) return;
        loaded = true; arch_for_file = arch;
        const std::string path = get_path(arch);
        std::ifstream f(path);
        if (!f.is_open()) return;
        std::string line;
        while (std::getline(f, line)) {
            auto q1=line.find('"'); if(q1==std::string::npos) continue;
            auto q2=line.find('"',q1+1); if(q2==std::string::npos) continue;
            auto q3=line.find('"',q2+1); if(q3==std::string::npos) continue;
            auto q4=line.find('"',q3+1); if(q4==std::string::npos) continue;
            try { map[std::stoull(line.substr(q1+1,q2-q1-1),nullptr,16)] =
                      line.substr(q3+1,q4-q3-1); }
            catch (...) {}
        }
        std::cerr << "[fused-tune] Loaded " << map.size() << " configs from " << path << std::endl;
    }

    void save() {
        const std::string path = get_path(arch_for_file);
        ensure_dir(path);
        std::ofstream f(path, std::ios::trunc);
        f << "{\n";
        bool first = true;
        for (const auto& [key, cfg] : map) {
            if (!first) f << ",\n";
            char hex[32]; snprintf(hex, sizeof(hex), "%016zx", key);
            f << "  \"" << hex << "\": \"" << cfg << "\"";
            first = false;
        }
        f << "\n}\n";
    }
};

// Get optimal perf_config for fused_epilogue (conv+bias+silu) kernel.
// Called when ROCMLIR_ENABLE_TUNING_FUSED=1; otherwise returns "" (use default).
// Uses the same candidate set as plain conv tuning (rdna_v3_candidates).
static std::string get_tuned_perf_config_fused(const ConvParams& p,
                                                const std::string& gen_path,
                                                const std::string& driver_path) {
    // Always attempt to load the fused cache (similar to plain conv cache behavior).
    // When ROCMLIR_ENABLE_TUNING_FUSED=1, run tuning search if shape not in cache.
    // When ROCMLIR_ENABLE_TUNING_FUSED=0 (default), use cached results from previous tuning.
    auto& cache = FusedEpiloguePerfCache::instance();
    cache.load(p.arch);

    const size_t key = p.hash() ^ static_cast<size_t>(0xF0504DEF00ULL);
    {
        std::lock_guard<std::mutex> lk(cache.mu);
        auto it = cache.map.find(key);
        if (it != cache.map.end()) return it->second;
    }

    // Shape not in fused cache: run tuning if enabled, else return ""
    const char* enable = std::getenv("ROCMLIR_ENABLE_TUNING_FUSED");
    if (!enable || std::string(enable) != "1") return "";

    // Use same v3 candidate set as plain conv (optimal for RDNA4 wave32)
    // Only tune for RDNA3/4 FP16 (where v3 configs are validated)
    const bool is_gfx1151 = p.arch.find("gfx1151") != std::string::npos;
    const bool is_rdna = !is_gfx1151 && (p.arch.find("gfx12") != std::string::npos ||
                                          p.arch.find("gfx11") != std::string::npos);
    if (!is_rdna || !p.fp16) return "";

    // Smaller candidate set focused on shapes common in fused conv+silu
    // Avoiding very large tiles that may be invalid for migraphx rock.conv
    static const std::vector<std::string> fused_candidates = {
        "", // default (rocMLIR heuristic)
        "v3:64,128,4,32,64,8,1,1,2,1,1",
        "v3:64,256,4,32,64,8,1,1,2,1,1",
        "v3:128,128,8,32,16,8,1,2,2,1,1",
        "v3:128,64,8,32,64,8,1,1,2,1,1",
        "v3:64,64,8,32,32,8,1,2,2,1,1",
        "v3:128,128,8,64,64,8,1,1,2,1,1",
        "v3:64,64,2,64,64,8,1,1,2,1,1",
        "v3:64,128,8,64,64,8,1,1,2,1,1",
        "v3:32,128,4,16,64,8,1,2,2,1,1",
        "v3:64,64,4,32,32,8,1,2,2,1,1",
        "v3:128,64,4,64,32,8,1,1,2,1,1",
        "v3:32,64,8,32,16,8,1,2,2,1,1",
        "v3:64,128,2,64,64,8,1,1,2,1,1",
        "v3:128,256,4,128,64,4,1,2,2,1,1",
        "v3:64,256,8,64,32,8,1,1,2,1,1",
        "v3:32,256,4,16,128,4,1,2,2,1,1",
        "v3:128,64,2,64,32,8,1,1,2,1,1",
        "v3:128,128,2,64,64,8,1,1,2,1,1",
        "v3:64,64,8,64,64,8,1,1,2,1,1",
    };

    std::string best_cfg;
    float best_ms = 1e9f;
    for (const auto& cfg : fused_candidates) {
        float ms = time_perf_config_fused_epilogue(p, driver_path, cfg);
        std::cerr << "[fused-tune] cfg=" << (cfg.empty() ? "default" : cfg)
                  << " → " << ms << "ms\n";
        if (ms < best_ms) { best_ms = ms; best_cfg = cfg; }
    }

    std::cerr << "[fused-tune] N=" << p.N << " C=" << p.C << " H=" << p.H
              << " K=" << p.K << " R=" << p.R << " stride=" << p.stride_h
              << " → best=" << (best_cfg.empty() ? "default" : best_cfg)
              << " (" << best_ms << "ms)\n";

    std::lock_guard<std::mutex> lk(cache.mu);
    cache.map[key] = best_cfg;
    cache.save();
    return best_cfg;
}

static std::string rocmlir_gen_cmd(const ConvParams& p,
                                    const std::string& gen_path,
                                    const std::string& perf_config = "") {
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
    if (!perf_config.empty()) {
        // No shell quoting needed: popen passes the string to /bin/sh -c,
        // and the perf_config value (e.g. "v3:64,...") contains no shell-special chars.
        // Extra \" would pass literal quote characters to rocmlir-gen, breaking parsing.
        cmd << " --perf_config=" << perf_config;
    }
    return cmd.str();
}

CompiledConv compile_conv(const ConvParams& p, const std::string& drv) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;
    const std::string gen    = find_rocmlir_gen(driver);

    // Optional: user-specified perf_config override
    const std::string perf_cfg = get_tuned_perf_config(p, gen, driver);

    // Step 1: generate valid rock.conv MLIR IR via rocmlir-gen subprocess
    const std::string gen_cmd = rocmlir_gen_cmd(p, gen, perf_cfg);
    int exit_code = 0;
    const std::string mlir_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0) {
        const std::string short_err = mlir_ir.size() > 500 ? mlir_ir.substr(0, 500) : mlir_ir;
        OPENVINO_THROW("rocmlir-gen failed (exit ", exit_code, "): ", short_err);
    }

    // Step 2: compile IR to HSACO
    return compile_mlir_ir(mlir_ir, p.arch, driver);
}

// ── MIGraphX dialect compilation: Conv + Bias + SiLU epilogue ───────────────
// Uses rocMLIR's 'migraphx' MLIR dialect (internal to rocMLIR, unrelated to MIGraphX framework)
// arbitrary elementwise epilogues. Compiled with:
//   rocmlir-driver -kernel-pipeline=migraphx,highlevel,gpu,rocdl,binary
//
// Advantages over rock.conv approach:
//   1. Epilogue fusion is handled by the migraphx pipeline automatically
//   2. Kernel names match MIGraphX (mlir_convolution_broadcast_add_sigmoid_mul)
//   3. No separate bias-add or SiLU kernel needed
//
// Currently used when ROCMLIR_EPILOGUE_FUSION=1 is set (experimental).
// The perf_config tuning (v3: candidates) is not applicable here — the
// migraphx pipeline selects its own tile configuration internally.

// Generate migraphx dialect MLIR for conv + bias + SiLU (+ optional skip-add + optional second SiLU+add).
//
// Epilogue variants supported:
//   !with_skip, !with_silu_add: conv+bias+silu       → 4-arg kernel (input,filter,bias → output)
//   with_skip,  !with_silu_add: conv+bias+silu+add   → 5-arg kernel (+ skip)
//   with_skip,  with_silu_add:  conv+bias+silu+add+silu+add → 6-arg kernel (+ skip + aux)
//
// The 6-arg variant fuses the C2f/C2PSA bottleneck pattern:
//   FC(conv+bias+silu+shortcut) → silu(fc_out) → add(silu, cv1_silu)
// where FC is the existing 5-arg kernel output, and the outer silu+add become the epilogue.
static std::string generate_fused_epilogue_ir(const ConvParams& p,
                                                         bool with_skip = false,
                                                         bool with_silu_add = false,
                                                         const std::string& perf_cfg = "") {
    const std::string dt = p.fp16 ? "f16" : "f32";
    const int OH = p.out_h(), OW = p.out_w();
    const size_t in_s  = (size_t)p.C * p.H * p.W;  // N stride
    const size_t in_s1 = (size_t)p.H * p.W;          // C stride
    const size_t flt_s = (size_t)(p.C/p.groups) * p.R * p.S;
    const size_t out_s  = (size_t)p.K * OH * OW;
    const size_t out_s1 = (size_t)OH * OW;

    // Kernel name reflects the epilogue chain (matches MIGraphX naming convention)
    const std::string fname = with_silu_add
        ? "mlir_convolution_broadcast_add_sigmoid_mul_add_sigmoid_mul_add"  // 6-arg: double silu+add
        : (with_skip
           ? "mlir_convolution_broadcast_add_sigmoid_mul_add"               // 5-arg: silu+add
           : "mlir_convolution_broadcast_add_sigmoid_mul");                  // 4-arg: silu only

    // !migraphx.shaped<N×C×H×W dt, strides>
    auto shaped = [&](std::initializer_list<int> dims,
                      std::initializer_list<size_t> strides) {
        std::ostringstream s;
        s << "<";
        bool first = true;
        for (int d : dims) { if (!first) s << "x"; s << d; first = false; }
        s << "x" << dt << ", ";
        first = true;
        for (size_t st : strides) { if (!first) s << "x"; s << st; first = false; }
        s << ">";
        return s.str();
    };

    const std::string in_sh  = shaped({p.N, p.C, p.H, p.W},  {in_s, in_s1, (size_t)p.W, 1});
    const std::string flt_sh = shaped({p.K, p.C/p.groups, p.R, p.S}, {flt_s, (size_t)p.R*p.S, (size_t)p.S, 1});
    const std::string out_sh = shaped({p.N, p.K, OH, OW}, {out_s, out_s1, (size_t)OW, 1});
    const std::string bias_sh = std::string("<") + std::to_string(p.K) + "x" + dt + ", 1>";

    std::ostringstream ir;
    ir << "module {\n";
    ir << "  func.func @" << fname << "(\n";
    ir << "      %arg0: !" << "migraphx.shaped" << in_sh  << ",\n";
    ir << "      %arg1: !" << "migraphx.shaped" << flt_sh << ",\n";
    ir << "      %arg2: !" << "migraphx.shaped" << bias_sh;
    if (with_skip)     ir << ",\n      %arg3: !migraphx.shaped" << out_sh;  // skip (shortcut)
    if (with_silu_add) ir << ",\n      %arg4: !migraphx.shaped" << out_sh;  // aux for second add
    ir << ")\n";
    ir << "      -> !migraphx.shaped" << out_sh << "\n";
    ir << "      attributes {arch = \"" << arch_triple(p.arch) << "\","
       << " kernel = \"mixr\", num_cu = " << p.num_cu << " : i64} {\n";

    ir << "    %0 = migraphx.convolution %arg0, %arg1 {"
       << "dilation = [" << p.dilation_h << ", " << p.dilation_w << "], "
       << "group = " << p.groups << " : i64, "
       << "padding = [" << p.pad_h << ", " << p.pad_h << ", "
                        << p.pad_w << ", " << p.pad_w << "], "
       << "padding_mode = 0 : i64, "
       << "stride = [" << p.stride_h << ", " << p.stride_w << "]";
    if (!perf_cfg.empty())
       ir << ", perf_config = \"" << perf_cfg << "\"";
    ir << "} : " << "!" << "migraphx.shaped" << in_sh << ", "
       << "!" << "migraphx.shaped" << flt_sh << " -> !" << "migraphx.shaped" << out_sh << "\n";

    // Broadcast bias along axis=1 (channel axis)
    std::string zeros(2 * out_sh.find("x" + dt) - out_sh.find("<") - 3, '0');
    ir << "    %1 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = ["
       << p.N << ", " << p.K << ", " << OH << ", " << OW << "]} : "
       << "!migraphx.shaped" << bias_sh << " -> !migraphx.shaped<"
       << p.N << "x" << p.K << "x" << OH << "x" << OW << "x" << dt << ", "
       << "0x1x0x0>\n";

    const std::string out_full = std::string("!migraphx.shaped") + out_sh;
    ir << "    %2 = migraphx.add %0, %1 : " << out_full << ", "
       << "!migraphx.shaped<" << p.N << "x" << p.K << "x" << OH << "x" << OW << "x" << dt << ", "
       << "0x1x0x0> -> " << out_full << "\n";
    ir << "    %3 = migraphx.sigmoid %2 : " << out_full << " -> " << out_full << "\n";
    ir << "    %4 = migraphx.mul %2, %3 : " << out_full << ", " << out_full << " -> " << out_full << "\n";

    if (with_skip) {
        // First add: silu(conv) + skip
        ir << "    %5 = migraphx.add %4, %arg3 : " << out_full << ", " << out_full << " -> " << out_full << "\n";
        if (with_silu_add) {
            // Second SiLU on the (silu+skip) result
            ir << "    %6 = migraphx.sigmoid %5 : " << out_full << " -> " << out_full << "\n";
            ir << "    %7 = migraphx.mul %5, %6 : " << out_full << ", " << out_full << " -> " << out_full << "\n";
            // Second add: silu(silu+skip) + aux
            ir << "    %8 = migraphx.add %7, %arg4 : " << out_full << ", " << out_full << " -> " << out_full << "\n";
            ir << "    return %8 : " << out_full << "\n";
        } else {
            ir << "    return %5 : " << out_full << "\n";
        }
    } else {
        ir << "    return %4 : " << out_full << "\n";
    }
    ir << "  }\n}\n";
    return ir.str();
}

// ── MIGraphX Conv+Bias+SkipAdd (no SiLU): mlir_convolution_broadcast_add_add ──
// Matches MIGraphX's 15-instance pattern: conv+bias (no SiLU) + skip-add.
// OV currently runs this as: conv+bias kernel + separate bias_add kernel (2 launches).
// This fuses them into ONE kernel: conv → broadcast(bias) → add → add(skip) → output.
// 4-arg kernel: (input, filter, bias, skip) → output
static std::string generate_fused_skip_ir(const ConvParams& p) {
    const std::string dt = p.fp16 ? "f16" : "f32";
    const int OH = p.out_h(), OW = p.out_w();
    const size_t in_s   = (size_t)p.C * p.H * p.W;
    const size_t in_s1  = (size_t)p.H * p.W;
    const size_t flt_s  = (size_t)(p.C/p.groups) * p.R * p.S;
    const size_t out_s  = (size_t)p.K * OH * OW;
    const size_t out_s1 = (size_t)OH * OW;

    auto sh = [&](std::initializer_list<int> dims, std::initializer_list<size_t> strides) {
        std::ostringstream s;
        s << "<";
        bool f = true;
        for (int d : dims) { if (!f) s << "x"; s << d; f = false; }
        s << "x" << dt << ", ";
        f = true;
        for (size_t st : strides) { if (!f) s << "x"; s << st; f = false; }
        s << ">";
        return s.str();
    };

    const std::string in_sh  = sh({p.N, p.C, p.H, p.W},  {in_s, in_s1, (size_t)p.W, 1});
    const std::string flt_sh = sh({p.K, p.C/p.groups, p.R, p.S}, {flt_s, (size_t)p.R*p.S, (size_t)p.S, 1});
    const std::string out_sh = sh({p.N, p.K, OH, OW}, {out_s, out_s1, (size_t)OW, 1});
    const std::string bias_sh = std::string("<") + std::to_string(p.K) + "x" + dt + ", 1>";
    const std::string out_full = "!migraphx.shaped" + out_sh;
    const std::string bias_bc_sh = "<" + std::to_string(p.N) + "x" + std::to_string(p.K)
        + "x" + std::to_string(OH) + "x" + std::to_string(OW) + "x" + dt + ", 0x1x0x0>";

    std::ostringstream ir;
    ir << "module {\n";
    ir << "  func.func @mlir_convolution_broadcast_add_add(\n";
    ir << "      %arg0: !migraphx.shaped" << in_sh  << ",\n";  // input
    ir << "      %arg1: !migraphx.shaped" << flt_sh << ",\n";  // filter
    ir << "      %arg2: !migraphx.shaped" << bias_sh << ",\n"; // bias
    ir << "      %arg3: " << out_full << ")\n";                  // skip
    ir << "      -> " << out_full << "\n";
    ir << "      attributes {arch = \"" << arch_triple(p.arch) << "\","
       << " kernel = \"mixr\", num_cu = " << p.num_cu << " : i64} {\n";

    ir << "    %0 = migraphx.convolution %arg0, %arg1 {"
       << "dilation = [" << p.dilation_h << ", " << p.dilation_w << "], "
       << "group = " << p.groups << " : i64, "
       << "padding = [" << p.pad_h << ", " << p.pad_h << ", " << p.pad_w << ", " << p.pad_w << "], "
       << "padding_mode = 0 : i64, stride = [" << p.stride_h << ", " << p.stride_w << "]"
       << "} : !migraphx.shaped" << in_sh << ", !migraphx.shaped" << flt_sh
       << " -> " << out_full << "\n";
    ir << "    %1 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = ["
       << p.N << ", " << p.K << ", " << OH << ", " << OW << "]} : "
       << "!migraphx.shaped" << bias_sh << " -> !migraphx.shaped" << bias_bc_sh << "\n";
    ir << "    %2 = migraphx.add %0, %1 : " << out_full << ", !migraphx.shaped" << bias_bc_sh
       << " -> " << out_full << "\n";
    ir << "    %3 = migraphx.add %2, %arg3 : " << out_full << ", " << out_full
       << " -> " << out_full << "\n";
    ir << "    return %3 : " << out_full << "\n";
    ir << "  }\n}\n";
    return ir.str();
}

// ── MIGraphX-style Conv+Bias+Reshape epilogue ─────────────────────────────
// Matches MIGraphX pattern: mlir_convolution_broadcast_add_reshape
// Used for Q/K/V projection convolutions in attention blocks:
//   conv 1×1 (N×C×H×W → N×K×H×W) → bias → reshape (N×K×H×W → N×K×(H*W))
//
// The reshape is a "free" logical operation — rocMLIR compiles the conv to write
// directly to the reshaped output layout, eliminating any intermediate copy.
// 3-arg kernel: (input, filter, bias) → reshaped_output
//
// reshape_dims: the target flat dimensions, e.g. {N, K, OH*OW}
static std::string generate_fused_reshape_ir(
        const ConvParams& p,
        const std::vector<int>& reshape_dims) {
    const std::string dt = p.fp16 ? "f16" : "f32";
    const int OH = p.out_h(), OW = p.out_w();
    const size_t in_s  = (size_t)p.C * p.H * p.W;
    const size_t in_s1 = (size_t)p.H * p.W;
    const size_t flt_s = (size_t)(p.C/p.groups) * p.R * p.S;
    const size_t out_s  = (size_t)p.K * OH * OW;
    const size_t out_s1 = (size_t)OH * OW;

    const std::string in_sh  = std::string("<") + std::to_string(p.N) + "x" + std::to_string(p.C)
        + "x" + std::to_string(p.H) + "x" + std::to_string(p.W) + "x" + dt + ", "
        + std::to_string(in_s) + "x" + std::to_string(in_s1) + "x" + std::to_string(p.W) + "x1>";
    const std::string flt_sh = std::string("<") + std::to_string(p.K) + "x" + std::to_string(p.C/p.groups)
        + "x" + std::to_string(p.R) + "x" + std::to_string(p.S) + "x" + dt + ", "
        + std::to_string(flt_s) + "x" + std::to_string(p.R*p.S) + "x" + std::to_string(p.S) + "x1>";
    const std::string out_sh = std::string("<") + std::to_string(p.N) + "x" + std::to_string(p.K)
        + "x" + std::to_string(OH) + "x" + std::to_string(OW) + "x" + dt + ", "
        + std::to_string(out_s) + "x" + std::to_string(out_s1) + "x" + std::to_string(OW) + "x1>";
    const std::string bias_sh = std::string("<") + std::to_string(p.K) + "x" + dt + ", 1>";

    // Build reshaped output shaped string (contiguous strides from outermost)
    std::string reshape_sh = "<";
    size_t total = 1;
    for (int d : reshape_dims) total *= d;
    // compute contiguous strides for reshape_dims
    std::vector<size_t> rsh_strides(reshape_dims.size());
    rsh_strides.back() = 1;
    for (int i = (int)reshape_dims.size() - 2; i >= 0; --i)
        rsh_strides[i] = rsh_strides[i+1] * reshape_dims[i+1];
    bool first_r = true;
    for (int d : reshape_dims) { if (!first_r) reshape_sh += "x"; reshape_sh += std::to_string(d); first_r = false; }
    reshape_sh += "x" + dt + ", ";
    first_r = true;
    for (size_t s : rsh_strides) { if (!first_r) reshape_sh += "x"; reshape_sh += std::to_string(s); first_r = false; }
    reshape_sh += ">";

    // Build dims string for migraphx.reshape
    std::string dims_str;
    for (size_t i = 0; i < reshape_dims.size(); ++i) {
        if (i) dims_str += ", ";
        dims_str += std::to_string(reshape_dims[i]);
    }

    std::ostringstream ir;
    ir << "module {\n";
    ir << "  func.func @mlir_convolution_broadcast_add_reshape(\n";
    ir << "      %arg0: !migraphx.shaped" << in_sh  << ",\n";
    ir << "      %arg1: !migraphx.shaped" << flt_sh << ",\n";
    ir << "      %arg2: !migraphx.shaped" << bias_sh << ")\n";
    ir << "      -> !migraphx.shaped" << reshape_sh << "\n";
    ir << "      attributes {arch = \"" << arch_triple(p.arch) << "\","
       << " kernel = \"mixr\", num_cu = " << p.num_cu << " : i64} {\n";

    const std::string out_full = "!migraphx.shaped" + out_sh;
    ir << "    %0 = migraphx.convolution %arg0, %arg1 {"
       << "dilation = [" << p.dilation_h << ", " << p.dilation_w << "], "
       << "group = " << p.groups << " : i64, "
       << "padding = [" << p.pad_h << ", " << p.pad_h << ", "
                        << p.pad_w << ", " << p.pad_w << "], "
       << "padding_mode = 0 : i64, "
       << "stride = [" << p.stride_h << ", " << p.stride_w << "]"
       << "} : !migraphx.shaped" << in_sh << ", !migraphx.shaped" << flt_sh
       << " -> " << out_full << "\n";

    // Broadcast bias along axis=1
    ir << "    %1 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = ["
       << p.N << ", " << p.K << ", " << OH << ", " << OW << "]} : "
       << "!migraphx.shaped" << bias_sh << " -> !migraphx.shaped<"
       << p.N << "x" << p.K << "x" << OH << "x" << OW << "x" << dt << ", 0x1x0x0>\n";
    ir << "    %2 = migraphx.add %0, %1 : " << out_full << ", "
       << "!migraphx.shaped<" << p.N << "x" << p.K << "x" << OH << "x" << OW << "x" << dt << ", 0x1x0x0>"
       << " -> " << out_full << "\n";

    // Reshape: flatten or reorder spatial dims
    ir << "    %3 = migraphx.reshape %2 {dims = [" << dims_str << "]} : "
       << out_full << " -> !migraphx.shaped" << reshape_sh << "\n";
    ir << "    return %3 : !migraphx.shaped" << reshape_sh << "\n";
    ir << "  }\n}\n";
    return ir.str();
}

// Compile a migraphx dialect MLIR module to HSACO
// Uses -kernel-pipeline=migraphx,highlevel,gpu,rocdl,binary
static CompiledConv compile_migraphx_ir(const std::string& mlir_ir,
                                          const std::string& arch,
                                          const std::string& driver_path) {
    const std::string driver = driver_path.empty() ? find_rocmlir_driver() : driver_path;
    // migraphx pipeline: migraphx ops → rock/linalg → gpu → rocdl → binary HSACO
    return compile_mlir_with_pipeline(mlir_ir, arch, driver,
        "migraphx,highlevel,gpu,rocdl,binary", "ov_migraphx_conv");
}

// ── Generate MIGraphX-format MLIR for Conv + Bias + SiLU ───────────────────
// (defined below, after generate_fused_epilogue_ir)
//
// Public entry: compile conv+bias+silu using migraphx dialect pipeline.
// Generates mlir_convolution_broadcast_add_sigmoid_mul kernel (same as MIGraphX).
// with_skip=true adds a residual skip-connection add after SiLU.
// NOTE: declared here after compile_migraphx_ir to satisfy forward reference order.
// Produces a complete rock.conv + linalg.generic SiLU module that rocmlir-driver
// can compile into a single fused HSACO kernel, eliminating the separate
// SwishOpImpl kernel call (measured 6.3ms / 12.8% of yolo26x GPU time).
//
// The MLIR structure matches MIGRAPHX_TRACE_MLIR=2 output for FP16:
//   module { func.func @mlir_conv_bias_silu(input, filter, bias, output) {
//     rock.conv(...)   // computes conv → tmp
//     linalg.generic   // computes tmp + bias → sigmoid(x)*x → output
//   }}
//
// Note: only SiLU (Swish) is fused here; ReLU/Sigmoid fall back to separate kernels.

static std::string generate_conv_bias_silu_ir(const ConvParams& p) {
    const std::string dt = p.fp16 ? "f16" : "f32";
    const int G   = p.groups;
    const int KpG = p.K / G;
    const int CpG = p.C / G;
    const int OH  = p.out_h();
    const int OW  = p.out_w();

    // Flat element counts
    const size_t in_elems   = (size_t)p.N * p.C * p.H * p.W;
    const size_t flt_elems  = (size_t)p.K * CpG * p.R * p.S;
    const size_t bias_elems = (size_t)p.K;
    const size_t out_elems  = (size_t)p.N * p.K * OH * OW;

    // Intermediate conv output (before bias+SiLU)
    const size_t conv_out_flat = (size_t)p.N * p.K * OH * OW;  // == out_elems

    // NCHW strides for transform maps (C outer, W inner)
    // filter: K x CpG x R x S  →  flat
    // input : N x C  x H x W   →  flat
    // output: N x K  x OH x OW →  flat

    std::ostringstream ir;
    ir << "module {\n";
    ir << "  func.func @mlir_conv_bias_silu(\n";
    ir << "      %arg0: memref<" << in_elems   << "x" << dt << ">,\n";  // input
    ir << "      %arg1: memref<" << flt_elems  << "x" << dt << ">,\n";  // filter
    ir << "      %arg2: memref<" << bias_elems << "x" << dt << ">,\n";  // bias
    ir << "      %arg3: memref<" << out_elems  << "x" << dt << ">)\n";  // output
    ir << "      attributes {arch = \"" << p.arch << ":sramecc+:xnack-\","
       << " kernel = \"mixr\", num_cu = " << p.num_cu << " : i64} {\n";

    if (p.fp16) {
        ir << "    %cst = arith.constant 1.0e+00 : f32\n";
    } else {
        ir << "    %cst = arith.constant 1.000000e+00 : f32\n";
    }

    // ── Filter transform: flat [K*CpG*R*S] → [G, KpG, CpG, R, S] ──────────
    ir << "    %0 = rock.transform %arg1 by"
       << " <affine_map<(d0, d1, d2, d3) -> ((d0 * " << CpG << " + d1) * " << p.R << " + d2) * " << p.S << " + d3)>"
       << " by [<Unmerge{" << KpG << ", " << CpG << ", " << p.R << ", " << p.S
       << "} [\"exp0\", \"exp1\", \"exp2\", \"exp3\"] at [0, 1, 2, 3] -> [\"dim0\"] at [0]>";
    if (G > 1) {
        ir << ", <AddDim{" << G << "} [\"g\"] at [0] -> [] at []>";
    }
    ir << "] bounds = [" << (G > 1 ? G : 1) << ", " << KpG << ", " << CpG
       << ", " << p.R << ", " << p.S << "] -> [" << flt_elems << "]>"
       << " : memref<" << flt_elems << "x" << dt << "> to memref<"
       << G << "x" << KpG << "x" << CpG << "x" << p.R << "x" << p.S << "x" << dt << ">\n";

    // ── Input transform: flat [N*C*H*W] → [N, G, CpG, H, W] ───────────────
    ir << "    %1 = rock.transform %arg0 by"
       << " <affine_map<(d0, d1, d2, d3) -> ((d1 * " << p.H << " + d2) * " << p.W << " + d3)>"
       << " by [<Unmerge{" << CpG << ", " << p.H << ", " << p.W
       << "} [\"exp1\", \"exp2\", \"exp3\"] at [1, 2, 3] -> [\"dim0\"] at [0]>"
       << ", <AddDim{1} [\"unit0\"] at [0] -> [] at []>] bounds = [1, " << G << ", " << CpG
       << ", " << p.H << ", " << p.W << "] -> [" << in_elems << "]>"
       << " : memref<" << in_elems << "x" << dt << "> to memref<1x" << G << "x" << CpG
       << "x" << p.H << "x" << p.W << "x" << dt << ">\n";

    // ── Intermediate alloc for conv output ────────────────────────────────
    ir << "    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x"
       << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << ">\n";

    // ── Output of alloc: [N, G, KpG, OH, OW] ─────────────────────────────
    ir << "    %4 = rock.transform %alloc by"
       << " <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * " << KpG << " + d2, d3, d4)>"
       << " by [<PassThrough [\"n\", \"h\", \"w\"] at [0, 3, 4] -> [\"n\", \"h\", \"w\"] at [0, 2, 3]>"
       << ", <Unmerge{1, " << KpG << "} [\"g\", \"k\"] at [1, 2] -> [\"k\"] at [1]>]"
       << " bounds = [1, 1, " << KpG << ", " << OH << ", " << OW << "] -> [1, " << G << ", "
       << KpG << ", " << OH << ", " << OW << "]>"
       << " : memref<1x" << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << "> to"
       << " memref<1x1x" << KpG << "x" << OH << "x" << OW << "x" << dt << ">\n";

    // ── rock.conv ──────────────────────────────────────────────────────────
    ir << "    rock.conv(%0, %1, %4) {"
       << "dilations = [" << p.dilation_h << " : index, " << p.dilation_w << " : index], "
       << "filter_layout = [\"g\", \"k\", \"c\", \"y\", \"x\"], "
       << "input_layout = [\"ni\", \"gi\", \"ci\", \"hi\", \"wi\"], "
       << "output_layout = [\"no\", \"go\", \"ko\", \"ho\", \"wo\"], "
       << "padding = [" << p.pad_h << " : index, " << p.pad_h << " : index, "
                         << p.pad_w << " : index, " << p.pad_w << " : index], "
       << "strides = [" << p.stride_h << " : index, " << p.stride_w << " : index]}"
       << " : memref<" << G << "x" << KpG << "x" << CpG << "x" << p.R << "x" << p.S << "x" << dt << ">,"
       << " memref<1x" << G << "x" << CpG << "x" << p.H << "x" << p.W << "x" << dt << ">,"
       << " memref<1x1x" << KpG << "x" << OH << "x" << OW << "x" << dt << ">\n";

    // ── Bias broadcast transform ───────────────────────────────────────────
    ir << "    %7 = rock.transform %arg2 by"
       << " <affine_map<(d0, d1, d2, d3) -> (d1)>"
       << " by [<Unmerge{" << p.K << "} [\"exp1\"] at [1] -> [\"dim0\"] at [0]>"
       << ", <AddDim{1} [\"unit0\"] at [0] -> [] at []>"
       << ", <AddDim{1} [\"unit2\"] at [2] -> [] at []>"
       << ", <AddDim{1} [\"unit3\"] at [3] -> [] at []>] bounds = [1, " << p.K << ", 1, 1] -> [" << p.K << "]>"
       << " : memref<" << p.K << "x" << dt << "> to memref<1x" << p.K << "x1x1x" << dt << ">\n";

    ir << "    %8 = rock.transform %7 by"
       << " <affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>"
       << " by [<PassThrough [\"dim0\"] at [0] -> [\"dim0\"] at [0]>"
       << ", <PassThrough [\"dim1\"] at [1] -> [\"dim1\"] at [1]>"
       << ", <Broadcast{1} [\"dim2\"] at [2] -> [\"dim2\"] at [2]>"
       << ", <Broadcast{1} [\"dim3\"] at [3] -> [\"dim3\"] at [3]>]"
       << " bounds = [1, " << p.K << ", " << OH << ", " << OW << "] -> [1, " << p.K << ", 1, 1]>"
       << " : memref<1x" << p.K << "x1x1x" << dt << "> to memref<1x" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";

    // ── Flatten alloc and bias for linalg.generic ─────────────────────────
    const size_t flat_KHW = (size_t)p.K * OH * OW;
    ir << "    %9 = rock.transform %alloc by"
       << " <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>"
       << " by [<Merge{1, " << p.K << "} [\"dim0\"] at [0] -> [\"col0\", \"col1\"] at [0, 1]>"
       << ", <PassThrough [\"dim1\"] at [1] -> [\"dim1\"] at [2]>"
       << ", <PassThrough [\"dim2\"] at [2] -> [\"dim2\"] at [3]>]"
       << " bounds = [" << p.K << ", " << OH << ", " << OW << "] -> [1, " << p.K << ", " << OH << ", " << OW << "]>"
       << " : memref<1x" << G << "x" << KpG << "x" << OH << "x" << OW << "x" << dt << "> to"
       << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";

    ir << "    %10 = rock.transform %8 by"
       << " <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>"
       << " by [<Merge{1, " << p.K << "} [\"dim0\"] at [0] -> [\"col0\", \"col1\"] at [0, 1]>"
       << ", <PassThrough [\"dim1\"] at [1] -> [\"dim1\"] at [2]>"
       << ", <PassThrough [\"dim2\"] at [2] -> [\"dim2\"] at [3]>]"
       << " bounds = [" << p.K << ", " << OH << ", " << OW << "] -> [1, " << p.K << ", " << OH << ", " << OW << "]>"
       << " : memref<1x" << p.K << "x" << OH << "x" << OW << "x" << dt << "> to"
       << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";

    // ── linalg.generic: x + bias → SiLU(x+bias) ──────────────────────────
    ir << "    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<"
       << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";

    ir << "    linalg.generic {"
       << "indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,"
       << " affine_map<(d0, d1, d2) -> (d0, d1, d2)>,"
       << " affine_map<(d0, d1, d2) -> (d0, d1, d2)>],"
       << " iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}"
       << " ins(%9, %10 : memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">,"
       << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">)"
       << " outs(%alloc_0 : memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">) {\n";

    ir << "    ^bb0(%in: " << dt << ", %in_1: " << dt << ", %out: " << dt << "):\n";
    if (p.fp16) {
        // For f16, extend to f32 for arithmetic, then truncate back
        ir << "      %e0 = arith.extf %in : f16 to f32\n";
        ir << "      %e1 = arith.extf %in_1 : f16 to f32\n";
        ir << "      %sum = arith.addf %e0, %e1 : f32\n";
        ir << "      %neg = arith.negf %sum : f32\n";
        ir << "      %exp = math.exp %neg : f32\n";
        ir << "      %one_p_exp = arith.addf %exp, %cst : f32\n";
        ir << "      %sig = arith.divf %cst, %one_p_exp : f32\n";
        ir << "      %silu = arith.mulf %sum, %sig : f32\n";
        ir << "      %res = arith.truncf %silu : f32 to f16\n";
        ir << "      linalg.yield %res : f16\n";
    } else {
        ir << "      %sum = arith.addf %in, %in_1 : f32\n";
        ir << "      %neg = arith.negf %sum : f32\n";
        ir << "      %exp = math.exp %neg : f32\n";
        ir << "      %one_p_exp = arith.addf %exp, %cst : f32\n";
        ir << "      %sig = arith.divf %cst, %one_p_exp : f32\n";
        ir << "      %silu = arith.mulf %sum, %sig : f32\n";
        ir << "      linalg.yield %silu : f32\n";
    }
    ir << "    }\n";

    // ── Write result to output ─────────────────────────────────────────────
    ir << "    %11 = rock.transform %alloc_0 by"
       << " <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>"
       << " by [<Unmerge{" << p.K << "} [\"exp1\"] at [1] -> [\"dim0\"] at [0]>"
       << ", <PassThrough [\"dim1\"] at [2] -> [\"dim1\"] at [1]>"
       << ", <PassThrough [\"dim2\"] at [3] -> [\"dim2\"] at [2]>"
       << ", <AddDim{1} [\"unit0\"] at [0] -> [] at []>] bounds = [1, " << p.K
       << ", " << OH << ", " << OW << "] -> [" << flat_KHW << "]>"
       << " : memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << "> to"
       << " memref<1x" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";

    ir << "    %12 = rock.transform %11 by"
       << " <affine_map<(d0) -> (0, d0 floordiv " << (OH * OW) << ", (d0 mod " << (OH * OW)
       << ") floordiv " << OW << ", d0 mod " << OW << ")>"
       << " by [<Merge{1, " << p.K << ", " << OH << ", " << OW
       << "} [\"dim0\"] at [0] -> [\"col0\", \"col1\", \"col2\", \"col3\"] at [0, 1, 2, 3]>]"
       << " bounds = [" << out_elems << "] -> [1, " << p.K << ", " << OH << ", " << OW << "]>"
       << " : memref<1x" << p.K << "x" << OH << "x" << OW << "x" << dt << "> to"
       << " memref<" << out_elems << "x" << dt << ">\n";

    ir << "    memref.copy %12, %arg3 : memref<" << out_elems << "x" << dt << "> to"
       << " memref<" << out_elems << "x" << dt << ">\n";
    ir << "    return\n";
    ir << "  }\n";
    ir << "}\n";
    return ir.str();
}

// ── Patch rocmlir-gen IR to add bias + optional SiLU epilogue ────────────────
// rocmlir-gen produces: func @rock_conv...(%arg0:filter, %arg1:input, %arg2:output)
// We patch it to:       func @mlir_conv_bias_silu(..., %arg2:bias, %arg3:output)
// and append a linalg.generic that computes: output[i] = silu(conv_result[i] + bias[k])
// This uses rocmlir-gen's correct affine maps (no manual generation needed).
// When p.C_full > 0, the input arg receives the full (pre-split) tensor and a Slice
// transform is inserted to extract channels [p.c_start, p.c_start + p.C).
static std::string patch_ir_bias_silu(const std::string& base_ir, const ConvParams& p,
                                       bool with_silu) {
    const std::string dt = p.fp16 ? "f16" : "f32";
    const int OH = p.out_h(), OW = p.out_w();
    const size_t out_flat = (size_t)p.N * p.K * OH * OW;
    const size_t bias_flat = (size_t)p.K;
    // When slice is active, input arg holds C_full channels; convolution uses C channels
    const bool has_slice = (p.C_full > 0 && p.C_full != p.C);
    const int C_full = has_slice ? p.C_full : p.C;
    const size_t in_flat_full = (size_t)p.N * C_full * p.H * p.W;
    const size_t in_flat_used = (size_t)p.N * p.C * p.H * p.W;

    std::string ir = base_ir;

    // 1. Rename function
    {
        const std::string old_fn = "@rock_conv_gkc01_ngc01_ngk01_0";
        std::string new_fn = with_silu ? (has_slice ? "@mlir_slice_conv_bias_silu" : "@mlir_conv_bias_silu")
                                        : (has_slice ? "@mlir_slice_conv_bias" : "@mlir_conv_bias");
        auto pos = ir.find(old_fn);
        if (pos == std::string::npos) return "";  // unexpected format
        ir.replace(pos, old_fn.size(), new_fn);
    }

    // 2. Insert bias arg before output arg in function signature.
    // Signature: ...(%arg0: memref<Axdt>, %arg1: memref<Bxdt>, %arg2: memref<Cxdt>)
    // We find "%arg2: memref<{out_flat}x{dt}>)" and replace with
    //          "%arg2: memref<{bias_flat}x{dt}>, %arg3: memref<{out_flat}x{dt}>)"
    {
        const std::string old_arg2 = "%arg2: memref<" + std::to_string(out_flat) + "x" + dt + ">)";
        const std::string new_args = "%arg2: memref<" + std::to_string(bias_flat) + "x" + dt + ">, "
                                   + "%arg3: memref<" + std::to_string(out_flat) + "x" + dt + ">)";
        auto pos = ir.find(old_arg2);
        if (pos == std::string::npos) return "";
        ir.replace(pos, old_arg2.size(), new_args);
    }

    // 3. Replace the output transform line:
    //    "%N = rock.transform %arg2 by <...> : memref<out_flatxdt> to memref<...>"
    // with:
    //    "%conv_out_alloc = memref.alloc() : memref<out_flatxdt>
    //     %N = rock.transform %conv_out_alloc by <...> : memref<out_flatxdt> to memref<...>"
    {
        // Find the full line: "    %N = rock.transform %arg2 by"
        const std::string prefix = "rock.transform %arg2 by";
        auto pos = ir.find(prefix);
        if (pos == std::string::npos) return "";

        // Find start of the assignment (%N =) by searching backward for "    %"
        size_t line_start = pos;
        while (line_start > 0 && ir[line_start] != '\n') --line_start;
        ++line_start;  // skip the newline

        // Insert alloc before this line
        const std::string alloc_line = "    %conv_out_alloc = memref.alloc() {alignment = 64 : i64} : memref<"
                                      + std::to_string(out_flat) + "x" + dt + ">\n";
        ir.insert(line_start, alloc_line);

        // Now replace %arg2 in the transform (appears after the inserted alloc line)
        // Find the exact "%arg2" in "rock.transform %arg2 by"
        const std::string arg2_marker = "%arg2 by";
        pos = ir.find(arg2_marker, line_start);  // find %arg2 within the transform line
        if (pos == std::string::npos) return "";
        const std::string new_ref = "%conv_out_alloc by";
        ir.replace(pos, arg2_marker.size(), new_ref);
    }

    // 4. Replace %arg2 reference in rock.conv args (appears as the 3rd operand)
    //    rock.conv(%0, %1, %2) → stays as %2 which is now transform of %conv_out_alloc

    // 5. Before "return", insert bias broadcast + linalg.generic epilogue
    {
        const std::string ret = "    return\n  }";
        auto pos = ir.rfind(ret);
        if (pos == std::string::npos) return "";

        std::ostringstream epi;

        // 3D linalg.generic epilogue via memref.expand_shape — NO integer division.
        // Reshapes flat [N*K*OH*OW] alloc to 3D [K, OH, OW] using expand_shape.
        // Bias is a 1D [K] memref; affine_map<(d0,d1,d2)->(d0)> handles broadcast.
        // This matches what MIGraphX generates, but using expand_shape instead of rock.transform.
        const size_t ohw = (size_t)OH * OW;
        epi << "    // ── bias + " << (with_silu ? "SiLU" : "add") << " epilogue (3D, no int div) ────\n";
        if (with_silu) {
            epi << "    %cst = arith.constant 1.0e+00 : f32\n";
        }
        // Reshape flat conv output [K*OH*OW] → 3D [K, OH, OW]
        epi << "    %conv_3d = memref.expand_shape %conv_out_alloc [[0, 1, 2]]"
            << " output_shape [" << p.K << ", " << OH << ", " << OW << "]"
            << " : memref<" << out_flat << "x" << dt << "> into"
            << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";
        // Reshape flat output arg3 [K*OH*OW] → 3D [K, OH, OW]
        epi << "    %out_3d = memref.expand_shape %arg3 [[0, 1, 2]]"
            << " output_shape [" << p.K << ", " << OH << ", " << OW << "]"
            << " : memref<" << out_flat << "x" << dt << "> into"
            << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";

        // 3D parallel linalg.generic: bias broadcast via affine_map<(d0,d1,d2)->(d0)>
        epi << "    linalg.generic {\n"
            << "      indexing_maps = [\n"
            << "        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,\n"  // conv_3d
            << "        affine_map<(d0, d1, d2) -> (d0)>,\n"            // bias (broadcast H,W)
            << "        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],\n"  // out_3d
            << "      iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
            << "      ins(%conv_3d, %arg2 : memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">,"
            << " memref<" << p.K << "x" << dt << ">)\n"
            << "      outs(%out_3d : memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">) {\n";
        epi << "    ^bb0(%in: " << dt << ", %bias_v: " << dt << ", %out: " << dt << "):\n";
        if (p.fp16) {
            epi << "      %e0 = arith.extf %in : f16 to f32\n";
            epi << "      %eb = arith.extf %bias_v : f16 to f32\n";
            epi << "      %s = arith.addf %e0, %eb : f32\n";
            if (with_silu) {
                epi << "      %neg = arith.negf %s : f32\n";
                epi << "      %exp = math.exp %neg : f32\n";
                epi << "      %denom = arith.addf %exp, %cst : f32\n";
                epi << "      %sig = arith.divf %cst, %denom : f32\n";
                epi << "      %silu_f = arith.mulf %s, %sig : f32\n";
                epi << "      %res = arith.truncf %silu_f : f32 to f16\n";
            } else {
                epi << "      %res = arith.truncf %s : f32 to f16\n";
            }
        } else {
            epi << "      %s = arith.addf %in, %bias_v : f32\n";
            if (with_silu) {
                epi << "      %neg = arith.negf %s : f32\n";
                epi << "      %exp = math.exp %neg : f32\n";
                epi << "      %denom = arith.addf %exp, %cst : f32\n";
                epi << "      %sig = arith.divf %cst, %denom : f32\n";
                epi << "      %res = arith.mulf %s, %sig : f32\n";
            } else {
                epi << "      %res = %s\n";
            }
        }
        epi << "      linalg.yield %res : " << dt << "\n";
        epi << "    }\n";

        ir.insert(pos, epi.str());
    }

    return ir;
}

// ── Patch rocmlir-gen IR to add bias + SiLU + skip-Add epilogue ──────────────
// Extends patch_ir_bias_silu to produce a 5-arg kernel:
//   @mlir_conv_bias_silu_add(input, filter, bias, skip_input, output)
// The epilogue computes: output[i] = SiLU(conv[i] + bias[k]) + skip_input[i]
// skip_input has the same flat shape as the output (N*K*OH*OW).
static std::string patch_ir_bias_silu_add(const std::string& base_ir, const ConvParams& p) {
    const std::string dt = p.fp16 ? "f16" : "f32";
    const int OH = p.out_h(), OW = p.out_w();
    const size_t out_flat = (size_t)p.N * p.K * OH * OW;
    const size_t bias_flat = (size_t)p.K;

    std::string ir = base_ir;

    // 1. Rename function to 5-arg variant
    {
        const std::string old_fn = "@rock_conv_gkc01_ngc01_ngk01_0";
        const std::string new_fn = "@mlir_conv_bias_silu_add";
        auto pos = ir.find(old_fn);
        if (pos == std::string::npos) return "";
        ir.replace(pos, old_fn.size(), new_fn);
    }

    // 2. Replace 3-arg signature with 5-arg: (input, filter, bias, skip_input, output)
    // Old: %arg2: memref<{out_flat}xdt>)
    // New: %arg2: memref<{bias_flat}xdt>, %arg3: memref<{out_flat}xdt>, %arg4: memref<{out_flat}xdt>)
    {
        const std::string old_arg2 = "%arg2: memref<" + std::to_string(out_flat) + "x" + dt + ">)";
        const std::string new_args = "%arg2: memref<" + std::to_string(bias_flat) + "x" + dt + ">, "
                                   + "%arg3: memref<" + std::to_string(out_flat) + "x" + dt + ">, "
                                   + "%arg4: memref<" + std::to_string(out_flat) + "x" + dt + ">)";
        auto pos = ir.find(old_arg2);
        if (pos == std::string::npos) return "";
        ir.replace(pos, old_arg2.size(), new_args);
    }

    // 3. Redirect rock.conv output to an intermediate alloc (same as bias-silu variant)
    {
        const std::string prefix = "rock.transform %arg2 by";
        auto pos = ir.find(prefix);
        if (pos == std::string::npos) return "";

        size_t line_start = pos;
        while (line_start > 0 && ir[line_start] != '\n') --line_start;
        ++line_start;

        const std::string alloc_line = "    %conv_out_alloc = memref.alloc() {alignment = 64 : i64} : memref<"
                                      + std::to_string(out_flat) + "x" + dt + ">\n";
        ir.insert(line_start, alloc_line);

        const std::string arg2_marker = "%arg2 by";
        pos = ir.find(arg2_marker, line_start);
        if (pos == std::string::npos) return "";
        ir.replace(pos, arg2_marker.size(), "%conv_out_alloc by");
    }

    // 4. Before "return", insert bias+SiLU+skip-Add linalg.generic epilogue
    {
        const std::string ret = "    return\n  }";
        auto pos = ir.rfind(ret);
        if (pos == std::string::npos) return "";

        std::ostringstream epi;
        epi << "    %cst = arith.constant 1.0e+00 : f32\n";

        // Reshape flat tensors to 3D [K, OH, OW]
        epi << "    %conv_3d = memref.expand_shape %conv_out_alloc [[0, 1, 2]]"
            << " output_shape [" << p.K << ", " << OH << ", " << OW << "]"
            << " : memref<" << out_flat << "x" << dt << "> into"
            << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";
        // skip_input is arg3
        epi << "    %skip_3d = memref.expand_shape %arg3 [[0, 1, 2]]"
            << " output_shape [" << p.K << ", " << OH << ", " << OW << "]"
            << " : memref<" << out_flat << "x" << dt << "> into"
            << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";
        // output is arg4
        epi << "    %out_3d = memref.expand_shape %arg4 [[0, 1, 2]]"
            << " output_shape [" << p.K << ", " << OH << ", " << OW << "]"
            << " : memref<" << out_flat << "x" << dt << "> into"
            << " memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">\n";

        // linalg.generic: 4 inputs (conv, bias, skip), 1 output
        epi << "    linalg.generic {\n"
            << "      indexing_maps = [\n"
            << "        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,\n"   // conv_3d
            << "        affine_map<(d0, d1, d2) -> (d0)>,\n"             // bias (broadcast)
            << "        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,\n"   // skip_3d
            << "        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],\n"  // out_3d
            << "      iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}\n"
            << "      ins(%conv_3d, %arg2, %skip_3d : "
            << "memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">, "
            << "memref<" << p.K << "x" << dt << ">, "
            << "memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">)\n"
            << "      outs(%out_3d : memref<" << p.K << "x" << OH << "x" << OW << "x" << dt << ">) {\n";
        epi << "    ^bb0(%in: " << dt << ", %bias_v: " << dt
            << ", %skip_v: " << dt << ", %out: " << dt << "):\n";
        if (p.fp16) {
            epi << "      %e0 = arith.extf %in : f16 to f32\n";
            epi << "      %eb = arith.extf %bias_v : f16 to f32\n";
            epi << "      %es = arith.extf %skip_v : f16 to f32\n";
            epi << "      %s = arith.addf %e0, %eb : f32\n";
            epi << "      %neg = arith.negf %s : f32\n";
            epi << "      %exp = math.exp %neg : f32\n";
            epi << "      %denom = arith.addf %exp, %cst : f32\n";
            epi << "      %sig = arith.divf %cst, %denom : f32\n";
            epi << "      %silu_f = arith.mulf %s, %sig : f32\n";
            epi << "      %sum = arith.addf %silu_f, %es : f32\n";
            epi << "      %res = arith.truncf %sum : f32 to f16\n";
        } else {
            epi << "      %s = arith.addf %in, %bias_v : f32\n";
            epi << "      %neg = arith.negf %s : f32\n";
            epi << "      %exp = math.exp %neg : f32\n";
            epi << "      %denom = arith.addf %exp, %cst : f32\n";
            epi << "      %sig = arith.divf %cst, %denom : f32\n";
            epi << "      %silu_f = arith.mulf %s, %sig : f32\n";
            epi << "      %res = arith.addf %silu_f, %skip_v : f32\n";
        }
        epi << "      linalg.yield %res : " << dt << "\n";
        epi << "    }\n";

        ir.insert(pos, epi.str());
    }

    return ir;
}

// (external tuning path removed — all kernels compiled via rocmlir-gen + rocmlir-driver)

// (removed: ONNX-based conv model generator — no longer needed)

// Placeholder to avoid unused-variable warnings at former call sites.
static std::string _removed_onnx_gen(const ConvParams&, const std::string&) { return ""; }

// Stub: formerly generated ONNX for external compilation. Now unused.
static std::string generate_conv_onnx(const ConvParams& p, const std::string& tmp_path) {
    (void)p; (void)tmp_path;
    // Use Python/onnx to create the model
    const std::string py_script = R"(
import sys, struct, os
N,C,H,W,K,R,S,pad_h,pad_w,stride_h,stride_w,dil_h,dil_w,G,fp16 = [int(x) for x in sys.argv[1:16]]
dt = 10 if fp16 else 1  # ONNX TensorProto.FLOAT16=10, FLOAT=1
out = sys.argv[16]

def make_tensor(name, dt, shape):
    dims = b''
    for d in shape:
        dims += struct.pack('<q', d)
    return b'\x0a' + bytes([len(name)]) + name.encode() + \
           b'\x20' + struct.pack('<i', dt) + \
           b'\x28' * len(shape) + dims

# Minimal ONNX binary - use onnx Python package
import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

dt_np = np.float16 if fp16 else np.float32
dt_onnx = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT

X = helper.make_tensor_value_info('X', dt_onnx, [N, C, H, W])
Y = helper.make_tensor_value_info('Y', dt_onnx, None)

W_val = np.ones([K, C//G, R, S], dtype=dt_np) * dt_np(0.01)
B_val = np.ones([K], dtype=dt_np) * dt_np(0.1)
W_init = numpy_helper.from_array(W_val, 'W')
B_init = numpy_helper.from_array(B_val, 'B')

conv = helper.make_node('Conv', ['X','W','B'], ['conv_out'],
    kernel_shape=[R,S], pads=[pad_h,pad_w,pad_h,pad_w],
    strides=[stride_h,stride_w], dilations=[dil_h,dil_w], group=G)
sig = helper.make_node('Sigmoid', ['conv_out'], ['sig'])
mul = helper.make_node('Mul', ['conv_out','sig'], ['Y'])

graph = helper.make_graph([conv,sig,mul], 'g', [X], [Y], [W_init, B_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
model.ir_version = 7
onnx.save(model, out)
print('OK')
)";
    const std::string script_file = tmp_path + ".py";
    const std::string onnx_file = tmp_path + ".onnx";
    {
        std::ofstream f(script_file);
        f << py_script;
    }

    // Find python with onnx available
    const char* py_env = std::getenv("ROCMLIR_PYTHON");
    const std::string python = py_env ? py_env : "python3";
    const std::string cmd = python + " " + script_file + " "
        + std::to_string(p.N) + " " + std::to_string(p.C) + " "
        + std::to_string(p.H) + " " + std::to_string(p.W) + " "
        + std::to_string(p.K) + " " + std::to_string(p.R) + " "
        + std::to_string(p.S) + " " + std::to_string(p.pad_h) + " "
        + std::to_string(p.pad_w) + " " + std::to_string(p.stride_h) + " "
        + std::to_string(p.stride_w) + " " + std::to_string(p.dilation_h) + " "
        + std::to_string(p.dilation_w) + " " + std::to_string(p.groups) + " "
        + std::to_string(p.fp16 ? 1 : 0) + " " + onnx_file + " 2>/dev/null";

    int exit_code = 0;
    const std::string result = run_cmd(cmd, exit_code);
    std::remove(script_file.c_str());

    if (exit_code != 0 || result.find("OK") == std::string::npos) return "";
    return onnx_file;
}

// (external conv compilation path removed)

CompiledConv compile_fused_conv_bias(const ConvParams& p, const std::string& drv) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;

    auto& kc = HsacoKernelCache::instance();
    const size_t cache_key = p.hash() ^ static_cast<size_t>(0xB1A5B1A5B1A50001ULL);
    { CompiledConv cached; if (kc.load(cache_key, p.arch, cached)) {
        std::cerr << "[BiasIR-cache] loaded bias kernel=" << cached.kernel_name << "\n";
        return cached; } }

    const std::string gen    = find_rocmlir_gen(driver);
    const std::string perf_cfg = get_tuned_perf_config(p, gen, driver);
    const std::string gen_cmd  = rocmlir_gen_cmd(p, gen, perf_cfg);

    int exit_code = 0;
    const std::string base_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0)
        OPENVINO_THROW("rocmlir-gen failed (exit ", exit_code, "): ", base_ir.substr(0, 300));

    // Try Conv+Bias fusion (eliminates separate bias_add_nchw kernel — 14% of GPU time)
    const std::string fused_ir = patch_ir_bias_silu(base_ir, p, false);
    static bool dumped = false;
    if (fused_ir.empty()) {
        std::cerr << "[BiasIR] patch failed (empty IR), falling back\n";
    } else {
        // Dump first patched IR for inspection
        if (!dumped) {
            dumped = true;
            FILE* f = fopen("/tmp/ov_bias_fused.mlir", "w");
            if (f) { fwrite(fused_ir.data(), 1, fused_ir.size(), f); fclose(f); }
        }
        try {
            auto result = compile_mlir_ir(fused_ir, p.arch, driver);
            result.bias_fused = true;
            std::cerr << "[BiasIR] bias fusion OK kernel=" << result.kernel_name << "\n";
            kc.save(cache_key, p.arch, result);
            return result;
        } catch (const std::exception& e) {
            std::cerr << "[BiasIR] compile failed: " << e.what() << "\n";
        }
    }
    // Fallback: plain conv kernel (caller applies bias separately)
    auto result = compile_mlir_ir(base_ir, p.arch, driver);
    kc.save(cache_key, p.arch, result);
    return result;
}

CompiledConv compile_fused_conv_bias_act(const ConvParams& p, Activation act, const std::string& drv) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;
    const bool with_silu = (act == Activation::Sigmoid);

    auto& kc = HsacoKernelCache::instance();
    // Use separate magic per silu/non-silu to avoid collision
    const size_t cache_key = p.hash() ^ (with_silu ? static_cast<size_t>(0xB1A5B1A5B1A50002ULL)
                                                    : static_cast<size_t>(0xB1A5B1A5B1A50008ULL));
    { CompiledConv cached; if (kc.load(cache_key, p.arch, cached)) {
        std::cerr << "[BiasIR-cache] loaded bias_act silu=" << with_silu
                  << " kernel=" << cached.kernel_name << "\n";
        return cached; } }

    const std::string gen    = find_rocmlir_gen(driver);
    const std::string perf_cfg = get_tuned_perf_config(p, gen, driver);
    const std::string gen_cmd  = rocmlir_gen_cmd(p, gen, perf_cfg);

    int exit_code = 0;
    const std::string base_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0)
        OPENVINO_THROW("rocmlir-gen failed for fused_bias_act (exit ", exit_code, ")");

    // Conv+Bias+SiLU fusion (eliminates bias_add + SwishOpImpl — 21% of GPU time)
    const std::string fused_ir = patch_ir_bias_silu(base_ir, p, with_silu);
    if (!fused_ir.empty()) {
        try {
            auto result = compile_mlir_ir(fused_ir, p.arch, driver);
            result.bias_fused = true;
            result.silu_fused = with_silu;
            std::cerr << "[BiasIR] fused_bias_act OK kernel=" << result.kernel_name
                      << " silu=" << with_silu << "\n";
            kc.save(cache_key, p.arch, result);
            return result;
        } catch (const std::exception& e) {
            std::cerr << "[BiasIR] fused_bias_act FAIL silu=" << with_silu
                      << " : " << e.what() << "\n";
        }
    }
    auto result = compile_mlir_ir(base_ir, p.arch, driver);
    kc.save(cache_key, p.arch, result);
    return result;
}

CompiledConv compile_fused_conv_bias_silu_add(const ConvParams& p, const std::string& drv) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;

    auto& kc = HsacoKernelCache::instance();
    const size_t cache_key = p.hash() ^ static_cast<size_t>(0xB1A5B1A5B1A50003ULL);
    { CompiledConv cached; if (kc.load(cache_key, p.arch, cached)) {
        std::cerr << "[BiasIR-cache] loaded bias_silu_add kernel=" << cached.kernel_name << "\n";
        return cached; } }

    const std::string gen    = find_rocmlir_gen(driver);
    const std::string perf_cfg = get_tuned_perf_config(p, gen, driver);
    const std::string gen_cmd  = rocmlir_gen_cmd(p, gen, perf_cfg);

    int exit_code = 0;
    const std::string base_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0)
        OPENVINO_THROW("rocmlir-gen failed for fused_bias_silu_add (exit ", exit_code, ")");

    const std::string fused_ir = patch_ir_bias_silu_add(base_ir, p);
    if (!fused_ir.empty()) {
        try {
            auto result = compile_mlir_ir(fused_ir, p.arch, driver);
            result.bias_fused    = true;
            result.silu_fused    = true;
            result.skip_add_fused = true;
            std::cerr << "[BiasIR] fused_bias_silu_add OK kernel=" << result.kernel_name << "\n";
            kc.save(cache_key, p.arch, result);
            return result;
        } catch (const std::exception& e) {
            std::cerr << "[BiasIR] fused_bias_silu_add FAIL: " << e.what() << "\n";
        }
    }
    // Fallback: plain Conv+Bias+SiLU (caller handles skip-add separately)
    auto result = compile_fused_conv_bias_act(p, Activation::Sigmoid, driver);
    // Note: fallback result is already cached by compile_fused_conv_bias_act
    return result;
}

// ── Slice + Conv+Bias+SiLU compilation via MIGraphX ──────────────────────────
// Generates a Slice → Conv+Sigmoid+Mul ONNX (so MIGraphX produces the
// mlir_slice_convolution_broadcast_add_sigmoid_mul kernel), then extracts
// and compiles the MLIR. arg0 receives the FULL (pre-slice) input tensor.
// p.C_full = full input channels, p.c_start = slice start, p.C = slice size.

static std::string generate_slice_conv_onnx(const ConvParams& p, const std::string& tmp_path) {
    const int C_full = (p.C_full > 0) ? p.C_full : p.C;
    const std::string py_script = R"(
import sys, onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
N,C_full,C,H,W,K,R,S,pad_h,pad_w,sh,sw,dh,dw,G,fp16,c_start = [int(x) for x in sys.argv[1:18]]
out = sys.argv[18]
dt_np = np.float16 if fp16 else np.float32
dt_onnx = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
X = helper.make_tensor_value_info('X', dt_onnx, [N, C_full, H, W])
Y = helper.make_tensor_value_info('Y', dt_onnx, None)
W_val = np.ones([K, C//G, R, S], dtype=dt_np) * dt_np(0.01)
B_val = np.ones([K], dtype=dt_np) * dt_np(0.1)
W_init = numpy_helper.from_array(W_val, 'W')
B_init = numpy_helper.from_array(B_val, 'B')
starts_init = numpy_helper.from_array(np.array([c_start], dtype=np.int64), 'starts')
ends_init = numpy_helper.from_array(np.array([c_start + C], dtype=np.int64), 'ends')
axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), 'axes')
nodes = [
    helper.make_node('Slice', ['X','starts','ends','axes'], ['X_slice']),
    helper.make_node('Conv', ['X_slice','W','B'], ['conv_out'],
        kernel_shape=[R,S], pads=[pad_h,pad_w,pad_h,pad_w],
        strides=[sh,sw], dilations=[dh,dw], group=G),
    helper.make_node('Sigmoid', ['conv_out'], ['sig']),
    helper.make_node('Mul', ['conv_out','sig'], ['Y']),
]
graph = helper.make_graph(nodes, 'g', [X], [Y],
    [W_init, B_init, starts_init, ends_init, axes_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 7
onnx.save(model, out)
print('OK')
)";
    const std::string script_file = tmp_path + "_slice.py";
    const std::string onnx_file = tmp_path + "_slice.onnx";
    {
        std::ofstream f(script_file);
        f << py_script;
    }
    const char* py_env = std::getenv("ROCMLIR_PYTHON");
    const std::string python = py_env ? py_env : "python3";
    const std::string cmd = python + " " + script_file + " "
        + std::to_string(p.N) + " " + std::to_string(C_full) + " " + std::to_string(p.C) + " "
        + std::to_string(p.H) + " " + std::to_string(p.W) + " "
        + std::to_string(p.K) + " " + std::to_string(p.R) + " " + std::to_string(p.S) + " "
        + std::to_string(p.pad_h) + " " + std::to_string(p.pad_w) + " "
        + std::to_string(p.stride_h) + " " + std::to_string(p.stride_w) + " "
        + std::to_string(p.dilation_h) + " " + std::to_string(p.dilation_w) + " "
        + std::to_string(p.groups) + " " + std::to_string(p.fp16 ? 1 : 0) + " "
        + std::to_string(p.c_start) + " " + onnx_file + " 2>/dev/null";
    int exit_code = 0;
    const std::string result = run_cmd(cmd, exit_code);
    std::remove(script_file.c_str());
    if (exit_code != 0 || result.find("OK") == std::string::npos) return "";
    return onnx_file;
}

CompiledConv compile_slice_conv_bias_silu(const ConvParams& p, const std::string& drv) {
    // Compile a Conv+Bias+SiLU kernel for a channel-sliced input.
    // The input is the full [N, C_full, H, W] tensor; the conv reads C channels
    // starting at channel c_start. At execute time, the caller applies the
    // c_start * H * W * elem_size byte offset to the input pointer.
    //
    // Implementation: compile a standard Conv+Bias+SiLU kernel for the sliced
    // shape [N, C, H, W] using rocmlir-gen + rocmlir-driver. The kernel itself
    // does not need to know about the slice — only the execute-time pointer matters.

    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;

    // Check persistent HSACO cache first
    auto& kern_cache = HsacoKernelCache::instance();
    const size_t cache_key = p.hash() ^ static_cast<size_t>(0x5C1CE5E1CE5E1CEULL);
    {
        CompiledConv cached;
        if (kern_cache.load(cache_key, p.arch, cached)) {
            std::cerr << "[SliceConv-cache] loaded " << cached.kernel_name
                      << " grid=" << cached.grid_x << " block=" << cached.block_x << "\n";
            return cached;
        }
    }

    // Build sliced params: treat the conv as a regular [N,C,H,W] → [N,K,OH,OW] conv.
    // The C_full / c_start fields are stored in the cache key but not used by
    // the kernel itself — the pointer offset is applied at execute time.
    ConvParams sliced = p;
    sliced.C_full  = 0;  // kernel uses C, not C_full
    sliced.c_start = 0;  // no offset inside the kernel

    const std::string gen    = find_rocmlir_gen(driver);
    const std::string perf_cfg = get_tuned_perf_config(sliced, gen, driver);
    const std::string gen_cmd  = rocmlir_gen_cmd(sliced, gen, perf_cfg);

    int exit_code = 0;
    const std::string base_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0)
        OPENVINO_THROW("rocmlir-gen failed for slice-conv (exit ", exit_code, ")");

    // Apply Conv+Bias+SiLU epilogue fusion
    const std::string fused_ir = patch_ir_bias_silu(base_ir, sliced, true);
    CompiledConv result;
    if (!fused_ir.empty()) {
        try {
            result = compile_mlir_ir(fused_ir, p.arch, driver);
            result.bias_fused = true;
            result.silu_fused = true;
        } catch (const std::exception& e) {
            std::cerr << "[SliceConv] bias+silu patch failed: " << e.what() << "\n";
            result = compile_mlir_ir(base_ir, p.arch, driver);
        }
    } else {
        result = compile_mlir_ir(base_ir, p.arch, driver);
    }

    result.skip_add_fused = false;
    kern_cache.save(cache_key, p.arch, result);

    std::cerr << "[SliceConv] compiled " << result.kernel_name
              << " grid=" << result.grid_x << " block=" << result.block_x << "\n";
    return result;
}

// ── Conv+Bias+SliceOutput+SiLU+Add(skip) kernel via MIGraphX ─────────────────
// Generates ONNX: Conv → Slice(axis=1, starts=[c_out_start], ends=[c_out_end]) → SiLU → Add(skip)
// MIGraphX pattern: mlir_convolution_broadcast_slice_add_sigmoid_mul_add
// K_full = total conv output channels, K_slice = c_out_end - c_out_start
// Arg order (5 args): (data, filter, bias, skip_input, output)

static std::string generate_conv_slice_out_silu_add_onnx(
        const ConvParams& p, int K_full, int c_out_start, int c_out_end,
        const std::string& tmp_path) {
    const int K_slice = c_out_end - c_out_start;
    const std::string py_script = R"(
import sys, onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
N,C,H,W,K_full,K_slice,R,S,pad_h,pad_w,sh,sw,dh,dw,G,fp16,c_out_start,c_out_end = [int(x) for x in sys.argv[1:19]]
out = sys.argv[19]
dt_np = np.float16 if fp16 else np.float32
dt_onnx = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
X = helper.make_tensor_value_info('X', dt_onnx, [N, C, H, W])
Skip = helper.make_tensor_value_info('Skip', dt_onnx, [N, K_slice, (H+2*pad_h-dh*(R-1)-1)//sh+1, (W+2*pad_w-dw*(S-1)-1)//sw+1])
Y = helper.make_tensor_value_info('Y', dt_onnx, None)
W_val = np.ones([K_full, C//G, R, S], dtype=dt_np)*dt_np(0.01)
B_val = np.ones([K_full], dtype=dt_np)*dt_np(0.1)
W_init = numpy_helper.from_array(W_val, 'W')
B_init = numpy_helper.from_array(B_val, 'B')
# Slice parameters
starts_init = numpy_helper.from_array(np.array([c_out_start], dtype=np.int64), 'starts')
ends_init   = numpy_helper.from_array(np.array([c_out_end],   dtype=np.int64), 'ends')
axes_init   = numpy_helper.from_array(np.array([1],           dtype=np.int64), 'axes')
nodes = [
    helper.make_node('Conv',    ['X','W','B'],           ['conv_out'],
        kernel_shape=[R,S], pads=[pad_h,pad_w,pad_h,pad_w],
        strides=[sh,sw], dilations=[dh,dw], group=G),
    helper.make_node('Slice',   ['conv_out','starts','ends','axes'], ['sliced']),
    helper.make_node('Sigmoid', ['sliced'],                ['sig']),
    helper.make_node('Mul',     ['sliced','sig'],          ['silu']),
    helper.make_node('Add',     ['silu','Skip'],           ['Y']),
]
graph = helper.make_graph(nodes, 'g', [X, Skip], [Y],
    [W_init, B_init, starts_init, ends_init, axes_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 7
onnx.save(model, out)
print('OK')
)";
    const std::string script_file = tmp_path + "_sliceout.py";
    const std::string onnx_file   = tmp_path + "_sliceout.onnx";
    { std::ofstream f(script_file); f << py_script; }
    const char* py_env = std::getenv("ROCMLIR_PYTHON");
    const std::string python = py_env ? py_env : "python3";
    const std::string cmd = python + " " + script_file + " "
        + std::to_string(p.N) + " " + std::to_string(p.C) + " "
        + std::to_string(p.H) + " " + std::to_string(p.W) + " "
        + std::to_string(K_full) + " " + std::to_string(K_slice) + " "
        + std::to_string(p.R) + " " + std::to_string(p.S) + " "
        + std::to_string(p.pad_h) + " " + std::to_string(p.pad_w) + " "
        + std::to_string(p.stride_h) + " " + std::to_string(p.stride_w) + " "
        + std::to_string(p.dilation_h) + " " + std::to_string(p.dilation_w) + " "
        + std::to_string(p.groups) + " " + std::to_string(p.fp16 ? 1 : 0) + " "
        + std::to_string(c_out_start) + " " + std::to_string(c_out_end) + " "
        + onnx_file + " 2>/dev/null";
    int exit_code = 0;
    const std::string result = run_cmd(cmd, exit_code);
    std::remove(script_file.c_str());
    if (exit_code != 0 || result.find("OK") == std::string::npos) return "";
    return onnx_file;
}

CompiledConv compile_conv_slice_out_silu_add(const ConvParams& p,
                                               int K_full, int c_out_start, int c_out_end,
                                               const std::string& drv) {
    // Compile Conv+Bias+SiLU+Add for the sliced output shape [N, K_slice, OH, OW].
    // The full output [N, K_full, OH, OW] is handled by the caller at execute time
    // by providing a pointer offset to the correct output slice.
    // K_slice = c_out_end - c_out_start.
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;

    auto& kern_cache = HsacoKernelCache::instance();
    const size_t cache_key = p.hash()
        ^ static_cast<size_t>(0xCCC7DDDEEE8FULL)
        ^ (static_cast<size_t>(K_full) << 32)
        ^ (static_cast<size_t>(c_out_start) << 16)
        ^ static_cast<size_t>(c_out_end);
    {
        CompiledConv cached;
        if (kern_cache.load(cache_key, p.arch, cached)) {
            std::cerr << "[SliceOutConv-cache] loaded " << cached.kernel_name
                      << " grid=" << cached.grid_x << " block=" << cached.block_x << "\n";
            return cached;
        }
    }

    // Compile a standard Conv+Bias+SiLU+Add kernel for the sliced output shape.
    // The slice is applied by the caller via pointer arithmetic at execute time.
    const std::string gen = find_rocmlir_gen(driver);
    const std::string perf_cfg = get_tuned_perf_config(p, gen, driver);
    const std::string gen_cmd = rocmlir_gen_cmd(p, gen, perf_cfg);

    int exit_code = 0;
    const std::string base_ir = run_cmd(gen_cmd, exit_code);
    if (exit_code != 0)
        OPENVINO_THROW("rocmlir-gen failed for slice-out-silu-add (exit ", exit_code, ")");

    const std::string fused_ir = patch_ir_bias_silu_add(base_ir, p);
    CompiledConv result;
    if (!fused_ir.empty()) {
        try {
            result = compile_mlir_ir(fused_ir, p.arch, driver);
            result.bias_fused    = true;
            result.silu_fused    = true;
            result.skip_add_fused = true;
        } catch (const std::exception& e) {
            std::cerr << "[SliceOutConv] silu+add patch failed: " << e.what() << "\n";
            result = compile_mlir_ir(base_ir, p.arch, driver);
        }
    } else {
        result = compile_mlir_ir(base_ir, p.arch, driver);
    }

    kern_cache.save(cache_key, p.arch, result);
    std::cerr << "[SliceOutConv] compiled " << result.kernel_name
              << " grid=" << result.grid_x << " block=" << result.block_x << "\n";
    return result;
}

// ── Public migraphx dialect compilation entry ─────────────────────────────
CompiledConv compile_conv_fused_epilogue(const ConvParams& p, const std::string& drv,
                                    bool with_skip, bool with_silu_add) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;

    // Get tuned perf_config for fused_epilogue path.
    // Priority: fused-specific tuning (ROCMLIR_ENABLE_TUNING_FUSED=1) >
    //           plain rock.conv tuning cache > heuristic default.
    // The fused path measures conv+bias+silu fused kernel time directly,
    // giving more accurate results than reusing the plain conv cache.
    const std::string gen = find_rocmlir_gen(driver);
    std::string perf_cfg = get_tuned_perf_config_fused(p, gen, driver);
    if (perf_cfg.empty()) {
        // Fall back to plain conv tuning cache (close approximation)
        perf_cfg = get_tuned_perf_config(p, gen, driver);
    }

    auto& kc = HsacoKernelCache::instance();
    // with_skip=false,with_silu_add=false → 0x04; with_skip=true → 0x05; with_silu_add=true → 0x09
    const size_t variant = with_silu_add ? static_cast<size_t>(0x09ULL)
                         : with_skip     ? static_cast<size_t>(0x05ULL)
                                         : static_cast<size_t>(0x04ULL);
    // Include perf_cfg hash in key: different configs → different kernels
    const size_t perf_hash = std::hash<std::string>{}(perf_cfg);
    const size_t cache_key = (p.hash() ^ (static_cast<size_t>(0xB1A5B1A5B1A50000ULL) | variant))
                             ^ perf_hash;
    { CompiledConv cached; if (kc.load(cache_key, p.arch, cached)) {
        std::cerr << "[FusedEpilogue-cache] loaded skip=" << with_skip << " silu_add=" << with_silu_add
                  << " kernel=" << cached.kernel_name << "\n";
        return cached; } }

    const std::string mlir_ir = generate_fused_epilogue_ir(p, with_skip, with_silu_add, perf_cfg);
    auto result = compile_migraphx_ir(mlir_ir, p.arch, driver);
    result.bias_fused     = true;
    result.silu_fused     = true;
    result.skip_add_fused = with_skip || with_silu_add;
    kc.save(cache_key, p.arch, result);
    return result;
}

// ── Conv+Bias+Reshape kernel (no SiLU): Q/K/V attention projection pattern ──
// Generates mlir_convolution_broadcast_add_reshape kernel.
// reshape_dims: target tensor shape after flattening spatial dims.
// Returns 3-arg kernel: (input, filter, bias) → reshaped output.
CompiledConv compile_conv_fused_reshape(const ConvParams& p,
                                            const std::vector<int>& reshape_dims,
                                            const std::string& drv) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;

    // Include reshape_dims in cache key to distinguish different reshape targets
    auto& kc = HsacoKernelCache::instance();
    size_t reshape_hash = 0;
    for (int d : reshape_dims) reshape_hash = reshape_hash * 31 + static_cast<size_t>(d);
    const size_t cache_key = (p.hash() ^ static_cast<size_t>(0xB1A5B1A5B1A50007ULL)) ^ reshape_hash;
    { CompiledConv cached; if (kc.load(cache_key, p.arch, cached)) {
        std::cerr << "[FusedEpilogue-cache] loaded reshape kernel=" << cached.kernel_name << "\n";
        return cached; } }

    const std::string mlir_ir = generate_fused_reshape_ir(p, reshape_dims);
    auto compiled = compile_migraphx_ir(mlir_ir, p.arch, driver);
    compiled.bias_fused = true;
    compiled.silu_fused = false;
    compiled.skip_add_fused = false;
    kc.save(cache_key, p.arch, compiled);
    return compiled;
}

// Conv+Bias+SkipAdd (NO SiLU): mlir_convolution_broadcast_add_add (4-arg kernel).
// Eliminates the separate bias_add kernel launch for FC with NO_ACTIVATION + has_add.
// Matches MIGraphX's 15-instance conv+bias+skip pattern in yolo26x.
CompiledConv compile_conv_fused_skip(const ConvParams& p, const std::string& drv) {
    const std::string driver = drv.empty() ? find_rocmlir_driver() : drv;

    auto& kc = HsacoKernelCache::instance();
    const size_t cache_key = p.hash() ^ static_cast<size_t>(0xB1A5B1A5B1A50006ULL);
    { CompiledConv cached; if (kc.load(cache_key, p.arch, cached)) {
        std::cerr << "[FusedEpilogue-cache] loaded skip_no_silu kernel=" << cached.kernel_name << "\n";
        return cached; } }

    const std::string mlir_ir = generate_fused_skip_ir(p);
    auto compiled = compile_migraphx_ir(mlir_ir, p.arch, driver);
    compiled.bias_fused     = true;
    compiled.silu_fused     = false;
    compiled.skip_add_fused = true;
    kc.save(cache_key, p.arch, compiled);
    return compiled;
}

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
