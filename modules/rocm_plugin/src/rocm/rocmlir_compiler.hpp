// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// rocMLIR-based convolution compiler.
// Generates rock.conv MLIR IR from convolution parameters and compiles it
// to HSACO binary using the rocmlir-driver tool (subprocess).
//
// This mirrors MIGraphX's fuse_mlir pass but operates out-of-process,
// invoking rocmlir-driver as a subprocess to avoid linking the full LLVM stack.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ov {
namespace rocm_gpu {
namespace rocmlir {

// ───────────────────────────────────────────────────────────────────────────
// Convolution parameters (NCHW input, KCRS filter, strided 2-D forward conv)
// ───────────────────────────────────────────────────────────────────────────
struct ConvParams {
    // Input tensor N×C×H×W
    int N = 1, C = 1, H = 1, W = 1;
    // Filter K×C×R×S  (K = output channels, R/S = kernel height/width)
    int K = 1, R = 1, S = 1;
    // Padding (symmetric)
    int pad_h = 0, pad_w = 0;
    // Stride
    int stride_h = 1, stride_w = 1;
    // Dilation
    int dilation_h = 1, dilation_w = 1;
    // Groups (1 = standard conv, C = depthwise)
    int groups = 1;
    // Data type: true = fp16, false = fp32
    bool fp16 = true;
    // Target GPU architecture string, e.g. "gfx950"
    std::string arch = "gfx950";
    // Number of compute units (used for tuning)
    int num_cu = 256;

    // Derived output spatial dims
    int out_h() const { return (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1; }
    int out_w() const { return (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1; }

    // Equality for cache keying
    bool operator==(const ConvParams& o) const;
    size_t hash() const;
};

// ───────────────────────────────────────────────────────────────────────────
// Compilation result
// ───────────────────────────────────────────────────────────────────────────
struct CompiledConv {
    std::vector<char> hsaco;       // Compiled HSACO binary
    std::string       kernel_name; // Entry point name (e.g. "mlir_conv_f16_...")
    size_t            workspace_bytes = 0;
    // perf_config string used (e.g. "v4:64,256,8,16,256,16,4,1,2,2,1,1")
    std::string perf_config;
    // Grid/block dims extracted from the compiled kernel
    int grid_x = 1, grid_y = 1, grid_z = 1;
    int block_x = 64, block_y = 1, block_z = 1;
};

// ───────────────────────────────────────────────────────────────────────────
// Activation type for fused kernels
// ───────────────────────────────────────────────────────────────────────────
enum class Activation { None, ReLU, Sigmoid };

// ───────────────────────────────────────────────────────────────────────────
// Compiler interface
// ───────────────────────────────────────────────────────────────────────────

// Generate rock.conv MLIR IR text for the given parameters.
// The output matches what MIGRAPHX_TRACE_MLIR=2 produces.
std::string generate_conv_ir(const ConvParams& p);

// Generate fused rock.conv + broadcast_add (bias) MLIR IR.
std::string generate_fused_conv_bias_ir(const ConvParams& p);

// Generate fused rock.conv + broadcast_add + activation MLIR IR.
std::string generate_fused_conv_bias_act_ir(const ConvParams& p, Activation act);

// Compile MLIR IR text to HSACO binary via rocmlir-driver subprocess.
// Returns CompiledConv on success; throws on failure.
// rocmlir_driver_path: path to rocmlir-driver binary (auto-detected if empty).
CompiledConv compile_mlir_ir(const std::string& mlir_ir,
                              const std::string& arch,
                              const std::string& rocmlir_driver_path = "");

// High-level: generate + compile convolution kernel.
CompiledConv compile_conv(const ConvParams& p,
                          const std::string& rocmlir_driver_path = "");

// High-level: generate + compile fused conv+bias kernel.
CompiledConv compile_fused_conv_bias(const ConvParams& p,
                                     const std::string& rocmlir_driver_path = "");

// High-level: generate + compile fused conv+bias+activation kernel.
CompiledConv compile_fused_conv_bias_act(const ConvParams& p,
                                         Activation act,
                                         const std::string& rocmlir_driver_path = "");

// Find rocmlir-driver in standard install locations.
// Search order: env ROCMLIR_DRIVER, /home/openvino/rocmlir_install/bin,
//               /opt/rocmlir/bin, PATH.
std::string find_rocmlir_driver();

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
