// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// rocMLIR-based convolution compiler.
// Generates rock.conv MLIR IR from convolution parameters and compiles it
// to HSACO binary using the rocmlir-driver tool (subprocess).
//
// Invokes rocmlir-driver as a subprocess to avoid linking the full LLVM stack.

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

    // Optional input channel slice: the input tensor passed at runtime has C_full channels
    // and we use channels [c_start, c_start + C) in the convolution.
    // When c_start == 0 and C_full == C, no slice is applied (normal mode).
    // This enables fusing VariadicSplit → FusedConvolution into a single kernel that
    // reads directly from the pre-split buffer with a channel offset.
    int C_full = 0;    // 0 means use C (no slice)
    int c_start = 0;   // channel start index in the full input tensor

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
    // Grid/block dims parsed from rocmlir-driver metadata:
    //   block_size = 256 : i32, grid_size = 600 : i32
    int grid_x = 1, grid_y = 1, grid_z = 1;
    int block_x = 256, block_y = 1, block_z = 1;
    // True when the kernel is a bias-fused variant:
    //   args = (filter, input, bias, output)   [4 args instead of 3]
    // The calling code must pass bias as 3rd argument.
    bool bias_fused = false;
    // True when bias+SiLU are both fused (implies bias_fused=true)
    bool silu_fused = false;
    // True when bias+SiLU+skip-Add are all fused (5-arg kernel: filter,input,bias,skip,output)
    // Implies bias_fused=true and silu_fused=true.
    bool skip_add_fused = false;
};

// ───────────────────────────────────────────────────────────────────────────
// Activation type for fused kernels
// ───────────────────────────────────────────────────────────────────────────
enum class Activation { None, ReLU, Sigmoid };

// ───────────────────────────────────────────────────────────────────────────
// Compiler interface
// ───────────────────────────────────────────────────────────────────────────

// Generate rock.conv MLIR IR text for the given parameters.
// Uses rocmlir-gen to generate the MLIR IR text.
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

// High-level: generate + compile fused conv+bias+SiLU+skip-Add kernel (5 args).
// Args order: (input, filter, bias, skip_input, output)
CompiledConv compile_fused_conv_bias_silu_add(const ConvParams& p,
                                               const std::string& rocmlir_driver_path = "");

// High-level: generate + compile Slice+Conv+Bias+SiLU fused kernel.
// p.C_full = full input channels, p.c_start = channel offset, p.C = used channels.
// The caller applies the c_start byte offset to the input pointer at execute time.
// Arg order: (input_slice, filter, bias, output)
CompiledConv compile_slice_conv_bias_silu(const ConvParams& p,
                                           const std::string& rocmlir_driver_path = "");

// High-level: Conv+Bias+SiLU+Add(skip) kernel for sliced output.
// K_full = total conv output channels. Output slice = [c_out_start, c_out_end).
// Arg order (5 args): (filter, data, bias, skip_input, output)
CompiledConv compile_conv_slice_out_silu_add(const ConvParams& p,
                                              int K_full,
                                              int c_out_start,
                                              int c_out_end,
                                              const std::string& rocmlir_driver_path = "");

// MIGraphX dialect compilation: conv+bias+silu with epilogue ops.
// Uses -kernel-pipeline=migraphx,highlevel,gpu,rocdl,binary instead of rock.conv full.
// Activated by ROCMLIR_EPILOGUE_FUSION=1 env var.
//
// Epilogue variants:
//   with_skip=false, with_silu_add=false → 4-arg: conv+bias+silu
//   with_skip=true,  with_silu_add=false → 5-arg: conv+bias+silu+skip_add
//   with_skip=true,  with_silu_add=true  → 6-arg: conv+bias+silu+skip_add+silu+aux_add
//     (fuses C2f/C2PSA bottleneck: FC(silu+shortcut) → silu(fc_out) → add(silu, cv1_silu))
CompiledConv compile_conv_migraphx(const ConvParams& p,
                                    const std::string& rocmlir_driver_path = "",
                                    bool with_skip = false,
                                    bool with_silu_add = false);

// Conv+Bias+SkipAdd (NO SiLU): mlir_convolution_broadcast_add_add (4-arg kernel).
// Fuses conv+bias+skip_add WITHOUT silu into one kernel.
// Eliminates separate launch_bias_add for NO_ACTIVATION + has_add cases.
// Matches MIGraphX's 15-instance mlir_convolution_broadcast_add_add pattern.
CompiledConv compile_conv_migraphx_skip(const ConvParams& p,
                                         const std::string& rocmlir_driver_path = "");

// Conv+Bias+Reshape kernel: Q/K/V attention projection pattern.
// reshape_dims: target shape after conv output, e.g. {N, K, OH*OW}.
// Generates mlir_convolution_broadcast_add_reshape (3-arg kernel).
// Matches MIGraphX's 168-instance pattern in yolo26x.
CompiledConv compile_conv_migraphx_reshape(const ConvParams& p,
                                            const std::vector<int>& reshape_dims,
                                            const std::string& rocmlir_driver_path = "");

// Find rocmlir-driver in standard install locations.
// Search order: env ROCMLIR_DRIVER, /home/openvino/rocmlir_install/bin,
//               /opt/rocmlir/bin, PATH.
std::string find_rocmlir_driver();

} // namespace rocmlir
} // namespace rocm_gpu
} // namespace ov
