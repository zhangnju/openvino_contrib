# OpenVINO ROCm Plugin — 性能指南

本文档面向希望在 AMD GPU（gfx1201/RDNA4、gfx1100/RDNA3、gfx1151/RDNA3.5、gfx950/CDNA3 等）上
使用 OpenVINO ROCm Plugin 进行推理性能调优的工程师，涵盖：rocMLIR 编译安装、Plugin 构建、Benchmark 复现、
性能相关环境变量，以及各架构的性能对比和优化历程。

---

## 目录

1. [环境依赖](#1-环境依赖)
2. [rocMLIR 编译安装](#2-rocmlir-编译安装)
3. [构建步骤](#3-构建步骤)
4. [安装 / 部署](#4-安装--部署)
5. [Benchmark 复现](#5-benchmark-复现)
6. [性能相关环境变量](#6-性能相关环境变量)
7. [各架构性能对比（yolo26x）](#7-各架构性能对比yolo26x)
8. [性能优化历程](#8-性能优化历程)
9. [MIGraphX 融合机制分析](#9-migraphx-融合机制分析)
10. [附录：常见问题](#附录常见问题)

---

## 1. 环境依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| OS | Ubuntu 22.04 / 24.04 (64-bit) | |
| ROCm | **7.2.x**（7.2.1 或 7.2.2） | 含 HIP、MIOpen、rocBLAS |
| GPU 架构 | gfx1201（RDNA4）/ gfx1100（RDNA3）/ gfx1151（RDNA3.5）/ gfx950（CDNA3） | |
| Python | 3.10+ | 用于 rocMLIR-gen 调优脚本 |
| CMake | 3.23+ | |
| 编译器 | `/opt/rocm/bin/amdclang++` | ROCm 自带，支持 HIP |

### 验证 GPU 架构

```bash
rocminfo | grep -E "gfx|Marketing" | head -10
```

### 验证 ROCm 安装

```bash
cat /opt/rocm/.info/version        # 确认 ROCm 版本
rocm-smi                           # 查看 GPU 显存占用
```

---

## 2. rocMLIR 编译安装

OpenVINO ROCm Plugin 使用 rocMLIR 作为卷积后端（替代不稳定的 MIOpen Immediate Mode）。
rocMLIR 将卷积编译为 HSACO GPU kernel，支持 Conv+Bias+SiLU 融合，在 RDNA/CDNA 架构上显著优于 MIOpen。

**重要**：必须在插件构建之前安装好 rocMLIR。

### 2.1 选择 rocMLIR 版本

| ROCm 版本 | 推荐 rocMLIR tag |
|-----------|----------------|
| ROCm 7.2.x | `rocm-6.3.0` |
| ROCm 7.0.x | `rocm-6.1.0` |

### 2.2 克隆 & 编译

```bash
git clone --branch rocm-6.3.0 --depth 1 \
    https://github.com/ROCm/rocMLIR.git /root/rocMLIR_src

cmake -S /root/rocMLIR_src -B /root/rocmlir_build \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DROCM_PATH=/opt/rocm \
    -DCMAKE_INSTALL_PREFIX=/root/rocmlir_install \
    -G Ninja

# 只编译必要的两个工具（约 20-40 分钟）
ninja -j$(nproc) -C /root/rocmlir_build rocmlir-gen rocmlir-driver

mkdir -p /root/rocmlir_install/bin
cp /root/rocmlir_build/bin/rocmlir-gen    /root/rocmlir_install/bin/
cp /root/rocmlir_build/bin/rocmlir-driver /root/rocmlir_install/bin/
```

### 2.3 验证

```bash
/root/rocmlir_install/bin/rocmlir-gen --help 2>&1 | head -3
/root/rocmlir_install/bin/rocmlir-gen -ph -t f16 \
    -batchsize 1 -in_channels 64 -in_h 32 -in_w 32 \
    -fil_h 3 -fil_w 3 -out_channels 64 --arch gfx1201 2>&1 | head -3
```

---

## 3. 构建步骤

### 3.1 克隆源码

```bash
git clone https://github.com/openvinotoolkit/openvino.git /home/openvino/openvino
cd /home/openvino/openvino && git submodule update --init --recursive

git clone https://github.com/openvinotoolkit/openvino_contrib.git /home/openvino/openvino_contrib
```

### 3.2 CMake 配置 & 编译

```bash
export OPENVINO_HOME=/home/openvino/openvino
export OPENVINO_CONTRIB=/home/openvino/openvino_contrib
export OPENVINO_BUILD_PATH=/home/openvino/openvino_build

mkdir -p ${OPENVINO_BUILD_PATH} && cd ${OPENVINO_BUILD_PATH}

cmake ${OPENVINO_HOME} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
    -DENABLE_ROCM=ON \
    -DENABLE_ROCMLIR=ON \
    -DENABLE_INTEL_NPU=OFF \
    -DENABLE_TESTS=OFF \
    -DBUILD_arm_plugin=OFF \
    -DBUILD_nvidia_plugin=OFF \
    -DBUILD_java_api=OFF \
    -DOPENVINO_EXTRA_MODULES="${OPENVINO_CONTRIB}/modules/rocm_plugin" \
    -DROCM_PATH=/opt/rocm \
    -DHIP_PATH=/opt/rocm/hip \
    -DROCMLIR_INSTALL_DIR=/root/rocmlir_install \
    -DCMAKE_HIP_ARCHITECTURES=<your_gpu_arch> \
    -G Ninja

ninja -j$(nproc) openvino_rocm_gpu_plugin benchmark_app
```

**按架构指定 `CMAKE_HIP_ARCHITECTURES`：**

| GPU | 架构 | 典型硬件 |
|-----|------|---------|
| Radeon AI PRO R9700 | `gfx1201` | RDNA4 工作站卡 |
| RX 7900 XTX | `gfx1100` | RDNA3 旗舰消费卡 |
| Ryzen AI MAX+ 395 集成显卡 | `gfx1151` | RDNA3.5 APU |
| Instinct MI350 | `gfx950` | CDNA3 数据中心卡 |

### 3.3 仅重编 Plugin

```bash
cd ${OPENVINO_BUILD_PATH}/build-modules/rocm_plugin && make -j$(nproc)
```

---

## 4. 安装 / 部署

```bash
export LD_LIBRARY_PATH=${OPENVINO_HOME}/bin/intel64/Release:/opt/rocm/lib:/root/rocmlir_install/lib:${LD_LIBRARY_PATH}

# 多卡环境：指定空闲 GPU
rocm-smi                          # 查看 VRAM 占用
export HIP_VISIBLE_DEVICES=1      # 选择 VRAM 空闲的卡
```

---

## 5. Benchmark 复现

### 5.1 吞吐量模式（Throughput）

```bash
HIP_VISIBLE_DEVICES=<空闲GPU> \
ROCM_FUSE_ATTENTION=1 \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint throughput -t 30
```

### 5.2 延迟模式（Latency）

```bash
HIP_VISIBLE_DEVICES=<空闲GPU> \
ROCM_FUSE_ATTENTION=1 \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint latency -t 30
```

### 5.3 最高性能配置（首次较慢，结果缓存）

> **重要**：`ROCMLIR_ENABLE_TUNING=1` 对每个卷积 shape 穷举搜索（约 50 个 shape，耗时 10-40 分钟）。
> 结果缓存于 `~/.cache/ov_rocmlir_tuning_<arch>.json`，后续直接加载无开销。
> **必须在独占 GPU 上运行 tuning**，共享 GPU 会导致测量噪声大、配置不准确。

```bash
# gfx1201 / gfx1100 推荐（tuning + skip fusion）
HIP_VISIBLE_DEVICES=<独占GPU> \
ROCM_FUSE_ATTENTION=1 \
ROCMLIR_ENABLE_TUNING=1 \
ROCMLIR_CONV_SKIP_FUSION=1 \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint throughput -t 30

# gfx950 推荐（tuning 效果最显著）
HIP_VISIBLE_DEVICES=<独占GPU> \
ROCM_FUSE_ATTENTION=1 \
ROCMLIR_ENABLE_TUNING=1 \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint throughput -t 30
```

### 5.4 GPU 级 Kernel 性能剖析

```bash
HIP_VISIBLE_DEVICES=1 ROCM_FUSE_ATTENTION=1 \
rocprof --stats -o /tmp/ov_profile.csv \
    ${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint throughput -t 10

sort -t',' -k4 -rn /tmp/ov_profile.stats.csv | head -20
```

---

## 6. 性能相关环境变量

### 6.1 ROCm Plugin 核心变量

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `ROCM_FUSE_ATTENTION` | `1` | 启用 Attention MatMul 融合（Q×Kᵀ + A×V 用 MIGraphX MLIR kernel 替代 rocBLAS）。**必须开启**以获得最优性能。 |
| `ROCM_FUSE_PE` | `1` | pe(V) 深度卷积与 Attention 输出融合（仅 gfx1201/RDNA4 稳定，gfx950 默认关）。 |
| `ROCM_ZEROCOPY_SPLIT` | `1` | VariadicSplit 零拷贝（QKV split 改为 buffer alias）。 |
| `ROCM_SWISH_INPLACE` | `1` | SiLU 激活原位写入，减少一次显存分配。 |

### 6.2 rocMLIR 调优变量

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `ROCMLIR_ENABLE_TUNING` | 未设置 | 设为 `1` 开启穷举调优。gfx950 可从 330→500 FPS（+52%），gfx1201 从 175→220 FPS（+26%）。首次慢，之后用缓存。 |
| `ROCMLIR_TUNING_CACHE` | `~/.cache/ov_rocmlir_tuning_<arch>.json` | 调优缓存路径。多用户环境可集中管理。 |
| `ROCMLIR_PERF_CONFIG` | 自动 | 强制指定特定 perf_config，覆盖调优缓存。 |
| `ROCMLIR_DRIVER` | 自动探测 | `rocmlir-driver` 可执行文件路径。 |

### 6.3 MLIR Epilogue Fusion 变量（MIGraphX 风格融合）

这组变量控制把 conv 之后的 elementwise 操作融合进 conv kernel，减少 kernel launch 次数，对标 MIGraphX 的 MLIR 深度融合。

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `ROCMLIR_CONV_SKIP_FUSION` | 未设置 | 设为 `1` 启用 **conv+bias+skip（无SiLU）→ 单 kernel**。消除 `NO_ACTIVATION + has_add` 情况下的单独 skip-add kernel。对应 MIGraphX 的 `mlir_convolution_broadcast_add_add` 模式（yolo26x 中有 15 个）。**gfx1201 实测 +9 FPS（216→225 FPS）**，达到与 MIGraphX 持平。 |
| `ROCMLIR_CONV_RESHAPE_FUSION` | 未设置 | 设为 `1` 启用 **conv+bias+reshape → 单 kernel**。将 Q/K/V 注意力投影后的 reshape 融入 conv kernel 写出（零开销视图变换），对应 MIGraphX 的 `mlir_convolution_broadcast_add_reshape` 模式。yolo26x 中有 9 个可融合的单消费者 reshape 模式。 |
| `ROCMLIR_EPILOGUE_FUSION` | 未设置 | 设为 `1` 同时启用 `ROCMLIR_CONV_SKIP_FUSION` 和 `ROCMLIR_CONV_RESHAPE_FUSION`，并额外开启 conv+bias+silu 的 migraphx 编译路径（**注意：gfx1201 上此路径可能比 v3: tuning 慢，建议单独使用 `ROCMLIR_CONV_SKIP_FUSION`**）。 |

### 6.4 HIP 系统变量

| 环境变量 | 说明 |
|----------|------|
| `HIP_VISIBLE_DEVICES` | 限制可见 GPU（0-based 索引）。多卡共享服务器**必须指定**，避免 Memory Access Fault。 |

### 6.5 推荐配置组合

```bash
# ── gfx1201（R9700）最优配置 ──
HIP_VISIBLE_DEVICES=<独占GPU> \
ROCM_FUSE_ATTENTION=1 \
ROCMLIR_ENABLE_TUNING=1 \
ROCMLIR_CONV_SKIP_FUSION=1 \
./benchmark_app -m model.onnx -d ROCM -hint throughput
# 预期：~225 FPS（与 MIGraphX 227 FPS 持平）

# ── gfx1100（RX 7900 XTX）最优配置 ──
HIP_VISIBLE_DEVICES=<独占GPU> \
ROCM_FUSE_ATTENTION=1 \
ROCMLIR_ENABLE_TUNING=1 \
./benchmark_app -m model.onnx -d ROCM -hint throughput
# 预期：~221 FPS（vs MIGraphX ~168 FPS，领先 32%）

# ── gfx950（MI350）最优配置 ──
HIP_VISIBLE_DEVICES=<独占GPU> \
ROCM_FUSE_ATTENTION=1 \
ROCMLIR_ENABLE_TUNING=1 \
./benchmark_app -m model.onnx -d ROCM -hint throughput
# 预期：~500 FPS（vs MIGraphX ~295 FPS，领先 69%）

# ── 调试 baseline（关闭所有融合）──
HIP_VISIBLE_DEVICES=<GPU> \
ROCM_FUSE_ATTENTION=0 ROCM_ZEROCOPY_SPLIT=0 ROCM_SWISH_INPLACE=0 \
./benchmark_app -m model.onnx -d ROCM -hint throughput
```

---

## 7. 各架构性能对比（yolo26x，FP16）

### 7.1 吞吐量（Throughput）模式

| GPU | 架构 | OV 基础 | OV + Tuning | OV + Tuning + SkipFusion | MIGraphX |
|-----|------|---------|------------|--------------------------|---------|
| Instinct MI350 | gfx950 / CDNA3 | ~330 FPS | **~500 FPS** | — | ~295 FPS |
| Radeon AI PRO R9700 | gfx1201 / RDNA4 | ~175 FPS | ~220 FPS | **~225 FPS** ✅ | ~227 FPS |
| RX 7900 XTX | gfx1100 / RDNA3 | ~170 FPS | **~221 FPS** | — | ~168 FPS |
| Ryzen AI MAX+ 395 | gfx1151 / RDNA3.5 | **~80 FPS** | ~80 FPS | — | ~111 FPS |

> ✅ = 达到或超过 MIGraphX 水平

### 7.2 延迟（Latency）模式 — 单请求串行

| GPU | 架构 | OV Median 延迟 | OV 单流吞吐 | MIGraphX Median |
|-----|------|---------------|------------|-----------------|
| Instinct MI350 | gfx950 | **5.20 ms** | ~192 FPS | 3.36 ms |
| Radeon AI PRO R9700 | gfx1201 | **7.30 ms** | ~137 FPS | 4.42 ms |
| RX 7900 XTX | gfx1100 | **6.97 ms** | ~143 FPS | 5.96 ms |
| Ryzen AI MAX+ 395 | gfx1151 | **13.96 ms** | ~72 FPS | 9.01 ms |

> **说明**：延迟模式下 OV 有 ~299 个 kernel launch 的 CPU dispatch 开销（~1.4ms/inf），MIGraphX 只有 ~84 个 kernel（~0.4ms/inf）。吞吐量模式通过多并发隐藏了这一开销。

---

## 8. 性能优化历程（yolo26x，gfx950）

以下记录了 OpenVINO ROCm Plugin 在 yolo26x（640×640 输入，FP16）上的完整优化历程。

| 阶段 | 优化项 | 修改文件 | 吞吐量 | 增量 | 说明 |
|------|--------|----------|--------|------|------|
| Baseline | MIOpen Immediate Mode | — | **~130 FPS** | — | gfx950 上 solution_id 85/88 不稳定，频繁 GPU fault |
| +rocMLIR | 卷积后端切换 rocMLIR | `fused_convolution.cpp` 工厂优先级 | **~200 FPS** | +70 | rock.conv → HSACO，绕过不稳定路径，支持 Conv+Bias+SiLU 融合 |
| +ZeroCopy | VariadicSplit 零拷贝 | `variadic_split_zero_copy.cpp` | **~220 FPS** | +20 | QKV split 改为 buffer alias，消除 D2D 拷贝 |
| +EWFusion | ElementwiseFusionPass | `elementwise_fusion_transformation.cpp` | **~250 FPS** | +30 | Sigmoid/Mul/Add 链融合为单 FusedElementwise kernel |
| +AttnFuse | Attention MatMul 融合 | `rocm_attention_matmul.cpp` | **~265 FPS** | +15 | Q×Kᵀ + A×V 用 MIGraphX MLIR kernel，32× attention GEMM 加速 |
| +HIPFirst | Add/Multiply/Clamp 工厂 HIP 优先 | `add.cpp`、`multiply.cpp`、`clamp.cpp` | **~330 FPS** | **+65** | 消除 `miopenOpTensor()→Op5dTensorGeneric`（原占 14.2% GPU 时间），HIP kernel 快 19× |
| +Tuning | rocMLIR perf_config 调优 | `rocmlir_compiler.cpp` | **~500 FPS** | +170 | `ROCMLIR_ENABLE_TUNING=1` 穷举最优 tile 配置，gfx950 获益最大 |

### gfx1201 专项优化历程

| 阶段 | 吞吐量 | 说明 |
|------|--------|------|
| 初始（仅 rocMLIR） | ~175 FPS | 默认 v3: heuristic |
| + Tuning（独占 GPU） | ~220 FPS | v3: 调优缓存，共享 GPU 下配置不准 |
| + `ROCMLIR_CONV_SKIP_FUSION=1` | **~225 FPS** | conv+bias+skip 无SiLU → 单 migraphx kernel，消除独立 skip-add launch |
| MIGraphX 对比 | ~227 FPS | 两者基本持平 |

---

## 9. MIGraphX 融合机制分析

### 9.1 MIGraphX 为何 kernel 少

通过 `MIGRAPHX_TRACE_MLIR=1` 分析 yolo26x，MIGraphX 使用以下 kernel 类型：

| MIGraphX kernel 类型 | 编译次数 | 对应 OV 处理方式 | 差异 |
|---------------------|---------|----------------|------|
| `conv+bias+silu` | 587 | `mlir_conv_bias_silu` | 两者均有，OV 已匹配 |
| `conv+bias+reshape` | 168 | conv kernel + 零开销 reshape 视图 | OV reshape 已零开销，无 kernel 差异 |
| `slice+conv+bias+silu` | 121 | `FusedConvolutionSlice` | OV 已匹配 |
| `conv+bias` (无SiLU) | 34 | `mlir_conv_bias` | 两者均有 |
| **`conv+bias+add+add`** | **15** | conv kernel + 单独 `bias_add` kernel | **OV 缺失！** |
| reshape+slice+attention | 22+22+34 | `RocmAttentionMatMulOp` | OV 用注意力 MLIR kernel 覆盖 |

**根本原因**：MIGraphX 把 `conv+bias+skip（无SiLU）` 的 bias-add 和 skip-add 融合进单 kernel (`mlir_convolution_broadcast_add_add`)；OV 原本会为此运行两个 kernel（conv+bias + 单独 bias_add）。`ROCMLIR_CONV_SKIP_FUSION=1` 修复了这一差距。

### 9.2 rocMLIR 的融合限制

- **✅ 支持**：一个 function 内 = 一个 conv + 任意 elementwise epilogue（add、mul、sigmoid、reshape 等）
- **❌ 不支持**：一个 function 内包含多个 conv（报错：`Multiple Fusion Roots detected`）

因此，OV 通过 epilogue fusion（`ROCMLIR_CONV_SKIP_FUSION` 等）逐渐逼近 MIGraphX 的 kernel 数量，而无法通过"多 conv 合一 kernel"超越。

### 9.3 MIGraphX 是否使用 HIP Graph

**否。** 通过 `nm -D libmigraphx_gpu.so` 确认 MIGraphX 没有导入任何 `hipGraph*` 相关 API。MIGraphX 的低 CPU dispatch 开销来自 **更少的 kernel 数量**（~84/inf vs OV 的 ~299/inf），而非 HIP Graph 技术。

OV Plugin 本身实现了 HIP Graph 支持（可通过 `ROCM_USE_HIP_GRAPH=YES` 开启），但目前在部分 GPU 上驱动支持不完整。

---

## 附录：常见问题

### Memory Access Fault

```
Memory access fault by GPU node-X (Agent handle: ...) on address ...
```

**原因**：目标 GPU 的 VRAM 已被其他进程占满，或使用了共享 GPU 上的内存。

**解决**：`rocm-smi` 查看 VRAM 占用，`HIP_VISIBLE_DEVICES=N` 指定空闲 GPU。

### Tuning 结果不准（性能偏低）

如果 `ROCMLIR_ENABLE_TUNING=1` 得到的 tuning 缓存反而使性能下降，原因是在**共享 GPU** 上运行 tuning 时，其他进程的负载导致各配置的测量时间不准，选出了错误的"最优"配置。

**解决**：删除缓存文件（`~/.cache/ov_rocmlir_tuning_gfx1201.json` 等），在**独占 GPU** 上重新运行 tuning。

### rocMLIR kernel 首次编译慢

首次推理时会 JIT 编译 MLIR kernel（约 30-120 秒），后续缓存至 `~/.cache/ov_rocmlir_tuning_<arch>.json`。

### 验证 HIP 优先路径（消除 Op5dTensorGeneric）

```bash
HIP_VISIBLE_DEVICES=1 ROCM_FUSE_ATTENTION=1 \
rocprof --stats -o /tmp/prof.csv \
    ./benchmark_app -m model.onnx -d ROCM -hint throughput -t 10

grep "Op5dTensor" /tmp/prof.stats.csv
# 若无输出，说明 MIOpen 通用路径已被正确绕过
```

### 验证 Conv+Skip 融合是否生效

```bash
HIP_VISIBLE_DEVICES=1 ROCM_FUSE_ATTENTION=1 ROCMLIR_CONV_SKIP_FUSION=1 \
rocprof --stats -o /tmp/prof_skip.csv \
    ./benchmark_app -m model.onnx -d ROCM -hint throughput -t 10

grep "mlir_convolution_broadcast_add_add" /tmp/prof_skip.stats.csv
# 若出现此 kernel，说明 conv+bias+skip 融合已生效
```
