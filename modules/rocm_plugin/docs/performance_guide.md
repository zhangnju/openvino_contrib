# OpenVINO ROCm Plugin — 性能指南

本文档面向希望在 AMD GPU（gfx1201/RDNA4、gfx1100/RDNA3、gfx950/CDNA3 等）上
使用 OpenVINO ROCm Plugin 进行推理性能调优的工程师。

---

## 目录

1. [环境依赖](#1-环境依赖)
2. [rocMLIR 编译安装](#2-rocmlir-编译安装)
3. [构建步骤](#3-构建步骤)
4. [Benchmark 复现](#4-benchmark-复现)
5. [性能相关环境变量](#5-性能相关环境变量)
6. [各架构性能对比](#6-各架构性能对比)
7. [优化历程与技术细节](#7-优化历程与技术细节)
8. [附录：常见问题](#8-附录常见问题)

---

## 1. 环境依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| OS | Ubuntu 22.04 / 24.04 (64-bit) | |
| ROCm | **7.2.x** | 含 HIP、MIOpen、rocBLAS |
| GPU | gfx1201 / gfx1100 / gfx950 | 见下表 |
| Python | 3.10+ | 用于调优脚本 |
| CMake | 3.23+（pip install cmake） | |
| Host 编译器 | `gcc/g++` 13.x | rocMLIR 构建用 |

### 验证 GPU 架构

```bash
rocminfo | grep -E "gfx|Marketing" | head -10
# 输出示例：
# Name: gfx1201
# Marketing Name: AMD Radeon AI PRO R9700
```

---

## 2. rocMLIR 编译安装

ROCm Plugin 使用 rocMLIR 作为主要卷积后端，通过 rocmlir-driver/rocmlir-gen
将 Conv+Bias+SiLU 等 epilogue 融合编译为 GPU kernel（HSACO）。

**注意**：必须在 Plugin 构建之前安装好 rocMLIR，并应用 patches。

### 2.1 选择 rocMLIR 版本

| ROCm 版本 | 推荐 rocMLIR 分支 |
|-----------|----------------|
| ROCm 7.2.x | `rocm-rel-7.2`（tag: `rocm-7.2.4`） |
| ROCm 7.0.x | `rocm-rel-7.0` |

### 2.2 编译（以 gfx1201 为例）

```bash
git clone --branch rocm-rel-7.2 --depth 1 \
    https://github.com/ROCm/rocMLIR.git /home/rocMLIR

mkdir -p /home/rocMLIR/build && cd /home/rocMLIR/build

# 安装 cmake（若系统没有）
pip install cmake

cmake /home/rocMLIR \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DGPU_TARGETS=gfx1201 \
    -DROCMLIR_DRIVER_E2E_TEST_ENABLED=0 \
    -DROCMLIR_DRIVER_PR_E2E_TEST_ENABLED=0 \
    -DROCMLIR_PARALLEL_LINK_JOBS=8 \
    -DROCMLIR_PARALLEL_COMPILE_JOBS=64 \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_PROJECTS="mlir;lld"

# 构建（约 30-60 分钟）
ninja -j$(nproc) rocmlir-driver rocmlir-gen
```

### 2.3 应用 rocm_plugin patches（必须）

ROCm Plugin 在 `patches/` 目录下提供了针对 rocMLIR 的补丁，
**必须在 ninja 构建之前 apply**：

```bash
ROCM_PLUGIN_DIR=/home/openvino/openvino_contrib/modules/rocm_plugin
ROCMLIR_DIR=/home/rocMLIR

# 应用所有 patch（按文件名顺序）
for patch in ${ROCM_PLUGIN_DIR}/patches/*.patch; do
    echo "Applying: $(basename $patch)"
    git -C ${ROCMLIR_DIR} am --3way < "$patch"
done
```

**patches 说明：**

| Patch 文件 | 内容 | 必要性 |
|-----------|------|--------|
| `0001-rocmlir-fix-AlignTiling-build-error.patch` | 修复 `AlignTiling.cpp` 在 g++ Release 构建时的语法错误 | 必须（否则编译失败） |
| `0002-tosatorock-features.patch` | 修复 `TosaToRock.cpp`：`ForwardConvConverter` 生成的 `rock.conv` 现在正确传入 GPU arch features（wmma\|dot\|...），使 `rock-affix-params` 能选择 WMMA 优化的 tile 配置 | 必须（fused_epilogue 调优需要） |
| `0003-driver-tuning-fallback.patch` | 为 `rocmlir-driver` 添加 `--tuning-fallback` 选项（默认 true）：注入的 perf_config 无效时自动 fallback 到 heuristic，避免 "Lowering failed" | 必须（fused_epilogue 调优需要） |

### 2.4 安装

```bash
mkdir -p /home/rocmlir_install/bin
cp /home/rocMLIR/build/bin/rocmlir-driver /home/rocmlir_install/bin/
cp /home/rocMLIR/build/bin/rocmlir-gen    /home/rocmlir_install/bin/

# 验证
/home/rocmlir_install/bin/rocmlir-driver --help | head -5
```

---

## 3. 构建步骤

### 3.1 CMake 配置

```bash
export OPENVINO_HOME=/home/openvino/openvino
export OPENVINO_CONTRIB=/home/openvino/openvino_contrib
export OPENVINO_BUILD_PATH=/home/openvino_build

mkdir -p ${OPENVINO_BUILD_PATH} && cd ${OPENVINO_BUILD_PATH}

cmake ${OPENVINO_HOME} \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DENABLE_ROCM=ON \
    -DENABLE_ROCMLIR=ON \
    -DROCMLIR_INSTALL_DIR=/home/rocmlir_install \
    -DENABLE_OV_ONNX_FRONTEND=ON \
    -DENABLE_TESTS=OFF \
    -DBUILD_arm_plugin=OFF \
    -DBUILD_nvidia_plugin=OFF \
    -DBUILD_java_api=OFF \
    -DBUILD_ollama_openvino=OFF \
    -DOPENVINO_EXTRA_MODULES="${OPENVINO_CONTRIB}/modules" \
    -DWHEEL_VERSION=2024.4.0

ninja -j$(nproc) openvino_rocm_gpu_plugin benchmark_app openvino_onnx_frontend
```

**按架构指定 `CMAKE_HIP_ARCHITECTURES`：**

| GPU | 架构 | `CMAKE_HIP_ARCHITECTURES` |
|-----|------|--------------------------|
| Radeon AI PRO R9700 | RDNA4 | `gfx1201` |
| RX 7900 XTX | RDNA3 | `gfx1100` |
| Instinct MI350 | CDNA3 | `gfx950` |

---

## 4. Benchmark 复现

### 4.1 环境准备

```bash
export PATH=/home/rocmlir_install/bin:$PATH
source ${OPENVINO_HOME}/scripts/setupvars/setupvars.sh
# 或手动设置：
export LD_LIBRARY_PATH=${OPENVINO_HOME}/bin/intel64/Release:/opt/rocm/lib:$LD_LIBRARY_PATH
```

### 4.2 复现最优性能（gfx1201，280 FPS）

完整的三步流程，从零开始复现 **280 FPS**：

#### 步骤 A：plain rock.conv 调优（可选，约 20-40 分钟）

```bash
# 在独占 GPU 上运行，为 plain conv 形状生成调优配置
# 若已有 ov_rocmlir_tuning_gfx1201_release_new.json 可跳过
ROCMLIR_ENABLE_TUNING=1 \
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx1201.json \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM.0 -t 300 -nireq 1 -b 1
```

#### 步骤 B：fused_epilogue 专项调优（推荐，约 20-60 分钟）

> 这是获得 280 FPS 的关键步骤。对每个 conv+bias+silu fused kernel
> 独立测量并缓存最优 perf_config，结果存入独立的 fused 缓存文件。

```bash
# 在独占 GPU 上运行（共享 GPU 会导致测量误差）
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx1201.json \
ROCMLIR_EPILOGUE_FUSION=1 ROCM_FUSE_ATTENTION=1 \
ROCMLIR_ENABLE_TUNING_FUSED=1 \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM.0 -t 300 -nireq 1 -b 1

# 调优结果自动写入：~/.cache/ov_rocmlir_tuning_gfx1201_fused.json
# 典型输出（44 shapes，36 个找到更优配置）：
# [fused-tune] N=1 C=3 H=640 K=96 R=3 stride=2 → best=v3:128,64,4,64,32,... (0.024ms)
```

#### 步骤 C：正式 Benchmark

```bash
# 创建 hipGraph 配置文件
printf '{"ROCM": {"ROCM_USE_HIP_GRAPH": "YES"}}' > /tmp/rocm_hg.json

# 正式测试（fused 调优缓存自动加载，hipGraph 启用）
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx1201.json \
ROCMLIR_EPILOGUE_FUSION=1 ROCM_FUSE_ATTENTION=1 \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM.0 \
    -load_config /tmp/rocm_hg.json \
    -t 120 -nireq 8 -b 1

# 预期输出：
# Throughput: ~280 FPS
# Median latency: ~28ms
# Max latency: ~46ms（稳定，无 GPU fault）
```

#### 快速验证（跳过调优，约 5 分钟）

```bash
# 使用预置的调优缓存（如果有）
ROCMLIR_TUNING_CACHE=/path/to/ov_rocmlir_tuning_gfx1201.json \
ROCMLIR_EPILOGUE_FUSION=1 ROCM_FUSE_ATTENTION=1 \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM.0 \
    -load_config /tmp/rocm_hg.json \
    -t 60 -nireq 8 -b 1
# 预期：~240 FPS（无 fused 调优，仅使用 plain conv cache）
```

### 4.3 gfx950（MI350）复现

```bash
ROCMLIR_ENABLE_TUNING=1 \
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx950.json \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM.0 -t 180 -nireq 1 -b 1
# 预期调优后：~500 FPS
```

### 4.4 MIGraphX 对比参考

```bash
# MIGraphX 参考（需安装 ROCm 7.2+）
migraphx-driver perf --onnx /path/to/yolo26x.onnx \
    --fp16 --gpu --iterations 2000
# gfx1201 输出示例：Rate: 224 inferences/sec (4.46ms)
# gfx950 输出示例：Rate: 226 inferences/sec
```

### 4.5 逐 op 分析（找瓶颈）

```bash
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM.0 \
    -t 30 -nireq 1 -b 1 -pc \
    2>&1 | grep EXECUTED | python3 -c "
import sys, re
from collections import defaultdict
ops = defaultdict(lambda:[0,0.0])
for l in sys.stdin:
    m = re.search(r'layerType: (\S+)\s+execType: (\S+)\s+realTime .ms.:\s+(\S+)', l)
    if m:
        ops[m.group(1)][0]+=1; ops[m.group(1)][1]+=float(m.group(3))
total = sum(v[1] for v in ops.values())
print(f'Total: {total:.3f}ms')
for k,(c,t) in sorted(ops.items(),key=lambda x:-x[1][1])[:15]:
    print(f'  {c:4d}x {k:<35} {t:.3f}ms ({100*t/total:.1f}%)')
"
# gfx1201 典型输出：
# Total: 5.625ms
#  138x FusedConvolution               4.105ms (72.9%)
#    9x FusedGroupConvolution          0.340ms  (6.0%)
#   14x FusedElementwise               0.225ms  (4.0%)
#    6x MatMul                         0.087ms  (1.5%)
```

### 4.6 GPU kernel 级分析（rocprof）

```bash
# 用 rocprof 获取 GPU kernel 时间统计
rocprof --stats -o /tmp/kstats.csv \
  benchmark_app -m yolo26x.onnx -d ROCM.0 -t 5 -nireq 1 -b 1

# 分析 per-inference 时间（假设 516 inferences）
python3 -c "
import csv
rows = list(csv.DictReader(open('/tmp/kstats.csv')))
rows.sort(key=lambda r: -float(r['TotalDurationNs']))
n_inf = 516
print(f'Kernel breakdown (per inference):')
for r in rows[:10]:
    n = r['Name'].replace('.kd','')[:45]
    us = float(r['AverageNs'])/1000
    calls_inf = int(r['Calls'])/n_inf
    print(f'  {calls_inf:4.0f}x {n:<45} avg={us:.1f}µs')
"
# gfx1201 典型输出（yolo26x, fused tuning）：
# 102x mlir_convolution_broadcast_add_sigmoid_mul   avg=22-25µs  (调优前 28.9µs)
#  30x mlir_convolution_broadcast_add_sigmoid_mul_add avg=20-24µs
```

---

## 5. 性能相关环境变量

### 5.1 核心融合变量（默认已开启）

| 功能 | 状态 | 说明 |
|------|------|------|
| fused_epilogue dialect kernel | **默认 ON** | Conv+Bias+SiLU 单 kernel 融合。关闭：`ROCMLIR_EPILOGUE_FUSION=0` |
| Conv+Reshape 融合 | **默认 ON** | Conv→Reshape 融合进 kernel，消除独立 Transpose。关闭：`ROCMLIR_EPILOGUE_FUSION=0` |
| Swish no-op 消除 | **默认 ON** | 已 fuse 的 SiLU Swish 节点不进入 dispatch 队列（131 个 no-op） |
| Attention MatMul 融合 | **默认 ON** | QKV Attention 用 rocMLIR MLIR kernel。关闭：`ROCM_FUSE_ATTENTION=0` |
| HSACO 磁盘缓存 | **默认 ON** | 编译 kernel 持久化到 `~/.cache/ov_rocmlir_cache_<arch>/`，warm start 1.3s |
| hipGraph（可选） | **默认 OFF** | 通过 `load_config` JSON 开启，消除 CPU dispatch overhead |
| pe(V) 融合 | **默认 ON（gfx1201）** | pe(V) depthwise conv 与 AV 输出融合 |
| VariadicSplit 零拷贝 | **默认 ON** | QKV split buffer alias，零显存复制 |
| GroupConv Swish 消除 | **默认 ON** | GroupConv→Swish no-op（6 个 model.23 节点） |
| SliceConv Swish 消除 | **默认 ON** | FusedConvolutionSlice 后 Swish no-op（15 个节点） |

### 5.2 调优相关变量

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `ROCMLIR_ENABLE_TUNING` | 未设置 | 设为 `1` 启用 plain rock.conv 穷举 perf_config 调优（~50 shapes，10-40 分钟）。**需独占 GPU**。 |
| `ROCMLIR_TUNING_CACHE` | `~/.cache/ov_rocmlir_tuning_<arch>.json` | plain conv 调优缓存路径 |
| `ROCMLIR_ENABLE_TUNING_FUSED` | 未设置 | 设为 `1` 启用 **fused_epilogue 专项**调优（conv+bias+silu kernel，20-60 分钟）。结果额外提升 ~16% FPS。**需独占 GPU**。 |
| `ROCMLIR_TUNING_CACHE_FUSED` | `~/.cache/ov_rocmlir_tuning_<arch>_fused.json` | fused_epilogue 调优缓存路径（自动加载，无需设置） |

### 5.3 hipGraph 配置

hipGraph 通过 OpenVINO 配置文件（JSON）开启，不支持环境变量：

```bash
# 创建 hipGraph 配置文件
printf '{"ROCM": {"ROCM_USE_HIP_GRAPH": "YES"}}' > /tmp/rocm_hg.json

# 使用配置文件启动
benchmark_app -m model.onnx -d ROCM.0 -load_config /tmp/rocm_hg.json ...
```

**hipGraph 注意事项**：
- 首次运行会 capture 所有 kernel（Max latency 可能有 spike），后续稳定
- 与 fused_epilogue 调优完全兼容（fused cache 热启动后）
- 对 nireq=8 吞吐量模式效果有限（CPU dispatch 被 GPU 执行掩盖），
  主要价值是稳定 Max latency（204ms → 46ms）

### 5.4 回退与调试变量

| 环境变量 | 说明 |
|----------|------|
| `ROCMLIR_EPILOGUE_FUSION=0` | 关闭 fused_epilogue dialect，退回 rock dialect |
| `ROCM_FUSE_ATTENTION=0` | 关闭 Attention MatMul 融合，退回 rocBLAS |
| `ROCM_FUSE_PE=0` | 关闭 pe(V) conv 融合 |
| `ROCM_SWISH_INPLACE=0` | 关闭 Swish 原位写入优化 |
| `HIP_VISIBLE_DEVICES=N` | 多卡环境必须指定，避免内存冲突 |

### 5.5 最优配置一览

```bash
# 最优吞吐量（需 fused 调优缓存）
ROCMLIR_TUNING_CACHE=/path/to/ov_rocmlir_tuning_gfx1201.json \
ROCMLIR_EPILOGUE_FUSION=1 \
ROCM_FUSE_ATTENTION=1 \
benchmark_app -m model.onnx -d ROCM.0 \
    -load_config /tmp/rocm_hg.json \
    -t 120 -nireq 8 -b 1
# → ~280 FPS（gfx1201, yolo26x）

# 最低延迟（单流）
ROCMLIR_TUNING_CACHE=/path/to/ov_rocmlir_tuning_gfx1201.json \
ROCMLIR_EPILOGUE_FUSION=1 ROCM_FUSE_ATTENTION=1 \
benchmark_app -m model.onnx -d ROCM.0 \
    -load_config /tmp/rocm_hg.json \
    -hint latency -nireq 1 -b 1
# → ~4.9ms median（gfx1201, yolo26x, fused 调优后）
```

---

## 6. 各架构性能对比

### 6.1 yolo26x FP16 吞吐量（Throughput, batch=1, 120s）

| GPU | 架构 | OV（无调优） | OV（plain tuning） | OV（fused tuning + hipGraph） | MIGraphX |
|-----|------|------------|-------------------|------------------------------|---------|
| **Radeon AI PRO R9700** | **gfx1201 / RDNA4** | ~212 FPS | ~240 FPS | **~280 FPS ✅** | ~224 FPS |
| RX 7900 XTX | gfx1100 / RDNA3 | ~170 FPS | **~221 FPS ✅** | — | ~168 FPS |
| Instinct MI350 | gfx950 / CDNA3 | ~330 FPS | **~500 FPS ✅** | — | ~295 FPS |

> ✅ = 超过 MIGraphX

### 6.2 yolo26x FP16 单流延迟（nireq=1，fused 调优后）

| GPU | 架构 | OV 中位延迟 | OV 最小延迟 | MIGraphX 中位延迟 |
|-----|------|------------|------------|-----------------|
| Radeon AI PRO R9700 | gfx1201 | ~5.0 ms | ~4.9 ms | 4.44 ms |
| RX 7900 XTX | gfx1100 | ~6.5 ms | — | ~6.0 ms |
| Instinct MI350 | gfx950 | ~2.5 ms | — | ~3.4 ms |

### 6.3 gfx1201 GPU kernel 时间（rocprof, nireq=1）

| Kernel | 调用次数/inference | 调优前 avg | 调优后 avg | 总时间 |
|--------|-----------------|-----------|-----------|--------|
| mlir_conv_bias_silu（fused） | 102 | 28.9µs | ~22-25µs | ~2.3ms |
| mlir_conv_bias_silu_add | 30 | 26.3µs | ~20-24µs | ~0.65ms |
| mlir_conv_bias_silu（slice） | 15 | 12.5µs | 12.5µs | 0.19ms |
| FusedElementwise（3-input add） | 14 | 8.1µs | 8.1µs | 0.11ms |
| **总 GPU kernel 时间** | | **4.58ms** | **~3.8ms** | |

> fused_epilogue 调优把主 conv kernel 平均时间从 28.9µs 降至约 22-25µs（-15% ~ -20%）

### 6.4 MIGraphX 参考（gfx1201）

```
Rate: 224.2 inferences/sec
GPU kernel time: ~4.46ms (rocprof measurement)
Conv kernels: 80 + 30 = 110 launches/inference
```

---

## 7. 优化历程与技术细节

### 7.1 gfx1201 性能优化全历程（yolo26x FP16）

| 阶段 | FPS | 关键改动 | commit |
|------|-----|---------|--------|
| 初始基线 | 104 | FusedConvolution 路由错误 | — |
| +FusedConv 修复 | 141 | `consumers_count` + `Reshape(Constant)` 修复 | — |
| +WMMA auto-select | 157 | gfx12xx 返回 `""` 自动选 WMMA | — |
| +Tuning（v3 候选集） | 181 | `ROCMLIR_ENABLE_TUNING=1` | — |
| +Transpose/Tile 预分配 | **212** | 消除推理时动态 `hipMalloc`（-1.5ms/inf） | — |
| **+fused_epilogue dialect 默认** | **233** | Conv kernel 换用 MIGraphX MLIR 编译路径（-2.1ms/inf） | `d61b8449` |
| +rocmlir-driver path fix | 235 | torch_ning 机器 rocmlir-driver 路径修复 | `584735a0` |
| +HSACO disk cache（6 条路径） | 236 | compile time 15s→1.3s，Max latency 改善 | `191f10d4` |
| +GroupConv pass 顺序 | 237 | FuseGroupConvBiasAdd 提前，+6 Swish 消除 | `191f10d4` |
| +SliceConv Swish 消除 | 238 | FusedConvolutionSlice Swish no-op，+15 消除 | `191f10d4` |
| **+hipGraph（pinned pool）** | **240** | per-request hipHostMalloc，Max latency 204ms→65ms | `02baea7c` |
| +fused_epilogue 命名规范 | 241 | migraphx→fused_epilogue 内部命名（无功能变化） | `a5d4585f` |
| +perf_config 注入（rocMLIR patches） | 241 | TosaToRock features 修复 + tuning-fallback | `94bbc729` |
| **+fused_epilogue 专项调优** | **277 (无 hipGraph)** | ROCMLIR_ENABLE_TUNING_FUSED=1，44 shapes 调优 | `27978b71` |
| **fused_epilogue + hipGraph** | **~280** | 稳定（HSACO cache warm 后） | — |
| MIGraphX 参考 | 224 | — | — |

### 7.2 主要优化技术说明

#### A. fused_epilogue dialect 默认启用（+10% FPS）

**问题**：rock dialect 的 `patch_ir_bias_silu` 用 `linalg.generic` 作为 epilogue，
Conv 输出经过中间 `memref.alloc`（GPU 内存分配）才能做 bias+silu，
产生额外的内存读写。

**解决方案**：
- `migraphx.convolution` + 内联 `sigmoid`/`mul` 算子，单 kernel 完成
- GPU 编译器将 epilogue 放在寄存器中，无中间 alloc
- `MarkConvReshapeEpiloguePass` 额外将 Conv→Reshape（Attention 投影）融合进 kernel，
  消除 `/model.23/Transpose`（0.41ms → 0.02ms）

**内部命名**：代码中称为 `fused_epilogue`（非 MIGraphX 框架，是 rocMLIR 的一个 MLIR dialect）。

#### B. hipGraph 支持（Max latency -70%）

**实现**：per-request `hipHostMalloc` pinned pool，为以下 op 提供稳定 H2D source：
- `FusedElementwiseOp`：aux_ptrs（多输入 Add 的 aux 地址数组）
- `VariadicSplitOp`：output_ptrs（各分支输出地址）
- `SplitOp`：output_ptrs

**效果**：
- Max latency：204ms → 46ms（消除冷启动 kernel 编译 spike）
- 吞吐量：+3 FPS（nireq=8 时 CPU overhead 被掩盖，收益小）
- 稳定性：连续 3 次 120s 测试 280 FPS，无 GPU fault

#### C. fused_epilogue 专项调优（+16% FPS）

**问题**：migraphx/fused_epilogue 路径（rocMLIR 的 MIGraphX dialect 到 rock.conv 转换）
之前没有经过 perf_config 调优，使用 `rock-affix-params` 的默认 heuristic tile 配置。

**三层 rocMLIR patch 实现**：
1. **`TosaToRock.cpp`（patch #2）**：修复 `ForwardConvConverter` 不传 `features` 给
   `rock.conv` 的 bug。修复后 `rock-affix-params` 能选择 WMMA 优化的 tile 配置。
2. **`rocmlir-driver.cpp`（patch #3）**：添加 `--tuning-fallback=true` 选项，
   注入的 perf_config 非法时自动 fallback 到 heuristic，不会导致 "Lowering failed"。
3. **OV 插件（`rocmlir_compiler.cpp`）**：`generate_fused_epilogue_ir()` 接受 `perf_cfg`
   参数，注入到 `migraphx.convolution {perf_config="..."}`，经过
   `MIGraphXToTosa → TosaToRock` 传播到 `rock.conv`。

**调优搜索**（`time_perf_config_fused_epilogue`）：
- 编译 conv+bias+silu fused MLIR + GPU 实际执行测时（hipEvent）
- 搜索 20 个预选 v3 候选 config（针对 RDNA4 wave32 优化）
- 结果存入独立 fused cache 文件

**效果**（yolo26x gfx1201，44 shapes）：
- 36/44 shapes 找到比默认更好的 config
- 平均 conv+silu kernel 时间：28.9µs → 22-25µs（-15% ~ -20%）
- 整体吞吐：238 FPS → 277 FPS（+16.3%）

#### D. HSACO 磁盘缓存扩展

**问题**：原先只有 SliceConv 和 SliceOut 路径有磁盘缓存，
其他 6 条编译路径每次进程启动都重新编译（约 15s）。

**修复**：为 `compile_fused_conv_bias`、`compile_fused_conv_bias_act`、
`compile_fused_conv_bias_silu_add`、`compile_conv_fused_epilogue`（3 variants）、
`compile_conv_fused_reshape`、`compile_conv_fused_skip` 添加 load/save 逻辑，
meta 文件格式扩展加入 `flags`（bias/silu/skip_add 标志位）。

**效果**：warm start 编译时间从 ~15s 降至 ~1.3s，Max latency spike 消失。

### 7.3 内存使用分析

| 组件 | OV | MIGraphX |
|------|-----|---------|
| 中间张量（mutable blob） | **43 MB**（MemorySolver coloring） | 54 MB（scratch） |
| 模型权重（const） | 106 MB | ~50 MB |
| HSACO GPU code segment | ~300 MB（~150 kernels） | ~0 MB（嵌入 .so） |

### 7.4 GPU kernel 效率对比（rocprof 实测）

```
OV GPU kernel 时间（fused 调优后）：~3.8ms/inference
MIGraphX GPU kernel 时间：~4.46ms/inference
→ OV GPU kernel 效率领先 MIGraphX ~15%
```

主要差距来自 OV 有 152 次 kernel launch（vs MIGraphX 115 次），
但每个 kernel 的计算效率已接近或超过 MIGraphX。

---

## 8. 附录：常见问题

### rocMLIR 构建失败：`_GLIBCXX_USE_CXX11_ABI` 宏重复

**原因**：HIP cmake 配置的 `INTERFACE_COMPILE_DEFINITIONS` 重复。

**修复**：
```bash
cd /home/rocMLIR/build
python3 -c "
import re, shutil
shutil.copy('build.ninja', 'build.ninja.bak')
with open('build.ninja') as f: content = f.read()
fixed = re.sub(r' -D_GLIBCXX_USE_CXX11_ABI=\\\"[^\\\"]*\\\"', '', content)
count = len(re.findall(r' -D_GLIBCXX_USE_CXX11_ABI=\\\"', content))
with open('build.ninja', 'w') as f: f.write(fixed)
print(f'Fixed {count} occurrences')
"
ninja -j$(nproc) rocmlir-driver rocmlir-gen
```

### Memory Access Fault（多 GPU 环境）

```
Memory access fault by GPU node-X on address ...
```

**解决**：
```bash
rocm-smi --showmeminfo vram   # 查看各 GPU VRAM 占用
HIP_VISIBLE_DEVICES=N benchmark_app ...  # 指定空闲 GPU
```

### fused_epilogue 调优期间 "Lowering failed"

**原因**：rocMLIR patches 未应用（缺少 `0002` 和 `0003` patch）。

**解决**：重新应用所有 patches 并重新构建 rocmlir-driver。

### hipGraph 首次运行 Max latency 高（200ms+）

**原因**：HSACO disk cache 冷启动，kernel 编译在推理中进行。

**解决**：
```bash
# 先不开 hipGraph 热身一次（让 HSACO cache 填满）
ROCMLIR_TUNING_CACHE=... benchmark_app -m model.onnx -d ROCM.0 -t 30 -nireq 8

# 再开 hipGraph
benchmark_app -m model.onnx -d ROCM.0 -load_config /tmp/rocm_hg.json -t 120 -nireq 8
```

### 调优缓存使性能下降

**原因**：在共享 GPU 上调优，测量误差导致选了次优配置。

**解决**：
```bash
# 删除并重新在独占 GPU 上调优
rm ~/.cache/ov_rocmlir_tuning_gfx1201.json
rm ~/.cache/ov_rocmlir_tuning_gfx1201_fused.json
ROCMLIR_ENABLE_TUNING=1 benchmark_app -m model.onnx -d ROCM.0 -t 300 -nireq 1
ROCMLIR_ENABLE_TUNING_FUSED=1 benchmark_app -m model.onnx -d ROCM.0 -t 300 -nireq 1
```

### 验证 fused_epilogue 是否生效

```bash
benchmark_app -m model.onnx -d ROCM.0 -t 5 -nireq 1 \
    2>&1 | grep -E "FusedEpilogue-cache|BiasIR-cache" | head -3
# 应看到：[FusedEpilogue-cache] loaded skip=0 silu_add=0 kernel=mlir_convolution_...
```

### 验证 fused 调优缓存已加载

```bash
benchmark_app -m model.onnx -d ROCM.0 -t 5 -nireq 1 \
    2>&1 | grep "fused-tune"
# 应看到：[fused-tune] Loaded N configs from ...
```

### cmake 找不到 cmake 二进制

```bash
pip install cmake
which cmake   # 应在 /opt/venv/bin/cmake 或 ~/.local/bin/cmake
```
