# OpenVINO ROCm Plugin — 性能指南

本文档面向希望在 AMD GPU（gfx1201/RDNA4、gfx1100/RDNA3、gfx950/CDNA3 等）上
使用 OpenVINO ROCm Plugin 进行推理性能调优的工程师。

---

## 目录

1. [环境依赖](#1-环境依赖)
2. [rocMLIR 编译安装](#2-rocmlir-编译安装)
3. [构建步骤](#3-构建步骤)
4. [Benchmark 复现 — yolo26x（视觉）](#4-benchmark-复现)
5. [Benchmark 复现 — bertsquad-12（Transformer）](#5-benchmark-复现--bertsquad-12transformer)
6. [性能相关环境变量](#6-性能相关环境变量)
7. [各架构性能对比](#7-各架构性能对比)
8. [优化历程与技术细节](#8-优化历程与技术细节)
9. [附录：常见问题](#9-附录常见问题)
10. [BERT/Transformer 模型优化](#10-berttransformer-模型优化)
11. [Triton Flash Attention](#11-triton-flash-attention)

---

## 1. 环境依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| OS | Ubuntu 22.04 / 24.04 (64-bit) | |
| ROCm | **7.2.x** | 含 HIP、MIOpen、rocBLAS |
| GPU | gfx1201 / gfx1100 / gfx950 | 见下表 |
| Python | 3.10+ | 用于调优脚本 |
| Triton | >= 3.5.0 | Flash Attention AOT 编译（`pip install triton`） |
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
| `0004-lds-bank-conflict-fix-f16-kpack.patch` | 修复 `loweringUtils.cpp`：f16 kpack>1 时 LDS stride 从 1 改为 bankWidth/elemBytes，避免相邻 lane 命中同一 4-byte bank | 推荐（gfx1100 WMMA ~3-5%） |
| `0005-wmma-vgpr-occupancy-filter.patch` | `GridwiseGemmParams.cpp`：估算 VGPR 用量，拒绝 occupancy<50% 的 tuning config（gfx1100 wave32, 1536 VGPRs/SIMD） | 推荐（避免低占用率 config） |
| `0006-wmma-occupancy-downgrade-and-b-prefetch.patch` | 两项 WMMA 优化：(1) double-buffer VGPR 超标时降级为 single-buffer (2) nRepeats<=4 时预取所有 B tile 再做 WMMA，分离 ds_load 和 v_wmma | 推荐（gfx1100 +5-7%） |

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

## 5. Benchmark 复现 — bertsquad-12（Transformer）

### 5.1 模型准备

```bash
# 下载 bertsquad-12.onnx（BERT-Base for SQuAD，TensorFlow 导出格式）
pip install onnx_models  # 或从 ONNX Model Zoo 获取
# 模型路径示例：/tmp/bert_model/bertsquad-12.onnx

# 验证模型输入
python3 -c "
import onnx
m = onnx.load('/tmp/bert_model/bertsquad-12.onnx')
for inp in m.graph.input:
    print(inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim])
"
# 输出示例：
# unique_ids_raw_output___9:0 [1]
# segment_ids:0 [1, 256]
# input_mask:0  [1, 256]
# input_ids:0   [1, 256]
```

### 5.2 复现 291 FPS（gfx1201）

bertsquad-12 是动态形状模型，必须通过 `-shape` 指定输入尺寸。
**无需 fused_epilogue 调优**，BERT 性能来自 Transformer-specific fusion pass，均在图变换阶段完成。

```bash
export OV_BIN=/home/openvino/bin/intel64/Release
export BERT_MODEL=/tmp/bert_model/bertsquad-12.onnx

# 标准复现命令（预期 ~291 FPS，median latency ~3.43ms）
${OV_BIN}/benchmark_app \
  -m ${BERT_MODEL} \
  -d ROCM \
  -niter 400 \
  -nireq 1 \
  -hint none \
  -api sync \
  -shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]"
```

**预期输出：**
```
[ INFO ] Throughput:   291.17 FPS
[ INFO ] Latency:
[ INFO ]    Median:    3.43 ms
[ INFO ]    Average:   3.43 ms
[ INFO ]    Min:       3.35 ms
[ INFO ]    Max:       3.81 ms
```

### 5.3 验证 Transformer Fusion Pass 是否生效

推理时会在 stderr 输出 fusion 日志。通过以下命令验证：

```bash
${OV_BIN}/benchmark_app \
  -m ${BERT_MODEL} -d ROCM -niter 2 -nireq 1 -hint none -api sync \
  -shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]" \
  2>&1 | grep -E "BertAttn|LayerNorm|FusedLN|FusedMasked|FusedFC"
```

**应看到（截取关键行）：**
```
[BertAttnFuse]  Running on 12 Softmax nodes
[BertAttnFuse]  Fused layer: seq=256 heads=12 dim=64    ← ×12 行
[LayerNormFusion] TF-style fused at axis=2 eps=1e-12 inner=768  ← ×25 行
[FusedLNPass]   Fused: rank=3 rows=256 hidden=768 axis=2        ← ×25 行
[BertAttn] Constructor: seq=256 heads=12 dim=64 bias_elems=65536 ← ×12 行
```

**关键指标：**

| Fusion | 期望数量 | 说明 |
|--------|---------|------|
| BertSelfAttention | 12 | 12 层 encoder attention，各编译为独立 rock.attention kernel |
| TF LayerNorm (LayerNormFusion) | 25 | 1 embeddings + 12×2 encoder，全部识别 |
| FusedLayerNorm (FusedLNPass) | 25 | 25 个 LayerNorm → native f16 HIP kernel |
| FusedMaskedSoftmax | 0 | 被 BertSelfAttentionFusion 消费，不剩余 standalone softmax |

### 5.4 与 MIGraphX 对比

```bash
# MIGraphX 参考（需安装 MIGraphX 2.15+）
migraphx-driver perf --gpu ${BERT_MODEL} \
  --input-dim @unique_ids_raw_output___9:0 1 \
  --input-dim @segment_ids:0 1 256 \
  --input-dim @input_mask:0 1 256 \
  --input-dim @input_ids:0 1 256
# gfx1201 预期：~194 FPS（约 5.14ms）
# OV 约快 50%
```

### 5.5 rocprof Kernel 分析

```bash
rocprof --stats -o /tmp/bert_profile.csv \
  ${OV_BIN}/benchmark_app \
  -m ${BERT_MODEL} -d ROCM -niter 100 -nireq 1 -hint none -api sync \
  -shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]" \
  2>/dev/null

# 按总 GPU 时间排序，查看 top kernels
sort -t, -k4 -rn /tmp/bert_profile.stats.csv | head -15
```

**gfx1201 典型 kernel 分布（per inference）：**

| Kernel | 调用/inference | 占比 | 说明 |
|--------|--------------|------|------|
| GEMM Tensile (FC layers) | ~60 | ~25% | 12层×Q/K/V/Out+FFN×2 矩阵乘 |
| broadcast_bias_add_h2 | ~49 | ~11% | FC bias add（__half2 向量化） |
| rock_attention | 12 | ~5% | 12层 fused attention kernel |
| layernorm_fused_kernel | 25 | ~6% | native f16 LayerNorm |
| masked_softmax_fused_kernel | 0 | 0% | 已融入 rock_attention |
| bias_gelu_fused | 12 | ~2% | FFN GELU activation |

### 5.6 常见问题

**问：模型加载时报 "Dynamic models are not supported"**

```bash
# 必须通过 -shape 指定所有输入的静态尺寸
-shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]"
```

**问：BertSelfAttentionFusion 报 "0 Softmax nodes found"**

Pass 顺序错误导致 MaskedSoftmaxPass 先消费了 Softmax 节点。确认代码中 pass 顺序为：
`BertSelfAttentionFusion` → `FusedMaskedSoftmaxPass`（参见 `rocm_graph_transformer.cpp`）。

**问：LayerNormFusion 只匹配 1/25**

OV 的 `ReshapeSinkingMatMul` 消除了 `[256,768]→[1,256,768]` Reshape 但未更新 ReduceMean axes。
确认 `layer_norm_fusion.cpp` 包含"axis fixup"逻辑（搜索 `gamma_size != inner_size`）。

**问：rock.attention 编译失败 "argument not found"**

`make_mlir()` 中 `fmt::format` 参数数量与占位符不匹配。确认 `#xbias2d` 相关的
dead code 已删除，`#xbias` 使用正确的参数数量（5 个 `{}`，传 5 个参数）。

---

## 6. 性能相关环境变量

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

## 7. 各架构性能对比

### 7.1 yolo26x FP16 吞吐量（Throughput, batch=1, 120s）

| GPU | 架构 | OV（无调优） | OV（plain tuning） | OV（fused tuning + hipGraph） | MIGraphX |
|-----|------|------------|-------------------|------------------------------|---------|
| **Radeon AI PRO R9700** | **gfx1201 / RDNA4** | ~212 FPS | ~240 FPS | **~280 FPS ✅** | ~224 FPS |
| RX 7900 XTX | gfx1100 / RDNA3 | ~170 FPS | **~221 FPS ✅** | — | ~168 FPS |
| Instinct MI350 | gfx950 / CDNA3 | ~330 FPS | **~500 FPS ✅** | — | ~295 FPS |

> ✅ = 超过 MIGraphX

### 7.2 bertsquad-12 FP16 吞吐量（seq=256, batch=1, nireq=1, sync, 400 iters）

| GPU | 架构 | OV ROCm Plugin | MIGraphX | OV 优势 |
|-----|------|---------------|---------|---------|
| **Radeon AI PRO R9700** | **gfx1201 / RDNA4** | **291 FPS ✅** | ~194 FPS | **+50%** |

**OV bertsquad-12 启用的 fusion（无需手动配置，全部默认开启）：**

| Fusion | 数量 | GPU 时间贡献 |
|--------|------|------------|
| `rock_attention`（BertSelfAttention） | 12 | ~5% |
| `layernorm_fused_kernel`（native f16） | 25 | ~6% |
| `bias_gelu_fused_kernel`（FC+GELU） | 12 | ~2% |
| rocBLAS GEMM（FC layers） | ~60 | ~25% |

**OV bertsquad-12 性能演进（gfx1201）：**

| 阶段 | FPS | 关键改动 |
|------|-----|---------|
| 初始（crash） | — | 多处 type-cast bug |
| Bug 修复后 | 136 | i32/f32 type-safe cast，MatMul transpose 恢复 |
| FullyConnected fusion | 183 | FC pattern matcher 修复（3 处 bug） |
| FusedFCGELU | 204 | FC+GELU → native HIP kernel |
| FusedMaskedSoftmax | 222 | Add+Softmax → native f16 kernel |
| MIGraphX 依赖移除 | 196 | 全替换为 native HIP + rocBLAS（重构期间） |
| BertSelfAttentionFusion | 208 | rock.attention fused attention kernel |
| TF LayerNorm 25/25 | 269 | axis fixup + 全匹配 native f16 kernel |
| bias-add __half2 向量化 | **291** | v_pk_add_f16 指令，2× bias kernel 速度 |
| **MIGraphX 参考** | 194 | — |

### 7.3 yolo26x FP16 单流延迟（nireq=1，fused 调优后）

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

## 8. 优化历程与技术细节

### 8.0 BERT/Transformer 优化技术详解

#### A. BertSelfAttentionFusion（最大单项收益，+33 FPS）

**目标**：将 12 层 encoder attention 的 Q×K^T → mask_add → softmax → ×V 融合为单 kernel。

**实现**：`BertSelfAttentionFusion` pass 从 Softmax 节点（v1/v8）反向追踪 Q、K、V 的 FullyConnected 输出，向前追踪 AV matmul → transpose → reshape，构建 `BertSelfAttention` 自定义 OV node。

`BertSelfAttentionOp` 在构造时用 rocMLIR 生成 `rock.attention` MLIR 并通过 `rocmlir-driver --kernel-pipeline=full` 编译为 HSACO：

```mlir
// 核心结构（seq=256, heads=12, dim=64）
rock.attention {
  qk = %Q * %K : memref<12x256x64xf16>, memref<12x64x256xf16>
  qk = elementwise otherIns(%B : memref<12x256x256xf16>) { ... }  // bias add
  %O = softmax(qk) * %V : memref<12x256x64xf16> -> memref<12x256x64xf16>
} {numHeadsQ = 12, softmaxType = f32, ...}
```

**Bias broadcast 方案**：attention mask 为 `[1,1,256,256]` flat，通过 affine_map `(h,sq,sk) → sq*256+sk` 广播到 `[12,256,256]`，实际 buffer 只有 65536 个 f16 元素。

**Pass 顺序约束**：`BertSelfAttentionFusion` 必须在 `FusedMaskedSoftmaxPass` 之前执行，否则 Softmax 节点已被消费，attention pattern 无法匹配。

#### B. TF LayerNorm 全融合（+61 FPS，最大收益）

**问题根因**：TensorFlow 导出的 BERT 使用 pre-folded LayerNorm：`y = x*(γ/σ) + (β-μ*γ/σ)`，与标准 `(x-μ)/σ*γ+β` 不同。

**检测算法**（`LayerNormFusionPass`）：从每个 `ReduceMean` 节点出发，BFS 追踪：
`ReduceMean → [Convert]* → Sub(x, mean) → Mul(diff,diff) → ReduceMean → Add(eps) → Sqrt → Div(1,√) → Mul(γ) → [Mul(x,W), Mul(mean,W)] → Sub(β, mean*W) → Add(xW, B)`

**Axis fixup bug**：OV 的 `ReshapeSinkingMatMul` 消除了 `[256,768]→[1,256,768]` 的 Reshape，但 ReduceMean 的 `axes=[1]` 未更新，导致 `inner_size=256` 而 gamma 有 768 个元素。修复：检测到 gamma 元素数与 `x.shape[axis]` 不符时，用 gamma 元素数反推正确 axis（最后维度）。

**Native kernel**（`layernorm_fused_kernel`）：单 HIP kernel 完成 mean→variance→rsqrt→scale+shift，全程在寄存器/shared memory，支持可选 residual skip add：

```
block_reduce_sum(x+skip) → mean
block_reduce_sum((x+skip-mean)²) → var
inv_std = rsqrt(var/cols + eps)
y[i] = ((x[i]+skip[i]) - mean) * inv_std * gamma[i] + beta[i]
```

消除了原来的 f16→f32→MIOpen→f32→f16 路径，节省约 25% GPU 时间（16.2% convert + 8.5% f32 subtract）。

#### C. FusedFCGELU（FC+GELU 两步融合）

**实现**：`FusedFCGELUPass` 识别 `FullyConnected → GELU` 模式。`FusedFCGELUOp::Execute` 分两步：
1. `rocblas_gemm_ex`：`gemm_buf = x × W^T`（f16，构造时预分配 `gemm_buf`）
2. `bias_gelu_fused_kernel`：`out[i] = (gemm_buf[i] + bias[col]) * 0.5 * (1 + erf(... * 0.7071))`

**Pass 注意**：BERT 的 W 矩阵以 `[out_dim, in_dim]` 存储（transpose FC），`out_dim` 从 GELU 输出 shape 读取，不能从 W shape 读（方向可能相反）。

#### D. bias-add __half2 向量化

BERT FC 的 bias add 占总 GPU 时间 11%（49 次/inference）。原始标量 kernel：
```cpp
out[idx] += bias[idx % cols];  // 1 f16/thread
```

改为 `__half2` 向量化，每线程处理 2 个 f16（BERT hidden=768/3072 均为偶数）：
```cpp
out_h2[idx] = __hadd2(out_h2[idx], bias_h2[col2]);  // v_pk_add_f16
```

### 8.1 gfx1201 性能优化全历程（yolo26x FP16）

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

## 9. 附录：常见问题

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


---

## 10. BERT/Transformer 模型优化

### 10.1 性能成就（gfx1201，FP16）

| 配置 | OV ROCm Plugin | MIGraphX FP16 | 比较 |
|---|---|---|---|
| **bertsquad-12（nireq=2）** | **~774 FPS** | ~700 FPS | **OV 快 10.6%** |
| bertsquad-12（nireq=1） | ~723 FPS | ~700 FPS | OV 快 3.3% |
| **yolo26x（吞吐量模式）** | **~280 FPS** | ~224 FPS | OV 快 25% |

测试条件：batch=1，seq_len=256，gfx1201/RDNA4，FP16，bertsquad-12.onnx，1000 次迭代

### 10.2 复现最优性能（774 FPS）

#### 环境要求

```bash
# 额外依赖（bertsquad-12 FP16 优化所需）
# hipBLASLt: ROCm 7.2 自带，无需单独安装
# hipRTC: ROCm 7.2 自带（/opt/rocm/lib/libhiprtc.so）
ls /opt/rocm/lib/libhipblaslt.so /opt/rocm/lib/libhiprtc.so
```

#### 模型准备

```bash
# 下载 bertsquad-12.onnx（来自 ONNX Model Zoo）
mkdir -p /tmp/bert_model
# 或使用现有模型路径
ls /tmp/bert_model/bertsquad-12.onnx
```

#### 一键复现

```bash
export OV_BIN=/home/openvino/bin/intel64/Release
export BERT_MODEL=/tmp/bert_model/bertsquad-12.onnx

# 方式 A：最高吞吐量（nireq=2，~774 FPS）
${OV_BIN}/benchmark_app \
  -m ${BERT_MODEL} -d ROCM \
  -niter 800 -nireq 2 -hint none \
  -shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]"

# 方式 B：最低单流延迟（nireq=1，~723 FPS，1.32ms）
${OV_BIN}/benchmark_app \
  -m ${BERT_MODEL} -d ROCM \
  -niter 800 -nireq 1 -hint none -api sync \
  -shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]"
```

> **注意**：首次推理会触发 hipRTC JIT 编译（约 1-2 秒），之后性能稳定。

#### 验证融合是否生效

```bash
${OV_BIN}/benchmark_app \
  -m ${BERT_MODEL} -d ROCM -niter 2 -nireq 1 -hint none -api sync \
  -shape "unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]" \
  2>&1 | grep -E "FusedLN|BertAttn|QKV|FusedFC|LayerNorm|JIT"
```

预期输出（关键行）：
```
[BertAttnFuse] Fused layer: seq=256 heads=12 dim=64       <- x12 (attention fusion)
[LayerNormFusion] 3-input FusedLN (src+bias+residual): rows=256 hidden=768  <- x24
[FusedLNPass] Plain LN: rows=256 hidden=768               <- x1 (embedding LN)
[FusedLN-JIT] Compiled rows=256 cols=768 ...              <- hipRTC kernel
[FusedQKVPass] Fused Q+K+V: seq=256 hidden=768            <- x12 (QKV fusion)
[FusedFCGELUPass] Fused FC+GELU: seq=256 ...              <- x12 (FFN fusion)
```

### 10.3 BERT 优化全历程（gfx1201）

| 阶段 | FPS | 关键技术 |
|------|-----|---------|
| 初始（crash） | -- | 多处 type-cast bug |
| Bug 修复后 | 136 | i32/f32 cast，MatMul transpose 恢复 |
| FC Fusion + BugFix | 183 | FullyConnectedTransformation 修复 |
| FC+GELU + Masked Softmax | 222 | FusedFCGELU，FusedMaskedSoftmax |
| BertSelfAttention Fusion | 208 | rock.attention JIT kernel |
| LayerNorm 25/25 全融合 | 291 | TF pre-folded 模式识别 + axis fixup |
| EliminateF16ToF32Convert | **376** | 最大单次：消除 28 个 f16->f32 转换 |
| hipBLASLt GEMM+bias | 411 | 单 kernel 替代双 kernel |
| QKV fusion（256x2304x768） | 421 | 3 个 FC 合并为 1 个大 GEMM |
| FusedFCGELU hipBLASLt | 482 | GELU_BIAS epilogue |
| TF-style GELU 匹配 | 521 | Pow->Mul->Add->Tanh 链识别 |
| Gather 47x加速 + nireq=2 | 552 | half2 向量化 warp-per-row gather |
| rocMLIR vs hipBLASLt 选择 | 588 | 每种 shape 自动 benchmark 选最优 |
| layernorm_fused_vec2 | 646 | half2 vectorized，warp-per-row |
| hipRTC 寄存器缓存 kernel | **702** | 零第二次 HBM 读取，3.3us/LN |
| **3-input FusedLayerNorm** | **~774** | LN(src+bias+residual) 单 pass，消除 FusedEW |

### 10.4 关键优化技术说明

#### A. 3-input FusedLayerNorm（最终大招）

**原理**：BERT 的 FusedElementwise 链为 `Add(src, bias) -> Add(result, residual)`。直接把两个 Add 都吸入 FusedLayerNorm，hipRTC kernel 在寄存器内完成：

```
out = LayerNorm(src + bias + residual) * gamma + beta
```

单 kernel pass，数据从 HBM 读取一次，缓存在寄存器，归一化后直接输出。

- **之前**：FusedEW(4.5us) + FusedLN(3.3us) = 7.8us x 24 层 = 187us
- **之后**：3-input JIT LN(~5.7us) x 24 层 = 137us
- 节省：50us/inference，消除 23 个 kernel launch

#### B. hipRTC JIT 寄存器缓存 kernel

BERT hidden=768，每 warp（32 线程）处理 1 行（12 个 half2 元素/线程）：

```cuda
// Pass 1: load src+bias+skip -> REGISTER CACHE, compute Welford
__half2 val[12];  // register cache
for (int i = 0; i < 12; i++) {
    val[i] = src[lane+i*32] + bias[lane+i*32] + skip[lane+i*32];
    // Welford online reduction (no cross-warp sync)
}
// Warp-only reduce (zero shared memory!)
mean = warp_sum(sum) / N;
inv_std = rsqrt(variance + eps);
// Pass 2: normalize from REGISTER (zero HBM read!)
for (int i = 0; i < 12; i++) {
    out[...] = (val[i] - mean) * inv_std * gamma[...] + beta[...];
}
```

vs 之前的 vec2 kernel（需 2 次 HBM 读），JIT 寄存器版本约快 1.5x（4.8us->3.3us）。

#### C. EliminateF16ToF32Convert

OV ConvertPrecision 在激活张量上插入 `Convert(f16->f32)` "解压" 节点，导致 FusedElementwise 跑 f32 kernel（2x 内存带宽浪费）。新增 Pass 精准删除这 28 个节点，让 FusedElementwise 全走 f16 path。

#### D. BertSelfAttentionFusion

用 rocMLIR `rock.attention` 把 Q*K^T + mask + softmax + *V 融合为单 kernel：

```mlir
rock.attention {
  qk = %Q * %K   // Q[heads,seq,dim] x K[heads,dim,seq]
  qk = elementwise(add_mask)  // fused mask add
  out = softmax(qk) * %V     // flash-attention style
} {numHeadsQ=12, softmaxType=f32}
```

偏置 `[1,1,256,256]` 通过 affine_map `(h,sq,sk)->sq*256+sk` 广播到 `[12,256,256]`，无额外内存。

### 10.5 性能调优技巧

#### 首选 nireq=2

对于 BERT seq=256，nireq=2 是最优配置：GPU 利用率从 74% 提升到 ~90%，吞吐量约提升 7%。

```bash
# 高吞吐量配置
benchmark_app -m bert.onnx -d ROCM -nireq 2 -hint none -shape ...
```

#### 预热后再测

首次推理包含 hipRTC JIT 编译（~1-2s）和 hipBLASLt algo 选择（~2-3s）。测试前至少运行 5+ 次迭代预热：

```bash
# 先预热，再正式测试
benchmark_app ... -niter 5  # 预热
benchmark_app ... -niter 800  # 正式测试
```

#### 不使用 INT8

bertsquad-12-int8.onnx 使用 `MatMulInteger` + `FusedMatMul` 格式，当前 OV ROCm 和 MIGraphX 均不支持，建议使用 FP16 版本。

---

## 11. Triton Flash Attention

对于包含长序列 self-attention 的模型（如 petr_large Sq=41760、bevformer_v2 Sq=10000），
标准的 `MatMul(QK) → Softmax → MatMul(AV)` 路径会产生 O(Sq²) 的 score matrix 中间 tensor，
可能导致显存不足（petr_large 单个 score matrix `[8, 41760, 41760]` = 26.6GB）。

Plugin 内置了基于 Triton 的 Flash Attention fusion pass，将 QK·Softmax·AV 融合为单个 kernel，
**不 materialize score matrix**，显存占用从 O(Sq²) 降至 O(Sq)。

### 11.1 依赖

| 组件 | 版本 | 安装 |
|------|------|------|
| **Triton** | >= 3.5.0 | `pip install triton` 或使用 ROCm 配套版本 |
| torch | (可选) | 仅 autotune 需要；无 torch 时使用 heuristic 默认配置 |

> **不需要** `flash_attn` pip 包。Kernel 源码（来自 [dao-AILab/flash-attention](https://github.com/dao-ailab/flash-attention)，BSD-3 许可）已内嵌在 AOT 编译脚本中。

### 11.2 工作原理

```
模型加载时 (首次, ~30-75s one-time cost):
  FlashAttentionTritonFusionPass 匹配图中的 QK MatMul → Softmax → AV MatMul
    ↓ (仅 Sq > 512 && Sk > 512 时触发)
  C++ 调用 python3 triton_flash_attn_aot.py (AOT 编译)
    ├─ 生成搜索空间 (基于 headdim/LDS/寄存器约束)
    ├─ GPU microbenchmark 每个候选 config
    ├─ 选最快 → triton.compile() → HSACO (GPU 二进制)
    └─ 缓存到 ~/.cache/ov_triton_fa/

模型加载时 (后续, ~0ms):
    cache hit → 直接加载 HSACO

推理时:
    hipModuleLaunchKernel (纯 HIP, 无 Python 依赖)
```

### 11.3 Autotune 搜索空间

Flash attention kernel 有三个关键 tile 参数：

| 参数 | 含义 | 影响 |
|------|------|------|
| BLOCK_M | Q tile 行数 (输出 tile 高度) | 决定 grid 大小和寄存器压力 |
| BLOCK_N | K/V tile 行数 (K-loop 步长) | 决定 LDS 用量和 loop 次数 |
| num_warps | workgroup 内的 warp 数 | 决定线程级并行度 |

**Kernel 内部数据流：**

```
每个 workgroup 处理一个 [BLOCK_M, D] 的输出 tile:

  Q tile [BLOCK_M, D] ── 常驻寄存器 (整个 K-loop 不换)

  for k = 0 to Sk step BLOCK_N:        ← K-loop
      K tile [BLOCK_N, D] ── HBM → LDS
      V tile [BLOCK_N, D] ── HBM → LDS

      S = Q × K^T       [BLOCK_M, BLOCK_N]  ── 寄存器
      P = softmax(S)     [BLOCK_M, BLOCK_N]  ── 寄存器 (online softmax)
      O += P × V         [BLOCK_M, D]        ── 寄存器累加

  写回 O [BLOCK_M, D] → HBM
```

**硬件约束 (gfx1100 RDNA3, wave32)：**

| 约束 | 公式 | 说明 |
|------|------|------|
| LDS 容量 | `2 × BLOCK_N × D × 2B ≤ 64KB` | K tile + V tile 同时在 LDS |
| 寄存器压力 | `(BLOCK_M / num_warps) × D ≤ ~1024` | Q tile 分配到每个 warp |
| 线程合理性 | `num_warps × 32 ≤ BLOCK_M × BLOCK_N` | 不能比 tile 元素还多 |

**经验剪枝规则：**

| 规则 | 原因 |
|------|------|
| `BLOCK_M ≥ 128` 时 `warps ≥ 2` | 大 tile 需要多 warp 并行处理行 |
| `BLOCK_M ≤ 32` 时 `warps < 8` | 小 tile 不需要太多 warp |
| `BLOCK_M = 16` 时 `warps < 4` | 16 行 / 4 warp = 每 warp 4 行, 线程利用率低 |
| `BLOCK_N = 32` 且 `BLOCK_M ≤ 32` 时跳过 | K-loop 步长太小, loop overhead 大 |
| `D ≤ 32` 且 `BLOCK_M > 64` 时 `warps ≥ 4` | 小 D 下大 BM 的寄存器利用率低 |
| `D ≥ 128` 时 `BLOCK_M ≥ 32` | 大 D 需要大 tile 摊销 K/V 加载开销 |

**各 headdim 的搜索空间大小：**

| headdim | 候选 configs | autotune 耗时 |
|---------|-------------|---------------|
| D = 32 | ~25 | ~75s |
| D = 64 | ~17 | ~51s |
| D = 128 | ~5 | ~15s |

### 11.4 gfx1100 调优结果

以 bevformer_v2 的 attention shape (B=8, H=1, Sq=Sk=10000, D=32) 为例：

| Config | 耗时 (ms) | vs 默认 |
|--------|----------|---------|
| **BM=16, BN=64, w=1** | **0.07** | **baseline (最优)** |
| BM=32, BN=64, w=2 | 0.08 | +14% |
| BM=16, BN=128, w=2 | 0.09 | +29% |
| BM=64, BN=128, w=4 | 0.16 | +129% |
| BM=128, BN=128, w=4 | 0.32 | +357% |

**为什么 D=32 时 BM=16, BN=64, w=1 最快：**

- Q tile 只有 16×32 = 512 elements, 1 warp 完全放进寄存器, 零寄存器溢出 (spill)
- BN=64: K-loop 每步处理 64 个 key, LDS 仅用 2×64×32×2 = 8KB (64KB 的 12.5%)
- 1 warp = 32 线程, grid blocks = (Sq/16) × (B×H) = 625×8 = 5000 blocks → GPU 30 CUs 完全占满
- 对比 BM=128, BN=128, w=4: Q tile = 32×32 = 1024/warp 寄存器压力大, grid 仅 79×8 = 632 blocks, occupancy 低

**端到端效果：**

| 模型 | Sq×Sk | 默认 (128×128) | 调优后 (16×64) | 提升 |
|------|-------|---------------|----------------|------|
| bevformer_v2 | 10000×10000 | 6.4 FPS | **11.5 FPS** | **+80%** |
| petr_large | 41760×41760 | 5.2 FPS | **11.1 FPS** | **+114%** |

### 11.5 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ROCM_DISABLE_FLASH_ATTN` | 未设置 (启用) | 设为 `1` 禁用 flash attention fusion |
| `ROCM_TRACE_TRITON_FA` | 未设置 | 设为 `1` 打印 fusion 匹配和 kernel 加载信息 |
| `ROCM_TRITON_FA_SCRIPT` | (自动检测) | 覆盖 AOT 编译脚本路径 |

### 11.6 手动 AOT 编译 / 预热缓存

可在部署前预编译所有 attention shape，避免首次运行的 autotune 延迟：

```bash
SCRIPT=<plugin_src>/rocm/triton_flash_attn_aot.py

# bevformer_v2 的 attention shapes
python3 $SCRIPT --seqlen_q=10000 --seqlen_k=10000 --headdim=32 --arch=gfx1100
python3 $SCRIPT --seqlen_q=10000 --seqlen_k=8700  --headdim=32 --arch=gfx1100
python3 $SCRIPT --seqlen_q=900   --seqlen_k=10000 --headdim=32 --arch=gfx1100

# petr_large 的 attention shapes
python3 $SCRIPT --seqlen_q=41760 --seqlen_k=41760 --headdim=32 --arch=gfx1100

# 缓存位于 ~/.cache/ov_triton_fa/，可拷贝到目标机器
```

### 11.7 支持的 Attention Pattern

Fusion pass 匹配以下图结构：

```
          ┌─ Q ──┐
MatMul(QK)│      ├──→ [Add(bias)] ──→ Softmax ──→ MatMul(AV) ──→ Output
          └─ K ──┘    [Mul(scale)]                    └─ V ──┘

条件:
  - Softmax axis = last dim
  - Q/K/V 为 fp16
  - headdim (D) ≤ 128
  - Sq > 512 且 Sk > 512 (小 attention 不使用 flash)
  - Q layout: [B,H,Sq,D] 或 [B,Sq,H,D] 或 [BH,Sq,D]
```
