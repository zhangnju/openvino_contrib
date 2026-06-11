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
将 Conv+Bias+SiLU 等 epilogue 融合编译为 GPU kernel（HSACO），
性能大幅优于 MIOpen Immediate Mode。

**注意**：必须在 Plugin 构建之前安装好 rocMLIR。

### 2.1 选择 rocMLIR 版本

| ROCm 版本 | 推荐 rocMLIR 分支 |
|-----------|----------------|
| ROCm 7.2.x | `rocm-rel-7.2`（tag: `rocm-7.2.4`） |
| ROCm 7.0.x | `rocm-rel-7.0` |

### 2.2 编译（以 gfx1201 为例）

rocMLIR 依赖从源码构建的 LLVM/MLIR。使用 **g++**（而非 amdclang++）编译
host 代码以避免 GCC 13 头文件兼容性问题：

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
    -DGPU_TARGETS=gfx1201 \        # 修改为目标架构
    -DROCMLIR_DRIVER_E2E_TEST_ENABLED=0 \
    -DROCMLIR_DRIVER_PR_E2E_TEST_ENABLED=0 \
    -DROCMLIR_PARALLEL_LINK_JOBS=8 \
    -DROCMLIR_PARALLEL_COMPILE_JOBS=64 \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_PROJECTS="mlir;lld"

# 构建（约 30-60 分钟，96 核机器约 15 分钟）
ninja -j$(nproc) rocmlir-driver rocmlir-gen
```

### 2.3 应用 rocm_plugin patches

ROCm Plugin 在 `patches/` 目录下提供了针对 rocMLIR 的补丁，**在 cmake 之前 apply**：

```bash
ROCM_PLUGIN_DIR=/home/openvino/openvino_contrib/modules/rocm_plugin
ROCMLIR_DIR=/home/rocMLIR

# 应用所有 patch（按文件名顺序）
for patch in ${ROCM_PLUGIN_DIR}/patches/*.patch; do
    echo "Applying: $(basename $patch)"
    git -C ${ROCMLIR_DIR} am --3way < "$patch"
done
```

也可以单独 apply 某个 patch：

```bash
# 0001: 修复 AlignTiling.cpp NDEBUG 模式下的语法错误
# （std::ignore reasonCallback → std::ignore = reasonCallback）
git -C ${ROCMLIR_DIR} am --3way \
    < ${ROCM_PLUGIN_DIR}/patches/0001-rocmlir-fix-AlignTiling-build-error.patch
```

若 `git am` 因版本差异失败，可用 `git apply`：

```bash
git -C ${ROCMLIR_DIR} apply \
    ${ROCM_PLUGIN_DIR}/patches/0001-rocmlir-fix-AlignTiling-build-error.patch
```

> **patches 目录说明**：
> - `0001-rocmlir-fix-AlignTiling-build-error.patch`：修复 `AlignTiling.cpp` 在 g++ Release 构建时的语法错误（`std::ignore reasonCallback` 缺少赋值运算符，导致编译失败）。

**已知编译问题及额外修复**（若 patch 未能完全覆盖）：

1. **build.ninja 中 `_GLIBCXX_USE_CXX11_ABI` 宏重复**（HIP cmake config bug，cmake 之后执行）：
```bash
python3 -c "
import re, shutil
shutil.copy('build.ninja', 'build.ninja.bak')
with open('build.ninja') as f: content = f.read()
fixed = re.sub(r' -D_GLIBCXX_USE_CXX11_ABI=\\\"[^\\\"]*\\\"', '', content)
with open('build.ninja', 'w') as f: f.write(fixed)
print('Fixed', len(re.findall(r' -D_GLIBCXX_USE_CXX11_ABI=\\\"', content)), 'occurrences')
"
```

### 2.4 安装

```bash
mkdir -p /home/rocmlir_install/bin
cp /home/rocMLIR/build/bin/rocmlir-driver /home/rocmlir_install/bin/
cp /home/rocMLIR/build/bin/rocmlir-gen    /home/rocmlir_install/bin/

# 验证
/home/rocmlir_install/bin/rocmlir-driver --version
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
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \    # 修改为目标架构
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

### 4.1 环境变量设置

```bash
export PATH=/home/rocmlir_install/bin:$PATH
export LD_LIBRARY_PATH=${OPENVINO_HOME}/bin/intel64/Release:/opt/rocm/lib:$LD_LIBRARY_PATH
```

### 4.2 一键最优性能

**gfx1201（R9700）** — 目标 >220 FPS：

```bash
# 步骤1：生成 rocMLIR 调优配置（首次运行，约 10-30 分钟）
# 在独占 GPU 上运行（共享 GPU 会导致配置不准）
ROCMLIR_ENABLE_TUNING=1 \
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx1201.json \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint throughput -nireq 14 -t 180

# 步骤2：正式 Benchmark（使用缓存配置）
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx1201.json \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint throughput -nireq 14 -t 60

# 预期：>230 FPS（超过 MIGraphX 224 FPS）
```

**gfx950（MI350）** — 目标 >400 FPS：

```bash
ROCMLIR_ENABLE_TUNING=1 \
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx950.json \
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM -hint throughput -t 60
# 预期：~500 FPS
```

### 4.3 性能对比：OV vs MIGraphX

```bash
# MIGraphX 参考（需安装 ROCm 7.2+）
migraphx-driver perf --onnx /path/to/yolo26x.onnx \
    --fp16 --gpu --iterations 2000
# gfx1201 输出示例：Rate: 224 inferences/sec
```

### 4.4 逐 op 分析（找瓶颈）

```bash
${OPENVINO_HOME}/bin/intel64/Release/benchmark_app \
    -m /path/to/yolo26x.onnx -d ROCM \
    -hint throughput -nireq 1 -t 30 -pc \
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
    print(f'  {c:4d}x {k:<30} {t:.3f}ms ({100*t/total:.1f}%)')
"
```

---

## 5. 性能相关环境变量

### 5.1 核心融合变量（默认已开启，无需手动设置）

从当前版本起，以下优化**默认启用**，无需设置环境变量：

| 功能 | 状态 | 说明 |
|------|------|------|
| migraphx dialect kernel | **默认 ON** | Conv+Bias+SiLU 使用 MIGraphX MLIR 编译路径，比 v3: rock dialect 快 25%。关闭：`ROCMLIR_EPILOGUE_FUSION=0` |
| Conv+Reshape 融合 | **默认 ON** | Conv→Reshape 融合进 kernel，消除独立 Transpose（yolo26x 中节省 0.4ms）。关闭：`ROCMLIR_EPILOGUE_FUSION=0` |
| Swish no-op 消除 | **默认 ON** | 已 fuse 的 SiLU Swish 节点不进入 dispatch 队列（352→256 dispatches）。 |
| Attention MatMul 融合 | **默认 ON** | QKV Attention 用 MIGraphX MLIR kernel。关闭：`ROCM_FUSE_ATTENTION=0` |
| pe(V) 融合 | **默认 ON（gfx1201）** | pe(V) depthwise conv 与 AV 输出融合（gfx950 关闭，稳定性）。 |
| VariadicSplit 零拷贝 | **默认 ON** | QKV split 改为 buffer alias，零显存复制。 |
| Transpose/Tile 预分配 | **默认 ON** | 构造时预分配 stride 数组，消除推理中的动态 hipMalloc。 |

### 5.2 调优相关变量

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `ROCMLIR_ENABLE_TUNING` | 未设置 | 设为 `1` 启用穷举 perf_config 调优（约 50 个 shape）。首次运行耗时 10-30 分钟，结果持久化到缓存文件。**需独占 GPU**。 |
| `ROCMLIR_TUNING_CACHE` | `~/.cache/ov_rocmlir_tuning_<arch>.json` | 调优缓存路径，可指定绝对路径实现团队共享。 |

### 5.3 回退与调试变量

| 环境变量 | 说明 |
|----------|------|
| `ROCMLIR_EPILOGUE_FUSION=0` | 关闭 migraphx dialect，退回 rock dialect（v3: perf_config）和所有 epilogue pass。用于对比和调试。 |
| `ROCM_FUSE_ATTENTION=0` | 关闭 Attention MatMul 融合，退回 rocBLAS。 |
| `ROCM_FUSE_PE=0` | 关闭 pe(V) conv 融合。 |
| `ROCM_SWISH_INPLACE=0` | 关闭 Swish 原位写入。 |
| `ROCMLIR_CONV_SKIP_FUSION=0` | 关闭 conv+bias+skip 融合 kernel（migraphx dialect 下该配置不影响主路径）。 |
| `HIP_VISIBLE_DEVICES=N` | 多卡环境必须指定，避免内存冲突（Memory Access Fault）。 |

### 5.4 最优配置（参考）

```bash
# 当前版本默认即最优，无需额外环境变量
# 可选：指定已调优的缓存文件
ROCMLIR_TUNING_CACHE=/path/to/ov_rocmlir_tuning_gfx1201.json \
benchmark_app -m model.onnx -d ROCM -hint throughput -nireq 14

# 仅调试：关闭所有融合（验证基准性能）
ROCMLIR_EPILOGUE_FUSION=0 \
ROCM_FUSE_ATTENTION=0 \
ROCM_SWISH_INPLACE=0 \
benchmark_app -m model.onnx -d ROCM -hint throughput
```

---

## 6. 各架构性能对比

### 6.1 yolo26x FP16 吞吐量（Throughput）

| GPU | 架构 | OV（默认） | OV + 调优缓存 | MIGraphX |
|-----|------|-----------|--------------|---------|
| **Radeon AI PRO R9700** | **gfx1201 / RDNA4** | ~175 FPS | **~233 FPS ✅** | ~224 FPS |
| RX 7900 XTX | gfx1100 / RDNA3 | ~170 FPS | **~221 FPS ✅** | ~168 FPS |
| Instinct MI350 | gfx950 / CDNA3 | ~330 FPS | **~500 FPS ✅** | ~295 FPS |

> ✅ = 超过 MIGraphX 水平

### 6.2 yolo26x FP16 单流延迟（Latency hint，nireq=1）

| GPU | 架构 | OV 中位延迟 | MIGraphX 中位延迟 |
|-----|------|------------|-----------------|
| Radeon AI PRO R9700 | gfx1201 | ~5.5 ms | 4.44 ms |
| RX 7900 XTX | gfx1100 | ~7.0 ms | ~6.0 ms |
| Instinct MI350 | gfx950 | ~2.5 ms | ~3.4 ms |

### 6.3 MIGraphX 参考（gfx1201）

```
Rate: 224.2 inferences/sec
Total time: 4.44ms (Median), Min: 4.35ms, Max: 4.66ms
```

---

## 7. 优化历程与技术细节

### 7.1 gfx1201 性能优化全历程（yolo26x FP16）

| 版本 | FPS | 关键改动 |
|------|-----|---------|
| 初始基线 | 104 FPS | FusedConvolution 路由错误 |
| +FusedConv 修复 | 141 FPS | `consumers_count` + `Reshape(Constant)` 修复 |
| +WMMA auto-select | 157 FPS | gfx12xx 返回 `""` 自动选 WMMA |
| +Tuning | ~181 FPS | `ROCMLIR_ENABLE_TUNING=1` |
| +Transpose/Tile 预分配 | **212 FPS** | 消除推理时动态 `hipMalloc`（-1.5ms/inf） |
| **+migraphx dialect 默认** | **233 FPS** | Conv kernel 换用 MIGraphX MLIR 编译路径（-2.1ms/inf） |
| MIGraphX 参考 | 224 FPS | — |

### 7.2 关键优化说明

#### A. Transpose/Tile 预分配（+18% FPS）

**问题**：`TransposeOp` 和 `TileOp` 在每次推理的 `Execute()` 里调用 `hipMalloc`
为 stride/offset 数组分配临时 GPU buffer，`hipMalloc` 会触发 GPU 全局同步，
引入 ~1.5ms/inference 的延迟。

**修复**：在 op 构造时（compile 阶段）预分配 device buffer，
`Execute()` 只做 `hipLaunchKernel`，不再有动态分配。

**修改文件**：
- `kernels/transpose.hip`：新增 `allocTransposeDeviceBuffers` / `freeTransposeDeviceBuffers`
- `ops/transpose.cpp/.hpp`：构造时调用 `alloc`，析构时 `free`，Execute 使用预分配 buffer
- `kernels/tile.hip`、`ops/tile.cpp/.hpp`：同上

#### B. migraphx dialect 默认启用（+10% FPS）

**问题**：rock dialect 的 `patch_ir_bias_silu` 用 `linalg.generic` 作为 epilogue，
Conv 输出需要经过中间 `memref.alloc`（GPU 内存分配）才能做 bias+silu，
产生额外的内存读写。

**migraphx dialect 的优势**：
- `migraphx.convolution` + 内联 `sigmoid`/`mul` 算子，单 kernel 完成
- GPU 编译器可以将 epilogue 操作放在寄存器中，不需要中间 alloc
- `MarkConvReshapeEpiloguePass` 额外将 Conv→Reshape（Attention 投影）融合进 kernel，
  消除了 `/model.23/Transpose`（0.41ms → 0.02ms）

**默认启用**：`ROCMLIR_EPILOGUE_FUSION` 条件从 `env=="1"` 改为 `env!="0"`。

**关闭方式**（用于对比调试）：
```bash
ROCMLIR_EPILOGUE_FUSION=0 benchmark_app ...
```

#### C. Swish no-op 消除

**问题**：`EliminateFusedSiluSwishPass` 标记了 110 个 Swish 为 inplace no-op，
但它们仍在 exec_sequence 中，每次推理都要 dispatch（即使 Execute 是空的）。

**修复**：在 `SubGraph::initExecuteSequence` 里，对标记了 `rocm_swish_inplace`
的节点直接 `continue`，不加入 exec_sequence，同时在每次推理前
`g_silu_applied_buffers.clear()` 防止状态泄漏。

dispatch 数量：352 → 256（减少 27%）。

### 7.3 内存使用分析

| 组件 | OV | MIGraphX |
|------|-----|---------|
| 中间张量（mutable blob） | **43 MB**（MemorySolver coloring） | 54 MB（scratch） |
| 模型权重（const） | 106 MB | ~50 MB |
| HSACO GPU code segment | ~300 MB（74 kernels × 2MB/kernel 加载膨胀） | ~0 MB（嵌入 .so） |

OV 的中间张量内存（43MB）已经优于 MIGraphX（54MB）。
主要内存差距来自 HSACO GPU code 加载，这是独立 `hipModuleLoadData` 调用的固有开销。

### 7.4 MIGraphX vs OV 融合机制对比

| 维度 | MIGraphX | OV ROCm Plugin |
|------|----------|----------------|
| 是否用 hipGraph | **否**（顺序单流） | 否（已实现基础设施，待完整验证） |
| conv epilogue | MLIR 内联融合（一步） | migraphx dialect（一步，已匹配） |
| kernel 数量/inference | ~127 | ~256（含 inplace no-op） |
| 内存分配 | compile 时全部预分配 | compile 时预分配，Transpose/Tile 已修复 |
| 单流延迟 | 4.44ms | ~5.5ms |
| 多流吞吐（最优 nireq） | 224 FPS（单流） | **233 FPS**（nireq=12-14） |

---

## 8. 附录：常见问题

### rocMLIR 构建失败：`_GLIBCXX_USE_CXX11_ABI` 宏重复

**原因**：HIP cmake 配置 (`hip::host` target) 的 `INTERFACE_COMPILE_DEFINITIONS`
在某些 ROCm 版本里包含格式错误的宏定义，被 Ninja 解析后重复。

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

### Memory Access Fault

```
Memory access fault by GPU node-X on address ...
```

**原因**：多 GPU 环境中指定了错误的 GPU，VRAM 被其他进程占用。

**解决**：
```bash
rocm-smi --showmeminfo vram  # 查看各 GPU VRAM 占用
HIP_VISIBLE_DEVICES=N benchmark_app ...  # 指定空闲 GPU
```

### cmake 找不到 cmake 二进制

ROCm 容器通常没有系统 cmake，用 pip 安装：
```bash
pip install cmake
which cmake  # 应在 /opt/venv/bin/cmake 或 ~/.local/bin/cmake
```

### 调优缓存使性能下降

在**共享 GPU** 上运行 tuning 时，其他进程的负载导致 perf_config 测量不准。

**解决**：在独占 GPU 上删除旧缓存重新调优：
```bash
rm ~/.cache/ov_rocmlir_tuning_gfx1201.json
ROCMLIR_ENABLE_TUNING=1 benchmark_app -m model.onnx -d ROCM -hint throughput -t 180
```

### 验证 migraphx dialect 是否生效

```bash
benchmark_app -m model.onnx -d ROCM -hint throughput -nireq 1 -t 5 \
    2>&1 | grep -E "EpilogueTag|SiLUAddEpilogue|migraphx" | head -5
# 应看到：[EpilogueTag] ... 或 [SiLUAddEpilogue] Compiled 6-arg kernel
```

### 验证 Transpose 预分配是否生效

```bash
# ROCM_LOG 会显示 hipMalloc 调用，预分配后推理时不应出现 hipMalloc
AMD_LOG_LEVEL=3 benchmark_app -m model.onnx -d ROCM -hint throughput -nireq 1 -t 3 \
    2>&1 | grep -c "hipMalloc"
# 期望：比之前少（主要剩下 compile 阶段的分配）
```
