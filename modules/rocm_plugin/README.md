# OpenVINO™ ROCm GPU Plugin

OpenVINO™ ROCm GPU Plugin 使 AMD GPU 能够运行深度神经网络推理，
基于 OpenVINO™ API，支持 RDNA4（gfx1201）、RDNA3（gfx1100）、CDNA3（gfx950）等架构。

## 性能亮点（yolo26x FP16，640×640）

| GPU | 架构 | OV ROCm Plugin | MIGraphX | 优势 |
|-----|------|---------------|---------|------|
| Radeon AI PRO R9700 | gfx1201 / RDNA4 | **~280 FPS** ✅ | ~224 FPS | +25% |
| RX 7900 XTX | gfx1100 / RDNA3 | **~221 FPS** ✅ | ~168 FPS | +32% |
| Instinct MI350 | gfx950 / CDNA3 | **~500 FPS** ✅ | ~295 FPS | +69% |

> ✅ 超过 MIGraphX 水平。gfx1201 数据基于 hipGraph + fused_epilogue 调优，
> 测试条件：yolo26x.onnx，batch=1，FP16，nireq=8，120s 稳定运行。

## 支持的平台

| OS | GPU 架构 | ROCm |
|----|---------|------|
| Ubuntu 22.04 / 24.04 (64-bit) | gfx1201 / gfx1100 / gfx950 | 7.2.x |

## 快速开始

详细构建和调优说明见 [性能指南](docs/performance_guide.md)。

### 最简步骤（以 gfx1201 为例）

```bash
# 1. 构建 rocMLIR（应用 patches）
git clone --branch rocm-rel-7.2 --depth 1 \
    https://github.com/ROCm/rocMLIR.git /home/rocMLIR
cd /home/rocMLIR && git am --3way \
    < /home/openvino/openvino_contrib/modules/rocm_plugin/patches/0001-rocmlir-fix-AlignTiling-build-error.patch \
    < /home/openvino/openvino_contrib/modules/rocm_plugin/patches/0002-tosatorock-features.patch \
    < /home/openvino/openvino_contrib/modules/rocm_plugin/patches/0003-driver-tuning-fallback.patch
mkdir -p /home/rocMLIR/build && cd /home/rocMLIR/build
cmake /home/rocMLIR -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DGPU_TARGETS=gfx1201 -DROCMLIR_DRIVER_E2E_TEST_ENABLED=0 \
    -DLLVM_ENABLE_PROJECTS="mlir;lld"
ninja -j$(nproc) rocmlir-driver rocmlir-gen
cp bin/rocmlir-driver bin/rocmlir-gen /home/rocmlir_install/bin/

# 2. 编译 OpenVINO + Plugin
cmake /home/openvino/openvino -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DENABLE_ROCM=ON -DENABLE_ROCMLIR=ON \
    -DROCMLIR_INSTALL_DIR=/home/rocmlir_install \
    -DENABLE_OV_ONNX_FRONTEND=ON \
    -DOPENVINO_EXTRA_MODULES=/home/openvino/openvino_contrib/modules
ninja -j$(nproc) openvino_rocm_gpu_plugin benchmark_app

# 3. 一步复现最优性能（gfx1201，目标 280 FPS）
# 步骤 A：fused_epilogue 专项调优（首次，约 20-60 分钟，需独占 GPU）
ROCMLIR_ENABLE_TUNING_FUSED=1 \
ROCMLIR_TUNING_CACHE=/path/to/ov_rocmlir_tuning_gfx1201.json \
./benchmark_app -m yolo26x.onnx -d ROCM.0 -t 300 -nireq 1 -b 1

# 步骤 B：正式测试（使用调优缓存 + hipGraph）
printf '{"ROCM": {"ROCM_USE_HIP_GRAPH": "YES"}}' > /tmp/rocm_hg.json
ROCMLIR_TUNING_CACHE=/path/to/ov_rocmlir_tuning_gfx1201.json \
ROCMLIR_EPILOGUE_FUSION=1 ROCM_FUSE_ATTENTION=1 \
./benchmark_app -m yolo26x.onnx -d ROCM.0 \
    -load_config /tmp/rocm_hg.json -t 120 -nireq 8 -b 1
# 预期：~280 FPS（稳定，Max latency ~46ms）
```

## 关键特性

### 默认启用的优化（无需手动设置）

- **fused_epilogue dialect kernel**：Conv+Bias+SiLU 使用 rocMLIR MIGraphX 编译路径，
  epilogue 操作在寄存器内完成，无中间内存分配，比 rock dialect 快 25%
- **HSACO 磁盘缓存**：所有 kernel 编译结果持久化到 `~/.cache/ov_rocmlir_cache_<arch>/`，
  warm start 编译时间从 ~15s 降至 ~1.3s
- **hipGraph 支持**：通过 `ROCM_USE_HIP_GRAPH=YES` 配置启用，
  消除 CPU dispatch overhead，Max latency 204ms→65ms
- **Conv+Reshape 融合**：Q/K/V Attention 投影的 Reshape 融合进 conv kernel
- **Swish no-op 消除**：已融入 conv 的 SiLU Swish 节点不进入 dispatch 队列
- **VariadicSplit/Split hipGraph 安全**：使用 pinned memory pool，
  hipGraph capture 时 H2D source 地址稳定
- **GroupConv pass 顺序优化**：FuseGroupConvBiasAdd 提前，使 SinkSwishAddPass
  能处理 GroupConv→Swish→Add，消除额外 Swish
- **FusedConvolutionSlice Swish 消除**：SliceConv 后的 Swish 正确标记为 no-op

### 可选调优（显著提升）

- **fused_epilogue 专项调优**：`ROCMLIR_ENABLE_TUNING_FUSED=1`，
  对每个 conv+bias+silu kernel 独立搜索最优 tile config，
  44 shapes 中 36 个找到更优配置，吞吐量额外提升 ~16%（238→277 FPS）

## 主要环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ROCMLIR_EPILOGUE_FUSION` | ON | 设为 `0` 关闭 fused_epilogue dialect 融合路径 |
| `ROCMLIR_ENABLE_TUNING` | OFF | 设为 `1` 启用 plain rock.conv perf_config 调优 |
| `ROCMLIR_ENABLE_TUNING_FUSED` | OFF | 设为 `1` 启用 fused_epilogue 专项 perf_config 调优 |
| `ROCMLIR_TUNING_CACHE` | `~/.cache/ov_rocmlir_tuning_<arch>.json` | plain conv 调优缓存路径 |
| `ROCMLIR_TUNING_CACHE_FUSED` | `~/.cache/ov_rocmlir_tuning_<arch>_fused.json` | fused_epilogue 调优缓存路径 |
| `ROCM_FUSE_ATTENTION` | ON | 设为 `0` 关闭 Attention 融合 |
| `HIP_VISIBLE_DEVICES` | 全部 | 多卡环境必须指定 GPU 索引 |

## 文档

- [性能指南](docs/performance_guide.md)：详细构建步骤、调优方法、各架构性能数据
- [CUDA Opset](docs/cuda_opset.md)：支持的算子列表
