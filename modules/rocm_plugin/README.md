# OpenVINO™ ROCm GPU Plugin

OpenVINO™ ROCm GPU Plugin 使 AMD GPU 能够运行深度神经网络推理，
基于 OpenVINO™ API，支持 RDNA4（gfx1201）、RDNA3（gfx1100）、CDNA3（gfx950）等架构。

## 性能亮点（yolo26x FP16，640×640）

| GPU | 架构 | OV ROCm Plugin | MIGraphX |
|-----|------|---------------|---------|
| Radeon AI PRO R9700 | gfx1201 / RDNA4 | **~233 FPS** ✅ | ~224 FPS |
| RX 7900 XTX | gfx1100 / RDNA3 | **~221 FPS** ✅ | ~168 FPS |
| Instinct MI350 | gfx950 / CDNA3 | **~500 FPS** ✅ | ~295 FPS |

> ✅ 超过 MIGraphX 水平

## 支持的平台

| OS | GPU 架构 | ROCm |
|----|---------|------|
| Ubuntu 22.04 / 24.04 (64-bit) | gfx1201 / gfx1100 / gfx950 | 7.2.x |

## 快速开始

详细构建和调优说明见 [性能指南](docs/performance_guide.md)。

### 最简步骤

```bash
# 1. 编译 rocMLIR（首次）
git clone --branch rocm-rel-7.2 --depth 1 \
    https://github.com/ROCm/rocMLIR.git /home/rocMLIR
mkdir -p /home/rocMLIR/build && cd /home/rocMLIR/build
cmake /home/rocMLIR -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DGPU_TARGETS=gfx1201 \
    -DROCMLIR_DRIVER_E2E_TEST_ENABLED=0 \
    -DLLVM_ENABLE_PROJECTS="mlir;lld"
ninja -j$(nproc) rocmlir-driver rocmlir-gen
cp bin/rocmlir-driver bin/rocmlir-gen /home/rocmlir_install/bin/

# 2. 编译 OpenVINO + Plugin
cmake /home/openvino/openvino -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DENABLE_ROCM=ON -DENABLE_ROCMLIR=ON \
    -DROCMLIR_INSTALL_DIR=/home/rocmlir_install \
    -DENABLE_OV_ONNX_FRONTEND=ON \
    -DOPENVINO_EXTRA_MODULES=/home/openvino/openvino_contrib/modules
ninja -j$(nproc) openvino_rocm_gpu_plugin benchmark_app openvino_onnx_frontend

# 3. 运行 Benchmark
export PATH=/home/rocmlir_install/bin:$PATH
export LD_LIBRARY_PATH=/home/openvino/bin/intel64/Release:/opt/rocm/lib:$LD_LIBRARY_PATH

# 首次：生成调优缓存（独占 GPU，约 10-30 分钟）
ROCMLIR_ENABLE_TUNING=1 \
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx1201.json \
./benchmark_app -m yolo26x.onnx -d ROCM -hint throughput -nireq 14 -t 180

# 正式测试
ROCMLIR_TUNING_CACHE=~/.cache/ov_rocmlir_tuning_gfx1201.json \
./benchmark_app -m yolo26x.onnx -d ROCM -hint throughput -nireq 14 -t 60
```

## 关键特性

### 默认启用的优化（无需手动设置）

- **migraphx dialect conv kernel**：Conv+Bias+SiLU 使用 MIGraphX MLIR 编译路径，
  epilogue 操作在寄存器内完成，无中间内存分配，比 rock dialect 快 25%
- **Conv+Reshape 融合**：Q/K/V Attention 投影的 Reshape 融合进 conv kernel，
  消除独立 Transpose（yolo26x 中节省 0.4ms/inference）
- **Attention MatMul 融合**：QKV Attention 用 MIGraphX MLIR kernel
- **pe(V) 融合**：pe(V) depthwise conv 与 AV 输出融合（gfx1201 专有）
- **VariadicSplit 零拷贝**：QKV split 通过 buffer alias 实现，零显存复制
- **Transpose/Tile 预分配**：stride 数组在构造时预分配 GPU buffer，
  消除推理时动态 `hipMalloc`（+18% FPS）
- **Swish no-op 消除**：已融入 conv 的 SiLU 对应的 Swish 节点从调度队列移除，
  dispatch 数量 352→256

### 关闭默认融合（调试用）

```bash
ROCMLIR_EPILOGUE_FUSION=0 benchmark_app ...  # 退回 rock dialect
ROCM_FUSE_ATTENTION=0 benchmark_app ...      # 退回 rocBLAS attention
```

## 主要环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `ROCMLIR_EPILOGUE_FUSION` | ON | 设为 `0` 关闭 migraphx dialect 和 conv+reshape 融合 |
| `ROCMLIR_ENABLE_TUNING` | OFF | 设为 `1` 启用 perf_config 穷举调优（需独占 GPU） |
| `ROCMLIR_TUNING_CACHE` | `~/.cache/ov_rocmlir_tuning_<arch>.json` | 调优缓存路径 |
| `ROCM_FUSE_ATTENTION` | ON | 设为 `0` 关闭 Attention 融合 |
| `HIP_VISIBLE_DEVICES` | 全部 | 多卡环境必须指定 GPU 索引 |

## 文档

- [性能指南](docs/performance_guide.md)：详细构建步骤、调优方法、各架构性能数据
- [CUDA Opset](docs/cuda_opset.md)：支持的算子列表
