# BERT GEMM 调优指南：用调优后的 rocMLIR 融合 GEMM 在 RDNA3 上超越 MIGraphX

本文档记录在 **gfx1100 (RDNA3, Radeon PRO W7900)** 上把 bertsquad-12 的 fp16
推理性能从落后 MIGraphX 提升到**超越** MIGraphX 的完整方案与原理。

## 结果

| 模型 | OV ROCm Plugin | MIGraphX | 优势 |
|------|---------------|----------|------|
| bertsquad-12 (seq=256, nireq=2) | **~624 FPS** | ~610 FPS | **+2.3%** |
| yolo26x (640×640, nireq=4) | **~192 FPS** | ~157 FPS | **+22%** |

> 测试条件：gfx1100 / W7900，ROCm 7.2，fp16，batch=1。

## 核心洞察

在 RDNA3 上，**调优后的 rocMLIR GEMM 显著快于 hipBLASLt**，但只有在以下两个前提同时满足时
才能转化为端到端收益：

1. **正确的 IR 形式 + 编译管线**：用 rocMLIR 的高层融合 dialect（`dot` + `add`
   [+激活链]）而非手写底层 `rock.gemm` + `linalg.generic`。前者经
   `--kernel-pipeline=migraphx,highlevel,gpu,rocdl,binary` 能把整条 GEMM+bias[+GELU]
   链真正融合进**单个 kernel**；后者的 epilogue 不会被高效融合（实测 GEMM+GELU 慢 10 倍）。
   > 注：`migraphx` 是 rocMLIR 工具自带的 pipeline/dialect 名称，与 MIGraphX 框架无关，
   > 无法更改；插件自身的函数/变量命名均避免该词以防混淆。

2. **调优 perf_config**：默认 heuristic tile 配置不够好。在 `dot` op 上注入
   `perf_config="v4:..."`，从 `rocmlir-gen --emit-tuning-space=quick` 的候选里实测选最优。

**单 GEMM 实测（gfx1100，含 epilogue）：**

| GEMM | hipBLASLt | rocMLIR 融合+调优 |
|------|-----------|------------------|
| QKV 256×2304×768 +bias | ~30µs | **~17µs** |
| attn-out 256×768×768 (纯GEMM) | ~16µs | **~8.5µs** |
| FFN-down 256×768×3072 (纯GEMM) | ~36µs | **~26µs** |

## 架构：决策与执行分离（关键）

最初的失败教训：在每次推理的 `Execute()` 里 benchmark rocMLIR vs hipBLASLt 选优，会
**污染计时、拖慢端到端、甚至 hang**（模型加载期 GPU 上下文未就绪时 hipEvent 计时会死锁）。

正确做法——**调优阶段决策，运行阶段零开销**：

- **调优阶段**（仅当 `ROCMLIR_ENABLE_TUNING=1`）：`get_tuned_gemm_config()` 枚举候选、
  实测、把最优 perf_config 持久化到共享 JSON 缓存。
- **运行阶段**（正常推理）：op 构造时查缓存——**有调优 config 就用 rocMLIR，没有就用
  hipBLASLt**。`Execute()` 是纯 dispatch，不做任何 benchmark / 计时。这保证零退化、零 hang。

```
构造期：
  cfg = get_tuned_gemm_config(M,N,K,transB,arch,ep)   // 命中缓存→cfg；未调优→""
  if (!cfg.empty()) { 编译 rocMLIR(cfg); use_rocmlir_ = true; }

Execute()：
  if (use_rocmlir_) launch_rocmlir(...);   // 纯执行
  else              hipBLASLt(...);         // 回退，行为同改动前
```

通用 `MatMulOp` 额外加一次性数值校验（rocMLIR 输出 vs rocBLAS 逐元素 <2%），
不通过则回退 rocBLAS——保证通用算子的正确性。

## 哪些算子接入、哪些不接入

| 算子 | shape | epilogue | 是否用 rocMLIR | 原因 |
|------|-------|----------|---------------|------|
| QKV 投影 | 256×2304×768 | +bias | ✅ | 调优后 17µs < hipBLASLt 30µs |
| FFN-down | 256×768×3072 | 纯GEMM | ✅ | 调优后 26µs < 36µs |
| attn-out FC | 256×768×768 | 纯GEMM | ✅ | 调优后 8.5µs < 16µs |
| FFN-up FC+GELU | 256×3072×768 | +bias+GELU | ❌ 留 hipBLASLt | rocMLIR 的 sigmoid-GELU 融合反而慢（30 vs 25µs） |

**经验法则（RDNA3）**：纯 GEMM 和 GEMM+bias 用调优 rocMLIR 稳赢；带 GELU 的融合 epilogue
不划算，保留 hipBLASLt。

## 多平台原则（不硬编码 arch）

- 所有 arch 字符串运行时取自 `props.gcnArchName`，并 strip `:sramecc+` 等后缀。
- 调优缓存文件名内嵌 arch（`ov_rocmlir_tuning_<arch>.json`），不同平台天然隔离。
- 未在某平台调优过 → `get_tuned_gemm_config` 返回 ""→ 自动走 hipBLASLt，无退化。

## 使用方法

### 一次性调优（独占 GPU，约 5–10 分钟）

```bash
export OV_BIN=/home/openvino_rocm/openvino/bin/intel64/Release
export PATH=/home/rocmlir_install/bin:$PATH
export LD_LIBRARY_PATH=$OV_BIN:/opt/rocm/lib:$LD_LIBRARY_PATH
SHAPE="unique_ids_raw_output___9:0[1],segment_ids:0[1,256],input_mask:0[1,256],input_ids:0[1,256]"

# 枚举 perf_config、实测选最优、写入 ~/.cache/ov_rocmlir_tuning_<arch>.json
ROCMLIR_ENABLE_TUNING=1 $OV_BIN/benchmark_app -m bertsquad-12.onnx -d ROCM \
    -niter 2 -nireq 1 -hint none -api sync -shape "$SHAPE"
```

### 正常推理（自动加载缓存，零调优开销）

```bash
$OV_BIN/benchmark_app -m bertsquad-12.onnx -d ROCM \
    -niter 800 -nireq 2 -hint none -shape "$SHAPE"
# 预期 ~624 FPS（gfx1100）
```

### 重新调优（清缓存）

```bash
rm ~/.cache/ov_rocmlir_tuning_<arch>.json ~/.cache/ov_rocmlir_gemm_<arch>/*
# 再跑一次 ROCMLIR_ENABLE_TUNING=1
```

## 涉及代码

- `src/rocm/rocmlir_gemm.{cpp,hpp}`：`make_gemm_mlir`（高层融合 IR + perf_config 注入）、
  `compile_rocmlir_gemm`（migraphx,highlevel pipeline）、`get_tuned_gemm_config`（调优+持久化）。
- `src/ops/fused_qkv_op.{cpp,hpp}`：QKV 接入（GEMM+bias，构造期 cache 决策）。
- `src/ops/matmul.{cpp,hpp}`：FFN-down/attn-out 裸 GEMM 接入（None epilogue + 数值校验）。

## 注意事项 / 踩坑

- 调优产物 JSON 的 key 用 `std::hash`，**不可离线复现**，无法手工编辑单条；需要时清整个文件重调。
- 不要在构造期或 Execute 里用 hipEvent 做 benchmark——构造期 GPU 上下文未就绪会 hang，
  Execute 期会污染计时并退化。决策只靠缓存是否存在。
- 高层融合 IR 的 func 属性必须填真实 `arch`/`num_cu`，否则 rocmlir-driver 崩溃（free 错误）。
- 调优是 21 候选 × 完整 pipeline 编译 / 每 shape，较慢；建议后台运行。
