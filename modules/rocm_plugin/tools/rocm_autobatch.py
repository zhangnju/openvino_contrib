#!/usr/bin/env python3
"""
rocm_autobatch.py — 自动搜索 OpenVINO ROCm Plugin 的最优 batch size

用法：
    python3 rocm_autobatch.py -m model.onnx [options]

搜索策略：
    对 batch_sizes × nireqs 的参数空间做网格扫描，
    最大化 total_throughput = batch_size × FPS（总图片/秒）。

输出：
    每组参数的 throughput 表格，以及最优配置的运行命令。
"""

import subprocess
import sys
import os
import argparse
import re
import json
from itertools import product


def parse_args():
    p = argparse.ArgumentParser(
        description="自动搜索 ROCm Plugin 最优 batch size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速扫描（每组 15s，3×4=12 组合）
  python3 rocm_autobatch.py -m yolo11s.onnx --t 15

  # 完整扫描（每组 30s，5×6=30 组合）
  python3 rocm_autobatch.py -m model.onnx -B 1 2 4 8 16 -N 1 2 4 8 12 16 --t 30

  # 指定输入 shape（动态模型必须）
  python3 rocm_autobatch.py -m model.onnx \\
      --input_name images --input_hw 640 640 --t 20

  # 附带调优缓存 + hipGraph
  python3 rocm_autobatch.py -m model.onnx --hipgraph \\
      --env ROCMLIR_TUNING_CACHE=/path/to/cache.json \\
            ROCMLIR_EPILOGUE_FUSION=1
""")
    p.add_argument("-m", "--model", required=True, help="ONNX/IR 模型路径")
    p.add_argument("-B", "--batch_sizes", nargs="+", type=int,
                   default=[1, 2, 4, 8],
                   help="待搜索的 batch size 列表（默认: 1 2 4 8）")
    p.add_argument("-N", "--nireqs", nargs="+", type=int,
                   default=[1, 2, 4, 8, 12],
                   help="待搜索的 nireq 列表（默认: 1 2 4 8 12）")
    p.add_argument("--t", type=int, default=20,
                   help="每组测试时长（秒，默认 20）")
    p.add_argument("-d", "--device", default="ROCM.0",
                   help="推理设备（默认: ROCM.0）")
    p.add_argument("--input_name", default=None,
                   help="输入 tensor 名称（用于 -shape 参数，如 'images'）")
    p.add_argument("--input_hw", nargs=2, type=int, default=None,
                   metavar=("H", "W"),
                   help="输入 H W（与 --input_name 配合使用，如 640 640）")
    p.add_argument("--input_c", type=int, default=3,
                   help="输入通道数（默认 3）")
    p.add_argument("--benchmark_app", default=None,
                   help="benchmark_app 路径（默认自动搜索）")
    p.add_argument("--env", nargs="+", default=[],
                   help="额外环境变量（KEY=VALUE 格式），如 ROCMLIR_TUNING_CACHE=...")
    p.add_argument("--hipgraph", action="store_true",
                   help="启用 hipGraph（ROCM_USE_HIP_GRAPH=YES）")
    p.add_argument("--output_json", default=None,
                   help="将结果保存为 JSON 文件")
    return p.parse_args()


def find_benchmark_app(hint=None):
    """搜索 benchmark_app 二进制"""
    import shutil
    candidates = [hint] if hint else []
    candidates += [
        os.path.join(os.environ.get("OV_INSTALL_PATH", ""), "bin/intel64/Release/benchmark_app"),
        os.path.expanduser("~/openvino/bin/intel64/Release/benchmark_app"),
        "/home/openvino/bin/intel64/Release/benchmark_app",
        "benchmark_app",
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    found = shutil.which("benchmark_app")
    if found:
        return found
    return None


def run_benchmark(benchmark_app, model, batch, nireq, t, device,
                  input_name=None, input_hw=None, input_c=3,
                  extra_env=None, hipgraph=False):
    """
    运行 benchmark_app，返回 (throughput_fps, ok)。
    throughput_fps: inferences/sec（FPS，不含 batch 乘数）。
    """
    import tempfile

    env = os.environ.copy()
    if extra_env:
        for kv in extra_env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v

    # hipGraph config file
    hg_cfg_path = None
    if hipgraph:
        hg = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"ROCM": {"ROCM_USE_HIP_GRAPH": "YES"}}, hg)
        hg.close()
        hg_cfg_path = hg.name

    cmd = [
        benchmark_app,
        "-m", model,
        "-d", device,
        "-t", str(t),
        "-nireq", str(nireq),
        "-b", str(batch),
    ]

    # Shape specification for dynamic-batch models
    if input_name and input_hw:
        H, W = input_hw
        shape_spec = f"{input_name}[{batch},{input_c},{H},{W}]"
        cmd += ["-shape", shape_spec]

    if hg_cfg_path:
        cmd += ["-load_config", hg_cfg_path]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            env=env, timeout=t + 120
        )
        output = proc.stdout + proc.stderr

        # Parse throughput
        m = re.search(r"Throughput:\s+([\d.]+)\s+FPS", output)
        if m:
            return float(m.group(1)), True

        # Detect known failure modes
        if any(msg in output for msg in [
            "Dynamic models are not supported",
            "Batch size is invalid",
            "Cannot reshape",
            "GPU coredump",
            "Memory access fault",
        ]):
            return 0.0, False

        return 0.0, False

    except subprocess.TimeoutExpired:
        return 0.0, False
    except Exception:
        return 0.0, False
    finally:
        if hg_cfg_path:
            try:
                os.unlink(hg_cfg_path)
            except OSError:
                pass


def format_ips(images_per_sec):
    if images_per_sec >= 1000:
        return f"{images_per_sec/1000:.2f}k img/s"
    return f"{images_per_sec:.1f} img/s"


def main():
    args = parse_args()

    benchmark_app = find_benchmark_app(args.benchmark_app)
    if not benchmark_app:
        print("ERROR: benchmark_app not found.")
        print("  Set --benchmark_app /path/to/benchmark_app  or add it to PATH.")
        return 1

    print(f"\n{'='*68}")
    print(f"  ROCm Auto-Batch Tuner")
    print(f"{'='*68}")
    print(f"  Model    : {args.model}")
    print(f"  Device   : {args.device}")
    print(f"  Batches  : {args.batch_sizes}")
    print(f"  Nireqs   : {args.nireqs}")
    print(f"  T/run    : {args.t}s")
    print(f"  hipGraph : {'ON' if args.hipgraph else 'OFF'}")
    if args.env:
        print(f"  Env      : {' '.join(args.env)}")
    total = len(args.batch_sizes) * len(args.nireqs)
    print(f"  Total    : {total} combinations  "
          f"(~{total * args.t // 60}min {total * args.t % 60}s)")
    print(f"{'='*68}\n")

    # Table header
    print(f"  {'Batch':>6}  {'nireq':>6}  {'FPS':>10}  {'img/s':>13}  Status")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*13}  {'-'*8}")

    results = []
    best_ips = 0.0
    best_config = None
    failed_batches = set()
    done = 0

    for batch in args.batch_sizes:
        if batch in failed_batches:
            # Skip larger batches after consistent failures
            for nireq in args.nireqs:
                print(f"  {batch:>6}  {nireq:>6}  {'---':>10}  "
                      f"{'---':>13}  skip")
                results.append({
                    "batch": batch, "nireq": nireq,
                    "fps": 0.0, "images_per_sec": 0.0, "ok": False,
                    "skip": True
                })
            continue

        batch_failures = 0

        for nireq in args.nireqs:
            done += 1
            sys.stdout.write(
                f"  {batch:>6}  {nireq:>6}  [running {done}/{total}]...\r"
            )
            sys.stdout.flush()

            fps, ok = run_benchmark(
                benchmark_app, args.model, batch, nireq, args.t,
                args.device, args.input_name, args.input_hw, args.input_c,
                args.env, args.hipgraph
            )

            ips = batch * fps if ok else 0.0
            status = "OK" if ok else "FAIL"
            if not ok:
                batch_failures += 1

            # Clear running line
            sys.stdout.write(" " * 70 + "\r")
            print(f"  {batch:>6}  {nireq:>6}  {fps:>10.1f}  "
                  f"{format_ips(ips):>13}  {status}")

            results.append({
                "batch": batch, "nireq": nireq,
                "fps": fps, "images_per_sec": ips, "ok": ok
            })

            if ok and ips > best_ips:
                best_ips = ips
                best_config = (batch, nireq, fps)

        # If all nireqs failed for batch>1, propagate to larger batches
        if batch > 1 and batch_failures == len(args.nireqs):
            failed_batches.add(batch)
            for larger in args.batch_sizes:
                if larger > batch:
                    failed_batches.add(larger)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*68}")

    if best_config:
        b, n, fps = best_config
        print(f"\n  Best configuration found:")
        print(f"    batch_size = {b}")
        print(f"    nireq      = {n}")
        print(f"    FPS        = {fps:.1f}  (inferences/s)")
        print(f"    Throughput = {format_ips(best_ips)}  (batch × FPS)")

        # Build optimal command string
        env_prefix = " ".join(args.env)
        shape_arg = ""
        if args.input_name and args.input_hw:
            H, W = args.input_hw
            shape_arg = f" -shape {args.input_name}[{b},{args.input_c},{H},{W}]"
        hg_arg = " -load_config /tmp/rocm_hg.json" if args.hipgraph else ""

        cmd_str = (
            (env_prefix + " " if env_prefix else "") +
            f"{benchmark_app} -m {args.model} -d {args.device}"
            f" -t 120 -nireq {n} -b {b}{shape_arg}{hg_arg}"
        )

        print(f"\n  Optimal benchmark command (120s run):")
        print(f"    {cmd_str}")

        if args.hipgraph:
            print(f"\n  Note: hipGraph config file must exist first:")
            print(f"    printf '{{\"ROCM\": {{\"ROCM_USE_HIP_GRAPH\": \"YES\"}}}}'"
                  f" > /tmp/rocm_hg.json")
    else:
        print("\n  No successful configuration found!")
        print("  Check model compatibility and device availability.")

    # Top-5 table
    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        print(f"\n  Top-5 by total throughput (img/s = batch × FPS):")
        print(f"  {'Rank':>4}  {'Batch':>6}  {'nireq':>6}  "
              f"{'FPS':>9}  {'img/s':>13}")
        print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*9}  {'-'*13}")
        top5 = sorted(ok_results, key=lambda x: -x["images_per_sec"])[:5]
        for i, r in enumerate(top5, 1):
            print(f"  {i:>4}  {r['batch']:>6}  {r['nireq']:>6}  "
                  f"{r['fps']:>9.1f}  {format_ips(r['images_per_sec']):>13}")

    # JSON output
    if args.output_json:
        out = {
            "model": args.model,
            "device": args.device,
            "t_per_run": args.t,
            "hipgraph": args.hipgraph,
            "best": {
                "batch": best_config[0], "nireq": best_config[1],
                "fps": best_config[2], "images_per_sec": best_ips,
            } if best_config else None,
            "results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved: {args.output_json}")

    print()
    return 0 if best_config else 1


if __name__ == "__main__":
    sys.exit(main())
