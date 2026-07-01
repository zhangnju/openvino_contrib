#!/usr/bin/env python3
"""
Triton Flash Attention AOT Compiler for OV ROCm Plugin.
Compiles the flash attention forward kernel to HSACO + JSON metadata.

On first compile for a given shape, runs autotune over candidate BLOCK_M/BLOCK_N/num_warps
configs, benchmarks each on the actual GPU, and persists the best to the cache.
Subsequent runs hit the cache and return immediately.

Dependencies: triton >= 3.5.0, torch (for autotune benchmark tensors)
No flash_attn pip package needed — kernel source is embedded below.

Kernel source adapted from:
  https://github.com/dao-ailab/flash-attention (flash_attn/flash_attn_triton.py)
  License: BSD-3-Clause
"""
import argparse, hashlib, json, math, os, sys

# ─── Embedded Flash Attention Forward Kernel ───────────────────────────────────
_KERNEL_SOURCE = '''
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, Bias, Out, Lse, TMP,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_bb, stride_bh, stride_bm,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0).to(tl.float32)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
'''

# ─── Autotune search space generation ──────────────────────────────────────────
def _generate_tune_space(headdim):
    """Generate valid (BLOCK_M, BLOCK_N, num_warps) candidates based on hardware constraints.

    Constraints (gfx1100 RDNA3, wave32, 64KB LDS, 256 VGPRs/wave):
    - BLOCK_M, BLOCK_N must be powers of 2, >= 16
    - K+V tiles in LDS: 2 * BLOCK_N * BLOCK_HEADDIM * 2 bytes <= 64KB
    - Q tile in registers: BLOCK_M * BLOCK_HEADDIM elements per warp (register pressure)
    - num_warps * 32 threads should not exceed output tile elements
    - More warps = more parallelism but higher register pressure per workgroup

    Targets ~10-15 candidates for fast autotune (~30s total).
    """
    BLOCK_HEADDIM = max(1 << (headdim - 1).bit_length(), 16)
    LDS_LIMIT = 65536  # 64KB

    candidates = set()
    for bm in [16, 32, 64, 128]:
        for bn in [32, 64, 128]:
            for nw in [1, 2, 4, 8]:
                # LDS: K[BN,D] + V[BN,D] in fp16
                if 2 * bn * BLOCK_HEADDIM * 2 > LDS_LIMIT:
                    continue
                # Thread count sanity
                if nw * 32 > bm * bn:
                    continue
                # Register pressure: Q tile per warp = BM/nw * D elements
                q_regs_per_warp = (bm // max(nw, 1)) * BLOCK_HEADDIM
                if q_regs_per_warp > 256 * 4:  # ~256 VGPRs, 4 elements per VGPR
                    continue
                # Prune combos that are clearly suboptimal
                if bm >= 128 and nw < 2: continue
                if nw >= 8 and bm <= 32: continue
                if headdim <= 32 and bm > 64 and nw < 4: continue
                if headdim >= 128 and bm < 32: continue
                # BN=32 is inefficient for large Sk (too many K-loop iterations)
                if bn == 32 and bm <= 32: continue
                # warps=4 on BM=16 wastes threads (only 16 rows, 4*32=128 threads)
                if bm == 16 and nw >= 4: continue
                candidates.add((bm, bn, nw))

    return sorted(candidates)

# Heuristic default (no autotune): used when torch is unavailable
_HEURISTIC_DEFAULT = {
    32:  (16, 64, 1),
    64:  (64, 64, 2),
    128: (128, 128, 8),
}

def _get_jit_fn():
    """Load the embedded kernel from a temp file (triton @jit needs inspect.getsource)."""
    import tempfile, importlib.util
    kernel_file = os.path.join(tempfile.gettempdir(), "_ov_fa_kernel.py")
    with open(kernel_file, 'w') as kf:
        kf.write(_KERNEL_SOURCE)
    spec = importlib.util.spec_from_file_location("_ov_fa_kernel", kernel_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._fwd_kernel


def _compile_one(jit_fn, headdim, seqlen_q, seqlen_k, bias_type, causal,
                  block_m, block_n, num_warps):
    """Compile a single config. Returns (hsaco_bytes, metadata) or None on failure."""
    import triton
    from triton.compiler import ASTSource, compile as triton_compile

    BLOCK_HEADDIM = max(1 << (headdim - 1).bit_length(), 16)
    EVEN_M = (seqlen_q % block_m == 0)
    EVEN_N = (seqlen_k % block_n == 0)
    EVEN_HEADDIM = (headdim == BLOCK_HEADDIM)

    runtime_params = [p for p in jit_fn.params if not p.is_constexpr]
    sig = {}
    for p in runtime_params:
        name = p.name
        if name in ('Q', 'K', 'V', 'Bias', 'Out'): sig[name] = '*fp16'
        elif name in ('Lse', 'TMP'): sig[name] = '*fp32'
        elif name == 'softmax_scale': sig[name] = 'fp32'
        else: sig[name] = 'i32'

    ce = {
        'BIAS_TYPE': bias_type, 'IS_CAUSAL': bool(causal),
        'BLOCK_HEADDIM': BLOCK_HEADDIM, 'EVEN_M': EVEN_M, 'EVEN_N': EVEN_N,
        'EVEN_HEADDIM': EVEN_HEADDIM, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
    }

    try:
        src = ASTSource(fn=jit_fn, signature=sig, constexprs=ce)
        target = triton.runtime.driver.active.get_current_target()
        compiled = triton_compile(src, target=target)
        return compiled.asm['hsaco'], compiled.metadata
    except Exception as e:
        return None


def _benchmark_config(jit_fn, headdim, seqlen_q, seqlen_k, bias_type, causal,
                      block_m, block_n, num_warps):
    """Benchmark a single config on GPU. Returns time in ms, or float('inf') on failure."""
    import torch, time

    # Use small batch for benchmark to save memory
    B, H, D = 2, 1, headdim
    # Cap seq lengths for benchmark to avoid OOM on huge shapes
    sq = min(seqlen_q, 2048)
    sk = min(seqlen_k, 2048)

    BLOCK_HEADDIM = max(1 << (D - 1).bit_length(), 16)
    EVEN_M = (sq % block_m == 0)
    EVEN_N = (sk % block_n == 0)
    sq_rounded = ((sq + 127) // 128) * 128
    scale = 1.0 / math.sqrt(D)

    try:
        Q = torch.randn(B, sq, H, D, dtype=torch.float16, device="cuda")
        K = torch.randn(B, sk, H, D, dtype=torch.float16, device="cuda")
        V = torch.randn(B, sk, H, D, dtype=torch.float16, device="cuda")
        O = torch.empty_like(Q)
        Lse = torch.empty(B * H, sq_rounded, dtype=torch.float32, device="cuda")
        TMP = torch.empty(B * H, sq_rounded, dtype=torch.float32, device="cuda")

        grid = lambda meta: ((sq + block_m - 1) // block_m, B * H)

        # Warmup
        for _ in range(2):
            jit_fn[grid](
                Q, K, V, None, O, Lse, TMP, scale,
                sq*H*D, H*D, D, sk*H*D, H*D, D, sk*H*D, H*D, D,
                0, 0, 0, sq*H*D, H*D, D,
                H, sq, sk, sq_rounded, D, sq // 32, sk // 32,
                BIAS_TYPE=bias_type, IS_CAUSAL=bool(causal),
                BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N,
                EVEN_HEADDIM=(D == BLOCK_HEADDIM), BLOCK_M=block_m, BLOCK_N=block_n,
                num_warps=num_warps, num_stages=1,
            )
        torch.cuda.synchronize()

        N = 5
        t0 = time.time()
        for _ in range(N):
            jit_fn[grid](
                Q, K, V, None, O, Lse, TMP, scale,
                sq*H*D, H*D, D, sk*H*D, H*D, D, sk*H*D, H*D, D,
                0, 0, 0, sq*H*D, H*D, D,
                H, sq, sk, sq_rounded, D, sq // 32, sk // 32,
                BIAS_TYPE=bias_type, IS_CAUSAL=bool(causal),
                BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N,
                EVEN_HEADDIM=(D == BLOCK_HEADDIM), BLOCK_M=block_m, BLOCK_N=block_n,
                num_warps=num_warps, num_stages=1,
            )
        torch.cuda.synchronize()
        return (time.time() - t0) / N * 1000
    except Exception:
        return float('inf')


def compile_flash_attn(seqlen_q, seqlen_k, headdim, bias_type, causal, arch, output_dir):
    BLOCK_HEADDIM = max(1 << (headdim - 1).bit_length(), 16) if headdim > 0 else 16

    key = f"sq{seqlen_q}_sk{seqlen_k}_d{headdim}_b{bias_type}_c{causal}_{arch}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    hsaco_path = os.path.join(output_dir, f"triton_fa_{h}.hsaco")
    json_path = os.path.join(output_dir, f"triton_fa_{h}.json")

    if os.path.exists(hsaco_path) and os.path.exists(json_path):
        print(f"[triton-fa-aot] cache hit: {hsaco_path}", file=sys.stderr)
        print(hsaco_path); print(json_path)
        return hsaco_path, json_path

    os.makedirs(output_dir, exist_ok=True)
    jit_fn = _get_jit_fn()

    # ── Autotune: generate valid configs, benchmark each ──
    candidates = _generate_tune_space(headdim)

    # Always include the heuristic default
    hd_key = min(k for k in _HEURISTIC_DEFAULT if k >= headdim)
    heuristic = _HEURISTIC_DEFAULT[hd_key]
    if heuristic not in candidates:
        candidates.append(heuristic)
        candidates.sort()

    # Try to benchmark on GPU (requires torch)
    best_time = float('inf')
    best_config = _HEURISTIC_DEFAULT[hd_key]  # fallback
    try:
        import torch
        has_torch = torch.cuda.is_available()
    except ImportError:
        has_torch = False

    if has_torch:
        print(f"[triton-fa-aot] autotuning {len(candidates)} configs for "
              f"sq={seqlen_q} sk={seqlen_k} d={headdim} on {arch}", file=sys.stderr)
        for bm, bn, nw in sorted(candidates):
            t = _benchmark_config(jit_fn, headdim, seqlen_q, seqlen_k,
                                  bias_type, causal, bm, bn, nw)
            tag = ""
            if t < best_time:
                best_time = t
                best_config = (bm, bn, nw)
                tag = " ★"
            print(f"  BM={bm:3d} BN={bn:3d} w={nw} -> {t:.2f} ms{tag}", file=sys.stderr)
        print(f"[triton-fa-aot] best: BM={best_config[0]} BN={best_config[1]} "
              f"w={best_config[2]} ({best_time:.2f} ms)", file=sys.stderr)
    else:
        print(f"[triton-fa-aot] no torch — using heuristic default "
              f"BM={best_config[0]} BN={best_config[1]} w={best_config[2]}", file=sys.stderr)

    BLOCK_M, BLOCK_N, num_warps = best_config

    # ── Compile the winner ──
    print(f"[triton-fa-aot] compiling sq={seqlen_q} sk={seqlen_k} d={headdim} "
          f"BM={BLOCK_M} BN={BLOCK_N} w={num_warps}", file=sys.stderr)

    result = _compile_one(jit_fn, headdim, seqlen_q, seqlen_k, bias_type, causal,
                          BLOCK_M, BLOCK_N, num_warps)
    if result is None:
        raise RuntimeError("Triton FA AOT compile failed")

    hsaco, metadata = result
    with open(hsaco_path, 'wb') as f: f.write(hsaco)

    meta = {
        "kernel_name": metadata.name,
        "shared_mem": metadata.shared,
        "num_warps": num_warps,
        "block_m": BLOCK_M, "block_n": BLOCK_N,
        "block_headdim": BLOCK_HEADDIM,
        "arch": arch, "seqlen_q": seqlen_q, "seqlen_k": seqlen_k,
        "headdim": headdim, "bias_type": bias_type, "causal": causal,
    }
    with open(json_path, 'w') as f: json.dump(meta, f, indent=2)
    print(f"[triton-fa-aot] saved {hsaco_path} ({len(hsaco)}B)", file=sys.stderr)
    print(hsaco_path); print(json_path)
    return hsaco_path, json_path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen_q", type=int, required=True)
    p.add_argument("--seqlen_k", type=int, required=True)
    p.add_argument("--headdim", type=int, required=True)
    p.add_argument("--bias_type", default="none")
    p.add_argument("--causal", type=int, default=0)
    p.add_argument("--arch", default="gfx1100")
    p.add_argument("--output_dir", default=os.path.expanduser("~/.cache/ov_triton_fa"))
    a = p.parse_args()
    compile_flash_attn(a.seqlen_q, a.seqlen_k, a.headdim, a.bias_type, a.causal, a.arch, a.output_dir)
