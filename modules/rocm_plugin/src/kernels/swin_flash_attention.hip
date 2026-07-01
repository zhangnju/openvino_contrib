// Swin WMMA fused attention kernel for gfx1100 (wave32, WMMA 16x16x16).
// One warp per (window, head) pair. Processes all sq rows.
//
// Algorithm:
//   1. Load K [sk, hd] and V [sk, hd] into LDS
//   2. For each M-tile of Q (16 rows):
//      a. Load Q tile [16, hd] from global
//      b. WMMA: QK = Q[16,hd] × K^T[hd,sk] → scores[16,sk]
//      c. Add bias (if present), apply scale
//      d. Online softmax: max → exp → sum → normalize
//      e. WMMA: out = P[16,sk] × V[sk,hd] → [16,hd]
//      f. Write output [16, hd] to global
//
// Supports: sq,sk up to 64 (4 WMMA tiles), hd up to 64 (4 K-tiles).
// sq=49 → padded to 64 (last tile partially valid, masked in softmax).

#include "swin_flash_attention.hpp"
#include <hip/hip_fp16.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// Round up to multiple of 16
static inline __host__ __device__ int round16(int x) { return (x + 15) & ~15; }

// WMMA fused attention: one warp handles one (window, head)
static __global__ void swin_flash_attention_kernel(
    const __half* __restrict__ Q,     // [nW, H, sq, hd]
    const __half* __restrict__ K,     // [nW, H, sk, hd]
    const __half* __restrict__ V,     // [nW, H, sk, hd]
    const __half* __restrict__ bias,  // [H, sq, sk] or nullptr
    __half* __restrict__ Out,         // [nW, H, sq, hd]
    int nW, int H, int sq, int sk, int hd, float scale)
{
    const int warp_id = blockIdx.x;  // one block = one warp = one (w,h)
    if (warp_id >= nW * H) return;
    const int w = warp_id / H;
    const int h = warp_id % H;
    const int lane = threadIdx.x;  // 0..31

    const int sq_r = round16(sq);
    const int sk_r = round16(sk);

    // Strides for [nW, H, S, hd] layout
    const int stride_w = H * sq * hd;
    const int stride_h = sq * hd;
    const int stride_wk = H * sk * hd;
    const int stride_hk = sk * hd;

    // Base pointers for this (window, head)
    const __half* q_base = Q + w * stride_w + h * stride_h;
    const __half* k_base = K + w * stride_wk + h * stride_hk;
    const __half* v_base = V + w * stride_wk + h * stride_hk;
    __half* o_base = Out + w * stride_w + h * stride_h;

    // LDS for K and V: [sk_r, hd] each
    extern __shared__ __half lds[];
    __half* k_lds = lds;                    // [sk_r, hd]
    __half* v_lds = lds + sk_r * hd;        // [sk_r, hd]

    // Load K and V into LDS (each lane loads several elements)
    for (int idx = lane; idx < sk_r * hd; idx += 32) {
        int s = idx / hd, d = idx % hd;
        k_lds[idx] = (s < sk) ? k_base[s * hd + d] : __float2half(0.0f);
        v_lds[idx] = (s < sk) ? v_base[s * hd + d] : __float2half(0.0f);
    }
    __syncthreads();

    // Bias pointer for this head
    const __half* bias_base = bias ? (bias + h * sq * sk) : nullptr;

    // WMMA intrinsic helpers (gfx1100 wave32)
    // Each wmma_f32_16x16x16_f16 takes half16 A, half16 B, float8 C → float8 C
    typedef _Float16 half16 __attribute__((ext_vector_type(16)));
    typedef float float8 __attribute__((ext_vector_type(8)));

    const int my_row = lane % 16;
    const int my_half = lane / 16;
    const int sk_tiles = sk_r / 16;
    const int hd_tiles = hd / 16;

    // Process Q in M-tiles of 16 rows
    for (int mt = 0; mt < sq_r / 16; mt++) {
        int q_row_start = mt * 16;

        // Load Q tile [16, hd] into registers
        // Each lane loads one row (my_row) of hd elements
        int q_row = q_row_start + my_row;
        float q_reg[4];  // max hd=64 → 4 values per lane (16-wide WMMA)
        for (int d = 0; d < hd; d++) {
            // Will be used in WMMA fragments
        }

        // Phase 1: QK^T via WMMA
        // scores[my_row, sk] stored as float for softmax
        float scores[4 * 8];  // max sk_tiles=4, each WMMA gives 8 values per lane
        int n_scores = sk_tiles * 8;

        for (int nt = 0; nt < sk_tiles; nt++) {
            float8 acc = {};
            for (int kt = 0; kt < hd_tiles; kt++) {
                // Q fragment: row my_row, cols [kt*16 .. kt*16+15]
                half16 q_frag;
                if (q_row < sq) {
                    for (int i = 0; i < 16; i++)
                        q_frag[i] = q_base[q_row * hd + kt * 16 + i];
                } else {
                    for (int i = 0; i < 16; i++) q_frag[i] = (_Float16)0.0f;
                }
                // K^T fragment: K[nt*16+my_row, kt*16..kt*16+15] transposed
                // K is [sk, hd], we want K^T[hd, sk], so B fragment reads K row-wise
                int k_col = nt * 16 + my_row;  // column in K^T = row in K
                half16 k_frag;
                for (int i = 0; i < 16; i++)
                    k_frag[i] = k_lds[k_col * hd + kt * 16 + i];

                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag, k_frag, acc);
            }
            for (int i = 0; i < 8; i++)
                scores[nt * 8 + i] = acc[i] * scale;
        }

        // Add bias if present
        if (bias_base) {
            for (int nt = 0; nt < sk_tiles; nt++) {
                for (int i = 0; i < 8; i++) {
                    int sk_idx = nt * 16 + my_half * 8 + i;
                    if (q_row < sq && sk_idx < sk)
                        scores[nt * 8 + i] += __half2float(bias_base[q_row * sk + sk_idx]);
                }
            }
        }

        // Mask out-of-bounds scores (sq_r > sq or sk_r > sk)
        for (int nt = 0; nt < sk_tiles; nt++) {
            for (int i = 0; i < 8; i++) {
                int sk_idx = nt * 16 + my_half * 8 + i;
                if (q_row >= sq || sk_idx >= sk)
                    scores[nt * 8 + i] = -1e9f;
            }
        }

        // Phase 2: Online softmax
        float mx = -1e9f;
        for (int i = 0; i < n_scores; i++) mx = fmaxf(mx, scores[i]);
        // Cross-half max (lanes 0-15 and 16-31 have different score columns)
        float sib_mx = __shfl_xor(mx, 16);
        mx = fmaxf(mx, sib_mx);

        float sm = 0.0f;
        for (int i = 0; i < n_scores; i++) {
            scores[i] = __expf(scores[i] - mx);
            sm += scores[i];
        }
        float sib_sm = __shfl_xor(sm, 16);
        sm += sib_sm;
        float inv_sm = (sm > 0.0f) ? 1.0f / sm : 0.0f;
        for (int i = 0; i < n_scores; i++) scores[i] *= inv_sm;

        // Phase 3: PV via WMMA: Out[16,hd] = P[16,sk] × V[sk,hd]
        for (int dt = 0; dt < hd_tiles; dt++) {
            float8 av_acc = {};
            for (int nt = 0; nt < sk_tiles; nt++) {
                // P fragment: attention weights for this tile
                half16 p_frag;
                for (int i = 0; i < 8; i++)
                    p_frag[i] = (_Float16)scores[nt * 8 + i];
                // Cross-half exchange for WMMA B-side
                for (int i = 0; i < 8; i++) {
                    float sv = __shfl_xor(scores[nt * 8 + i], 16);
                    p_frag[8 + i] = (_Float16)sv;
                }
                if (my_half == 1) {
                    half16 tmp = p_frag;
                    for (int i = 0; i < 8; i++) { p_frag[i] = tmp[8+i]; p_frag[8+i] = tmp[i]; }
                }

                // V fragment: V[nt*16+my_row, dt*16..dt*16+15]
                int v_row = nt * 16 + my_row;
                half16 v_frag;
                for (int i = 0; i < 16; i++)
                    v_frag[i] = v_lds[v_row * hd + dt * 16 + i];

                av_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_frag, v_frag, av_acc);
            }

            // Write output
            if (q_row < sq) {
                for (int i = 0; i < 8; i++)
                    o_base[q_row * hd + dt * 16 + my_half * 8 + i] = __float2half(av_acc[i]);
            }
        }
    }
}

void launchSwinFlashAttention(
    hipStream_t stream,
    const void* Q, const void* K, const void* V,
    const void* bias, void* out,
    int nW, int H, int sq, int sk, int hd,
    float scale)
{
    int sk_r = round16(sk);
    size_t lds_bytes = 2 * sk_r * hd * sizeof(__half);  // K + V

    hipLaunchKernelGGL(swin_flash_attention_kernel,
        dim3(nW * H), dim3(32),  // one warp per block
        lds_bytes, stream,
        (const __half*)Q, (const __half*)K, (const __half*)V,
        (const __half*)bias, (__half*)out,
        nW, H, sq, sk, hd, scale);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
