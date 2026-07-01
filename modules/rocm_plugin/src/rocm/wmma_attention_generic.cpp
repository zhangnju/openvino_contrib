#include "wmma_attention_generic.hpp"
#include <fmt/format.h>
#include <cstdio>

namespace ov { namespace rocm_gpu { namespace wmma_attn {

static std::string gen_source(int sq, int sk, int hd) {
    // Generic WMMA attention: Q[B,Sq,H,D] @ K[B,Sk,H,D]^T → softmax → @ V[B,Sk,H,D]
    // One warp (32 threads) per (batch, head). Each warp processes all Sq rows.
    // K, V loaded to LDS. Q streamed from global per M-tile.
    // WMMA 16×16×16 for both QK and PV.
    return fmt::format(R"(
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

extern "C" __global__ void __launch_bounds__(32)
wmma_generic_attention(
    const _Float16* __restrict__ Q,     // [B, Sq, H, D]
    const _Float16* __restrict__ K,     // [B, Sk, H, D]
    const _Float16* __restrict__ V,     // [B, Sk, H, D]
    _Float16* __restrict__ O,           // [B, Sq, H, D]
    int B, int H, float scale)
{{
    const int SQ = {0}, SK = {1}, HD = {2};
    const int bh = blockIdx.x;
    if (bh >= B * H) return;
    const int b = bh / H;
    const int h = bh % H;
    const int lane = threadIdx.x;
    const int my_row = lane % 16;
    const int my_half = lane / 16;

    // Strides for [B, S, H, D] layout
    const int stride_b = SQ * H * HD;  // Q/O batch stride
    const int stride_s = H * HD;       // Q/O seq stride
    const int stride_kb = SK * H * HD; // K/V batch stride
    const int stride_ks = H * HD;      // K/V seq stride

    // Load K into LDS [SK, HD]
    __shared__ _Float16 k_lds[{1}][{2}];
    __shared__ _Float16 v_lds[{1}][{2}];

    for (int s = lane; s < SK; s += 32) {{
        for (int d = 0; d < HD; d++) {{
            k_lds[s][d] = K[b * stride_kb + s * stride_ks + h * HD + d];
            v_lds[s][d] = V[b * stride_kb + s * stride_ks + h * HD + d];
        }}
    }}
    __syncthreads();

    // Process M-tiles (16 rows each)
    for (int mt = 0; mt < SQ / 16; mt++) {{
        int row_start = mt * 16;

        // Load Q tile [16, HD] — each lane loads one row
        int q_row = row_start + my_row;
        __shared__ _Float16 q_lds[16][{2}];
        for (int d = 0; d < HD; d++)
            q_lds[my_row][d] = Q[b * stride_b + q_row * stride_s + h * HD + d];
        __syncthreads();

        // QK[16, SK] = Q[16, HD] * K^T[HD, SK] via WMMA
        float8 qk_acc[{1}/16] = {{}};
        for (int kt = 0; kt < HD / 16; kt++) {{
            half16 q_frag;
            for (int i = 0; i < 16; i++) q_frag[i] = q_lds[my_row][kt * 16 + i];
            for (int nt = 0; nt < SK / 16; nt++) {{
                int bc = nt * 16 + my_row;
                half16 k_frag;
                for (int i = 0; i < 16; i++) k_frag[i] = k_lds[bc][kt * 16 + i];
                qk_acc[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(q_frag, k_frag, qk_acc[nt]);
            }}
        }}
        __syncthreads();

        // Extract scores and apply scale + softmax
        float scores[{1} / 2];
        for (int nt = 0; nt < SK / 16; nt++) {{
            for (int i = 0; i < 8; i++) {{
                scores[nt * 8 + i] = qk_acc[nt][i] * scale;
            }}
        }}

        // Softmax over SK/2 elements per half-warp
        float mx = scores[0];
        for (int i = 1; i < SK / 2; i++) mx = fmaxf(mx, scores[i]);
        float sib_mx = __shfl_xor(mx, 16);
        mx = fmaxf(mx, sib_mx);

        float sm = 0;
        for (int i = 0; i < SK / 2; i++) {{ scores[i] = __expf(scores[i] - mx); sm += scores[i]; }}
        float sib_sm = __shfl_xor(sm, 16);
        sm += sib_sm;
        float inv = 1.0f / sm;
        for (int i = 0; i < SK / 2; i++) scores[i] *= inv;

        // AV[16, HD] = P[16, SK] * V[SK, HD] via WMMA
        float8 av_acc[{2}/16] = {{}};
        for (int kt = 0; kt < SK / 16; kt++) {{
            half16 p_frag;
            for (int i = 0; i < 8; i++) p_frag[i] = (_Float16)scores[kt * 8 + i];
            for (int i = 0; i < 8; i++) {{
                float sv = __shfl_xor(scores[kt * 8 + i], 16);
                p_frag[8 + i] = (_Float16)sv;
            }}
            if (my_half == 1) {{
                half16 tmp = p_frag;
                for (int i = 0; i < 8; i++) {{ p_frag[i] = tmp[8+i]; p_frag[8+i] = tmp[i]; }}
            }}
            for (int dt = 0; dt < HD / 16; dt++) {{
                int vc = kt * 16 + my_row;
                half16 v_frag;
                for (int i = 0; i < 16; i++) v_frag[i] = v_lds[vc][dt * 16 + i];
                av_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(p_frag, v_frag, av_acc[dt]);
            }}
        }}

        // Write output O[b, q_row, h, d]
        _Float16* o_row = O + b * stride_b + q_row * stride_s + h * HD;
        for (int dt = 0; dt < HD / 16; dt++) {{
            for (int i = 0; i < 8; i++) {{
                o_row[dt * 16 + my_half * 8 + i] = (_Float16)av_acc[dt][i];
            }}
        }}
        __syncthreads();
    }}
}}
)", sq, sk, hd);
}

std::shared_ptr<WMMAAttnKernel> compile(int sq, int sk, int hd, const std::string& arch) {
    static const bool trace = std::getenv("ROCM_TRACE_WMMA_ATTN") != nullptr;

    if (sq % 16 != 0 || sk % 16 != 0 || hd % 16 != 0 || hd > 64 || sk > 512) {
        if (trace) fprintf(stderr, "[wmma-attn] unsupported: sq=%d sk=%d hd=%d\n", sq, sk, hd);
        return nullptr;
    }

    std::string source = gen_source(sq, sk, hd);
    std::string tag = fmt::format("wmma_attn_{}_{}_{}_{}", sq, sk, hd, arch);

    auto kernel = hiprtc::compile(source, "wmma_generic_attention", arch, tag);
    if (!kernel) return nullptr;

    auto result = std::make_shared<WMMAAttnKernel>();
    result->kernel = kernel;
    result->sq = sq; result->sk = sk; result->hd = hd;

    if (trace) fprintf(stderr, "[wmma-attn] compiled: sq=%d sk=%d hd=%d\n", sq, sk, hd);
    return result;
}

void launch(const WMMAAttnKernel& wk, hipStream_t stream,
            const void* Q, const void* K, const void* V, void* O,
            int batch, int heads, float scale) {
    struct Args {
        const void* Q; const void* K; const void* V; void* O;
        int B; int H; float scale;
    };
    Args args = {Q, K, V, O, batch, heads, scale};

    unsigned grid_x = batch * heads;
    hiprtc::launch(*wk.kernel, stream, grid_x, 32, &args, sizeof(args));
}

}}} // namespace
