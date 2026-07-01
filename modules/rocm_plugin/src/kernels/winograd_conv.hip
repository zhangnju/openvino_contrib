// Winograd F(4,3) convolution: optimized 3-phase implementation for gfx1100/gfx1201
// Phase 1: Input transform  [N,C,H,W] → [36, T, C]
// Phase 2: Batched GEMM     [36, T, C] × [36, C, K] → [36, T, K]  (rocBLAS strided batched)
// Phase 3: Output transform [36, T, K] → [N,K,H,W] + bias + relu
//
// v2 optimizations:
// - Interior/border tile split: interior tiles skip all boundary checks
// - Interior input: vectorized __half2 loads (3 loads per row vs 6 scalar)
// - Interior output: vectorized __half2 stores when aligned
// - No dynamic allocation — border kernel iterates all tiles, skips interior

#include "winograd_conv.hpp"
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

namespace ov {
namespace rocm_gpu {
namespace kernel {

// ── Filter transform (unchanged — runs once) ────────────────────────────

static __global__ void winograd_filter_transform_kernel(
    const __half* __restrict__ filter, __half* __restrict__ out, int K, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * C) return;
    int k = idx / C, c = idx % C;

    float f[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            f[i][j] = __half2float(filter[((k * C + c) * 3 + i) * 3 + j]);

    const float s6 = 1.0f / 6, s24 = 1.0f / 24, s12 = 1.0f / 12;
    float Gf[6][3];
    for (int j = 0; j < 3; j++) {
        Gf[0][j] = f[0][j] * 0.25f;
        Gf[1][j] = -(f[0][j] + f[1][j] + f[2][j]) * s6;
        Gf[2][j] = (-f[0][j] + f[1][j] - f[2][j]) * s6;
        Gf[3][j] = f[0][j] * s24 + f[1][j] * s12 + f[2][j] * s6;
        Gf[4][j] = f[0][j] * s24 - f[1][j] * s12 + f[2][j] * s6;
        Gf[5][j] = f[2][j];
    }
    float g[6][6];
    for (int i = 0; i < 6; i++) {
        g[i][0] = Gf[i][0] * 0.25f;
        g[i][1] = -(Gf[i][0] + Gf[i][1] + Gf[i][2]) * s6;
        g[i][2] = (-Gf[i][0] + Gf[i][1] - Gf[i][2]) * s6;
        g[i][3] = Gf[i][0] * s24 + Gf[i][1] * s12 + Gf[i][2] * s6;
        g[i][4] = Gf[i][0] * s24 - Gf[i][1] * s12 + Gf[i][2] * s6;
        g[i][5] = Gf[i][2];
    }
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            out[((a * 6 + b) * C + c) * K + k] = __float2half(g[a][b]);
}

// ── Transform helpers ───────────────────────────────────────────────────

__device__ __forceinline__ void bt_transform(const float d[6][6], float v[6][6]) {
    float tmp[6][6];
    for (int j = 0; j < 6; j++) {
        float d0=d[0][j], d1=d[1][j], d2=d[2][j], d3=d[3][j], d4=d[4][j], d5=d[5][j];
        tmp[0][j] = 4*d0 - 5*d2 + d4;
        tmp[1][j] = -4*d1 - 4*d2 + d3 + d4;
        tmp[2][j] = 4*d1 - 4*d2 - d3 + d4;
        tmp[3][j] = -2*d1 - d2 + 2*d3 + d4;
        tmp[4][j] = 2*d1 - d2 - 2*d3 + d4;
        tmp[5][j] = 4*d1 - 5*d3 + d5;
    }
    for (int i = 0; i < 6; i++) {
        float t0=tmp[i][0], t1=tmp[i][1], t2=tmp[i][2], t3=tmp[i][3], t4=tmp[i][4], t5=tmp[i][5];
        v[i][0] = 4*t0 - 5*t2 + t4;
        v[i][1] = -4*t1 - 4*t2 + t3 + t4;
        v[i][2] = 4*t1 - 4*t2 - t3 + t4;
        v[i][3] = -2*t1 - t2 + 2*t3 + t4;
        v[i][4] = 2*t1 - t2 - 2*t3 + t4;
        v[i][5] = 4*t1 - 5*t3 + t5;
    }
}

__device__ __forceinline__ void at_transform(const float m[6][6], float Y[4][4]) {
    float tmp[4][6];
    for (int j = 0; j < 6; j++) {
        float a0=m[0][j], a1=m[1][j], a2=m[2][j], a3=m[3][j], a4=m[4][j], a5=m[5][j];
        tmp[0][j] = a0 + a1 + a2 + a3 + a4;
        tmp[1][j] = a1 - a2 + 2*a3 - 2*a4;
        tmp[2][j] = a1 + a2 + 4*a3 + 4*a4;
        tmp[3][j] = a1 - a2 + 8*a3 - 8*a4 + a5;
    }
    for (int i = 0; i < 4; i++) {
        float t0=tmp[i][0], t1=tmp[i][1], t2=tmp[i][2], t3=tmp[i][3], t4=tmp[i][4], t5=tmp[i][5];
        Y[i][0] = t0 + t1 + t2 + t3 + t4;
        Y[i][1] = t1 - t2 + 2*t3 - 2*t4;
        Y[i][2] = t1 + t2 + 4*t3 + 4*t4;
        Y[i][3] = t1 - t2 + 8*t3 - 8*t4 + t5;
    }
}

// ── Input transform: interior tiles (no boundary checks) ────────────────

static __global__ void winograd_input_interior(
    const __half* __restrict__ input, __half* __restrict__ V,
    int N, int C, int H, int W, int tiles_h, int tiles_w,
    int ith_start, int ith_end, int itw_start, int itw_end)
{
    int ih = ith_end - ith_start, iw = itw_end - itw_start;
    int n_interior = ih * iw;
    int total_tiles = tiles_h * tiles_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_interior * C) return;

    int c  = idx % C;
    int it = (idx / C) % n_interior;
    int n  = idx / (C * n_interior);
    int th = it / iw + ith_start;
    int tw = it % iw + itw_start;
    int h0 = th * 4 - 1, w0 = tw * 4 - 1;

    const __half* base = input + ((size_t)n * C + c) * H * W;
    float d[6][6];

    if ((w0 & 1) == 0) {
        for (int i = 0; i < 6; i++) {
            const __half* row = base + (size_t)(h0 + i) * W + w0;
            __half2 p0 = *reinterpret_cast<const __half2*>(row);
            __half2 p1 = *reinterpret_cast<const __half2*>(row + 2);
            __half2 p2 = *reinterpret_cast<const __half2*>(row + 4);
            d[i][0] = __half2float(__low2half(p0));
            d[i][1] = __half2float(__high2half(p0));
            d[i][2] = __half2float(__low2half(p1));
            d[i][3] = __half2float(__high2half(p1));
            d[i][4] = __half2float(__low2half(p2));
            d[i][5] = __half2float(__high2half(p2));
        }
    } else {
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                d[i][j] = __half2float(base[(size_t)(h0 + i) * W + w0 + j]);
    }

    float v[6][6];
    bt_transform(d, v);

    int tg = n * total_tiles + th * tiles_w + tw;
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            V[((a * 6 + b) * N * total_tiles + tg) * C + c] = __float2half(v[a][b]);
}

// ── Input transform: border tiles (with boundary checks) ────────────────
// Iterates all tiles, skips the interior rectangle.

static __global__ void winograd_input_border(
    const __half* __restrict__ input, __half* __restrict__ V,
    int N, int C, int H, int W, int tiles_h, int tiles_w,
    int ith_start, int ith_end, int itw_start, int itw_end)
{
    int total_tiles = tiles_h * tiles_w;
    int n_border = total_tiles - (ith_end - ith_start) * (itw_end - itw_start);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_border * C) return;

    int c  = idx % C;
    int bt = (idx / C) % n_border;
    int n  = idx / (C * n_border);

    // Map linear border index → (th, tw) skipping interior rect
    // Layout: top strip (th < ith_start, all tw), then for each th in [ith_start, ith_end):
    //   left strip (tw < itw_start) + right strip (tw >= itw_end),
    // then bottom strip (th >= ith_end, all tw)
    int th, tw;
    int top_count = ith_start * tiles_w;
    if (bt < top_count) {
        th = bt / tiles_w;
        tw = bt % tiles_w;
    } else {
        int rem = bt - top_count;
        int side_per_row = itw_start + (tiles_w - itw_end);
        int mid_rows = ith_end - ith_start;
        int mid_count = mid_rows * side_per_row;
        if (rem < mid_count) {
            int row = rem / side_per_row;
            int col = rem % side_per_row;
            th = ith_start + row;
            tw = (col < itw_start) ? col : itw_end + (col - itw_start);
        } else {
            int bot = rem - mid_count;
            th = ith_end + bot / tiles_w;
            tw = bot % tiles_w;
        }
    }

    int h0 = th * 4 - 1, w0 = tw * 4 - 1;
    float d[6][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            int hi = h0 + i, wi = w0 + j;
            d[i][j] = (hi >= 0 && hi < H && wi >= 0 && wi < W)
                          ? __half2float(input[((size_t)n * C + c) * H * W + (size_t)hi * W + wi])
                          : 0.0f;
        }

    float v[6][6];
    bt_transform(d, v);

    int tg = n * total_tiles + th * tiles_w + tw;
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            V[((a * 6 + b) * N * total_tiles + tg) * C + c] = __float2half(v[a][b]);
}

// ── Output transform: interior tiles (no boundary checks) ───────────────

static __global__ void winograd_output_interior(
    const __half* __restrict__ M, const __half* __restrict__ bias,
    __half* __restrict__ output,
    int N, int K, int H, int W, int tiles_h, int tiles_w,
    int oth_end, int otw_end, bool do_relu)
{
    int n_interior = oth_end * otw_end;
    int total_tiles = tiles_h * tiles_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_interior * K) return;

    int k  = idx % K;
    int it = (idx / K) % n_interior;
    int n  = idx / (K * n_interior);
    int th = it / otw_end, tw = it % otw_end;

    int tg = n * total_tiles + th * tiles_w + tw;
    float m[6][6];
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            m[a][b] = __half2float(M[((a * 6 + b) * N * total_tiles + tg) * K + k]);

    float Y[4][4];
    at_transform(m, Y);

    float bv = bias ? __half2float(bias[k]) : 0.0f;
    int oh0 = th * 4, ow0 = tw * 4;
    __half* base = output + ((size_t)n * K + k) * H * W;

    if (!do_relu && (ow0 & 1) == 0) {
        for (int i = 0; i < 4; i++) {
            __half* row = base + (size_t)(oh0 + i) * W + ow0;
            __half2 v01 = __halves2half2(__float2half(Y[i][0] + bv), __float2half(Y[i][1] + bv));
            __half2 v23 = __halves2half2(__float2half(Y[i][2] + bv), __float2half(Y[i][3] + bv));
            *reinterpret_cast<__half2*>(row)     = v01;
            *reinterpret_cast<__half2*>(row + 2) = v23;
        }
    } else {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                float val = Y[i][j] + bv;
                if (do_relu && val < 0.0f) val = 0.0f;
                base[(size_t)(oh0 + i) * W + ow0 + j] = __float2half(val);
            }
    }
}

// ── Output transform: border tiles (with boundary checks) ───────────────

static __global__ void winograd_output_border(
    const __half* __restrict__ M, const __half* __restrict__ bias,
    __half* __restrict__ output,
    int N, int K, int H, int W, int tiles_h, int tiles_w,
    int oth_end, int otw_end, bool do_relu)
{
    int total_tiles = tiles_h * tiles_w;
    int n_border = total_tiles - oth_end * otw_end;
    if (n_border <= 0) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * n_border * K) return;

    int k  = idx % K;
    int bt = (idx / K) % n_border;
    int n  = idx / (K * n_border);

    // Map border index: right strip for th < oth_end, then bottom strip
    int th, tw;
    int right_w = tiles_w - otw_end;
    int top_border = oth_end * right_w;
    if (bt < top_border) {
        th = bt / right_w;
        tw = otw_end + bt % right_w;
    } else {
        int bi = bt - top_border;
        th = oth_end + bi / tiles_w;
        tw = bi % tiles_w;
    }

    int tg = n * total_tiles + th * tiles_w + tw;
    float m[6][6];
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            m[a][b] = __half2float(M[((a * 6 + b) * N * total_tiles + tg) * K + k]);

    float Y[4][4];
    at_transform(m, Y);

    float bv = bias ? __half2float(bias[k]) : 0.0f;
    int oh0 = th * 4, ow0 = tw * 4;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int oh = oh0 + i, ow = ow0 + j;
            if (oh < H && ow < W) {
                float val = Y[i][j] + bv;
                if (do_relu && val < 0.0f) val = 0.0f;
                output[((size_t)n * K + k) * H * W + (size_t)oh * W + ow] = __float2half(val);
            }
        }
}

// ── Fallback: single-kernel paths ───────────────────────────────────────

static __global__ void winograd_input_transform_kernel(
    const __half* __restrict__ input, __half* __restrict__ V,
    int N, int C, int H, int W, int tiles_h, int tiles_w)
{
    int total_tiles = tiles_h * tiles_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * total_tiles * C) return;

    int c = idx % C;
    int tile = (idx / C) % total_tiles;
    int n = idx / (C * total_tiles);
    int th = tile / tiles_w, tw = tile % tiles_w;
    int h0 = th * 4 - 1, w0 = tw * 4 - 1;

    float d[6][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            int hi = h0 + i, wi = w0 + j;
            d[i][j] = (hi >= 0 && hi < H && wi >= 0 && wi < W)
                          ? __half2float(input[((size_t)n * C + c) * H * W + (size_t)hi * W + wi])
                          : 0.0f;
        }

    float v[6][6];
    bt_transform(d, v);

    int tg = n * total_tiles + tile;
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            V[((a * 6 + b) * N * total_tiles + tg) * C + c] = __float2half(v[a][b]);
}

static __global__ void winograd_output_transform_kernel(
    const __half* __restrict__ M, const __half* __restrict__ bias,
    __half* __restrict__ output,
    int N, int K, int H, int W, int tiles_h, int tiles_w, bool do_relu)
{
    int total_tiles = tiles_h * tiles_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * total_tiles * K) return;

    int k = idx % K;
    int tile = (idx / K) % total_tiles;
    int n = idx / (K * total_tiles);
    int th = tile / tiles_w, tw = tile % tiles_w;
    int tg = n * total_tiles + tile;

    float m[6][6];
    for (int a = 0; a < 6; a++)
        for (int b = 0; b < 6; b++)
            m[a][b] = __half2float(M[((a * 6 + b) * N * total_tiles + tg) * K + k]);

    float Y[4][4];
    at_transform(m, Y);

    float bv = bias ? __half2float(bias[k]) : 0.0f;
    int oh0 = th * 4, ow0 = tw * 4;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int oh = oh0 + i, ow = ow0 + j;
            if (oh < H && ow < W) {
                float val = Y[i][j] + bv;
                if (do_relu && val < 0.0f) val = 0.0f;
                output[((size_t)n * K + k) * H * W + (size_t)oh * W + ow] = __float2half(val);
            }
        }
}

// ── Public API ────────────────────────────────────────────────────────────

void WinogradConv::filterTransform(const __half* filter, __half* transformed,
                                    int K, int C, hipStream_t stream) {
    int n = K * C;
    hipLaunchKernelGGL(winograd_filter_transform_kernel,
                       dim3((n + 255) / 256), dim3(256), 0, stream,
                       filter, transformed, K, C);
}

void WinogradConv::forward(const __half* input, const __half* filter_xform,
                            const __half* bias, __half* output,
                            __half* workspace,
                            int N, int C, int H, int W, int K,
                            bool relu,
                            rocblas_handle blas_handle,
                            hipStream_t stream) {
    int tiles_h = (H + 3) / 4;
    int tiles_w = (W + 3) / 4;
    int total_tiles = tiles_h * tiles_w;
    int T = N * total_tiles;

    __half* V = workspace;
    __half* M = workspace + (size_t)36 * T * C;
    constexpr int BLK = 256;

    // Interior tile ranges for input transform:
    // th*4-1 >= 0 → th >= 1;  th*4+4 < H → th < (H-4)/4
    int ith_s = 1, ith_e = (H >= 8) ? (H - 4) / 4 : 0;
    int itw_s = 1, itw_e = (W >= 8) ? (W - 4) / 4 : 0;
    int n_interior = (ith_e > ith_s && itw_e > itw_s)
                         ? (ith_e - ith_s) * (itw_e - itw_s) : 0;

    // Phase 1: Input transform
    if (n_interior >= 64 && n_interior > total_tiles / 3) {
        int n_border = total_tiles - n_interior;
        {
            int ne = N * n_interior * C;
            hipLaunchKernelGGL(winograd_input_interior,
                dim3((ne + BLK-1)/BLK), dim3(BLK), 0, stream,
                input, V, N, C, H, W, tiles_h, tiles_w, ith_s, ith_e, itw_s, itw_e);
        }
        {
            int ne = N * n_border * C;
            hipLaunchKernelGGL(winograd_input_border,
                dim3((ne + BLK-1)/BLK), dim3(BLK), 0, stream,
                input, V, N, C, H, W, tiles_h, tiles_w, ith_s, ith_e, itw_s, itw_e);
        }
    } else {
        int ne = N * total_tiles * C;
        hipLaunchKernelGGL(winograd_input_transform_kernel,
            dim3((ne + BLK-1)/BLK), dim3(BLK), 0, stream,
            input, V, N, C, H, W, tiles_h, tiles_w);
    }

    // Phase 2: Batched GEMM
    {
        const __half alpha_val = __float2half(1.0f);
        const __half beta_val = __float2half(0.0f);
        rocblas_set_stream(blas_handle, stream);
        rocblas_gemm_strided_batched_ex(
            blas_handle,
            rocblas_operation_none, rocblas_operation_none,
            K, T, C, &alpha_val,
            filter_xform, rocblas_datatype_f16_r, K, (rocblas_stride)C * K,
            V, rocblas_datatype_f16_r, C, (rocblas_stride)T * C,
            &beta_val,
            M, rocblas_datatype_f16_r, K, (rocblas_stride)T * K,
            M, rocblas_datatype_f16_r, K, (rocblas_stride)T * K,
            36, rocblas_datatype_f16_r,
            rocblas_gemm_algo_standard, 0, 0);
    }

    // Phase 3: Output transform
    // Interior: th*4+3 < H → th < (H-3)/4;  tw*4+3 < W → tw < (W-3)/4
    int oth_e = (H >= 4) ? (H - 3) / 4 : 0;
    int otw_e = (W >= 4) ? (W - 3) / 4 : 0;
    int on_interior = oth_e * otw_e;

    if (on_interior >= 64 && on_interior > total_tiles / 3) {
        int on_border = total_tiles - on_interior;
        {
            int ne = N * on_interior * K;
            hipLaunchKernelGGL(winograd_output_interior,
                dim3((ne + BLK-1)/BLK), dim3(BLK), 0, stream,
                M, bias, output, N, K, H, W, tiles_h, tiles_w, oth_e, otw_e, relu);
        }
        if (on_border > 0) {
            int ne = N * on_border * K;
            hipLaunchKernelGGL(winograd_output_border,
                dim3((ne + BLK-1)/BLK), dim3(BLK), 0, stream,
                M, bias, output, N, K, H, W, tiles_h, tiles_w, oth_e, otw_e, relu);
        }
    } else {
        int ne = N * total_tiles * K;
        hipLaunchKernelGGL(winograd_output_transform_kernel,
            dim3((ne + BLK-1)/BLK), dim3(BLK), 0, stream,
            M, bias, output, N, K, H, W, tiles_h, tiles_w, relu);
    }
}

size_t WinogradConv::workspaceBytes(int N, int C, int H, int W, int K) {
    int tiles_h = (H + 3) / 4;
    int tiles_w = (W + 3) / 4;
    int T = N * tiles_h * tiles_w;
    return ((size_t)36 * T * C + (size_t)36 * T * K) * sizeof(__half);
}

size_t WinogradConv::filterTransformBytes(int K, int C) {
    return (size_t)36 * K * C * sizeof(__half);
}

}  // namespace kernel
}  // namespace rocm_gpu
}  // namespace ov
