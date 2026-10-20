/*
 * Copyright (c) 2026-present, ETRI, All rights reserved.
 *
 * Host (CPU) experts for opt_prime.accel.moe: SwiGLU experts with stacked bf16 weights
 *   gate_up[E][2F][H] (gate rows 0..F-1, up rows F..2F-1), down[E][H][F]
 * out[r] += w[r][j] * down[e] @ (silu(gate_up[e][:F] @ x[r]) * (gate_up[e][F:] @ x[r]))   for each pair (r, j),
 * e = idx[r][j], skipping pairs with w == 0.  moe_host_forward_bw rounds w[r][j] * (...) to bf16 before accumulating,
 * as torch does when the routing weights are bf16 (the torch tier and HF eager experts multiply in the weights' dtype).  Memory bound for decode: each thread streams its own contiguous weight
 * rows (AVX-512 BF16 dot products, fp32 accumulation).  Build: gcc -O3 -march=native -fopenmp -shared -fPIC.
 */
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) {            /* round to nearest even */
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static inline float dot_bf16(const uint16_t *a, const uint16_t *b, int n) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512(a + i + 32), (__m512bh)_mm512_loadu_si512(b + i + 32));
    }
    for (; i + 32 <= n; i += 32)
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512(a + i), (__m512bh)_mm512_loadu_si512(b + i));
    float s = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < n; i++) s += bf2f(a[i]) * bf2f(b[i]);
    return s;
}

int moe_host_supported(void) { return __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512f"); }

/* first-touch copy: pages of dst land on the NUMA node of the thread that will stream them (same static split) */
void moe_host_place(uint16_t *dst, const uint16_t *src, int64_t rows, int64_t cols, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t r = 0; r < rows; r++) memcpy(dst + r * cols, src + r * cols, (size_t)cols * 2);
}

/* x: [R][H] bf16, idx: [R][K] int64 (< E), w: [R][K] fp32, out: [R][H] fp32 (accumulated into).
 * Pairs are grouped by expert so each expert's weights are streamed once for all of its rows. */
static void host_forward(int R, int K, int H, int F, int E, const uint16_t *x, const int64_t *idx, const float *w,
                         const uint16_t *gate_up, const uint16_t *down, float *out, int threads, int bf16_product) {
    int P = R * K;
    int *cnt = calloc((size_t)E + 1, sizeof(int));
    int *ord = malloc(sizeof(int) * (size_t)(P > 0 ? P : 1));
    for (int p = 0; p < P; p++) if (w[p] != 0.0f) cnt[idx[p] + 1]++;
    for (int e = 0; e < E; e++) cnt[e + 1] += cnt[e];
    int *fill = malloc(sizeof(int) * (size_t)E);
    memcpy(fill, cnt, sizeof(int) * (size_t)E);
    for (int p = 0; p < P; p++) if (w[p] != 0.0f) ord[fill[idx[p]]++] = p;
    int maxn = 0;
    for (int e = 0; e < E; e++) if (cnt[e + 1] - cnt[e] > maxn) maxn = cnt[e + 1] - cnt[e];
    float *g = malloc(sizeof(float) * (size_t)(maxn ? maxn : 1) * 2 * F);
    uint16_t *hb = malloc(sizeof(uint16_t) * (size_t)(maxn ? maxn : 1) * F);
    for (int e = 0; e < E; e++) {
        int n = cnt[e + 1] - cnt[e];
        if (!n) continue;
        const int *pp = ord + cnt[e];
        const uint16_t *gu = gate_up + (size_t)e * 2 * F * H;
        const uint16_t *dn = down + (size_t)e * H * F;
        #pragma omp parallel num_threads(threads)
        {
            #pragma omp for schedule(static)
            for (int o = 0; o < 2 * F; o++)
                for (int i = 0; i < n; i++)
                    g[(size_t)i * 2 * F + o] = bf2f(f2bf(dot_bf16(gu + (size_t)o * H, x + (size_t)(pp[i] / K) * H, H)));
            #pragma omp for schedule(static)
            for (int f = 0; f < F; f++)          /* silu(gate) * up, rounded like bf16 tensors */
                for (int i = 0; i < n; i++) {
                    float a = g[(size_t)i * 2 * F + f], s = bf2f(f2bf(a / (1.0f + expf(-a))));
                    hb[(size_t)i * F + f] = f2bf(s * g[(size_t)i * 2 * F + F + f]);
                }
            #pragma omp for schedule(static)
            for (int o = 0; o < H; o++)          /* each thread owns output columns o: no write conflicts */
                for (int i = 0; i < n; i++) {
                    int p = pp[i];
                    float y = bf2f(f2bf(dot_bf16(dn + (size_t)o * F, hb + (size_t)i * F, F)));
                    float c = w[p] * y;
                    out[(size_t)(p / K) * H + o] += bf16_product ? bf2f(f2bf(c)) : c;
                }
        }
    }
    free(cnt); free(ord); free(fill); free(g); free(hb);
}

void moe_host_forward(int R, int K, int H, int F, int E, const uint16_t *x, const int64_t *idx, const float *w,
                      const uint16_t *gate_up, const uint16_t *down, float *out, int threads) {
    host_forward(R, K, H, F, E, x, idx, w, gate_up, down, out, threads, 0);
}

void moe_host_forward_bw(int R, int K, int H, int F, int E, const uint16_t *x, const int64_t *idx, const float *w,
                         const uint16_t *gate_up, const uint16_t *down, float *out, int threads) {
    host_forward(R, K, H, F, E, x, idx, w, gate_up, down, out, threads, 1);
}
