// Copyright (c) 2026-present, ETRI, All rights reserved.
//
// Native host dispatcher for the blocking tiers of one offloaded MoE layer (decode fast path).
//
// One call splits the routed pairs of a few rows by owning tier, hands every blocking tier its rows on a persistent
// worker thread, and sums the partial outputs into one float32 buffer - all without the Python GIL. A tier is any
// C function with the signature below; built-in tiers: the native host kernel (moe_tier_cpu) and an expert server
// reached over a pipe (moe_tier_pipe, protocol of kernels/rngd_opt/template/src/bin/server.rs.tpl, op 4).
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// out[r, :] = sum over the row's pairs of w * expert_idx(x[r]); idx/w are [R, k] (padding pairs have w = 0)
typedef int (*moe_tier_fn)(void *ctx, int R, int H, const uint16_t *x, int k, const int64_t *idx, const float *w,
                           float *out);

#define MAX_TIERS 8
#define SPIN_NS 300000L                 // workers spin this long for the next job before sleeping

typedef struct {
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    atomic_int state;                   // 0 idle, 1 job posted, 2 done
    int stop;
    moe_tier_fn fn;
    void *ctx;
    int R, H, k, rc;
    uint16_t *x;
    int64_t *idx;
    float *w, *out;
    int32_t *rows;                      // row ids (into the call's rows) this job covers
    size_t cap_r, cap_rk, cap_rh;       // allocated rows / row*k / row*hidden
} job_t;

typedef struct {
    job_t jobs[MAX_TIERS];              // job t always runs tier t (a tier keeps its thread and OpenMP team)
    int posted[MAX_TIERS], n;           // tiers posted by the last start
} disp_t;

static long now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000L + t.tv_nsec;
}

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    for (;;) {
        long t0 = now_ns();
        while (atomic_load_explicit(&j->state, memory_order_acquire) != 1) {
            if (j->stop) return NULL;
            if (now_ns() - t0 > SPIN_NS) {
                pthread_mutex_lock(&j->mu);
                while (atomic_load(&j->state) != 1 && !j->stop) pthread_cond_wait(&j->cv, &j->mu);
                pthread_mutex_unlock(&j->mu);
                if (j->stop) return NULL;
            }
        }
        j->rc = j->fn(j->ctx, j->R, j->H, j->x, j->k, j->idx, j->w, j->out);
        atomic_store_explicit(&j->state, 2, memory_order_release);
    }
}

void *moe_disp_new(void) {
    disp_t *d = calloc(1, sizeof(disp_t));
    for (int i = 0; i < MAX_TIERS; i++) {
        job_t *j = &d->jobs[i];
        pthread_mutex_init(&j->mu, NULL);
        pthread_cond_init(&j->cv, NULL);
        atomic_init(&j->state, 0);
        pthread_create(&j->th, NULL, worker, j);
    }
    return d;
}

static void reserve(job_t *j, int R, int k, int H) {
    if ((size_t)R > j->cap_r) {
        free(j->rows);
        j->rows = malloc(sizeof(int32_t) * R);
        j->cap_r = R;
    }
    if ((size_t)R * H > j->cap_rh) {
        free(j->x); free(j->out);
        j->cap_rh = (size_t)R * H;
        j->x = malloc(sizeof(uint16_t) * j->cap_rh);
        j->out = malloc(sizeof(float) * j->cap_rh);
    }
    if ((size_t)(R * k) > j->cap_rk) {
        free(j->idx); free(j->w);
        j->idx = malloc(sizeof(int64_t) * R * k);
        j->w = malloc(sizeof(float) * R * k);
        j->cap_rk = R * k;
    }
}

// Split the [R, k] routed pairs by owner[expert]; post one job per blocking tier with pairs. counts[t] += pairs routed
// to tier t (all tiers). Returns the number of jobs posted (they run while the caller launches device work).
int moe_disp_start(void *dp, int R, int k, int H, const int64_t *idx, const float *w, const uint16_t *x,
                   const int32_t *owner, const int32_t *local, int ntiers, const int8_t *blocking, void *const *fns,
                   void *const *ctxs, int64_t *counts) {
    disp_t *d = (disp_t *)dp;
    int njobs = 0, active[MAX_TIERS] = {0};
    if (ntiers > MAX_TIERS) return -1;
    // pass 1: which blocking tiers have pairs; per-row pair count per tier -> k per tier
    int kmax[MAX_TIERS] = {0}, nrows[MAX_TIERS] = {0};
    for (int r = 0; r < R; r++) {
        int cnt[MAX_TIERS] = {0};
        for (int j = 0; j < k; j++) {
            int64_t e = idx[r * k + j];
            int t = e < 0 ? ntiers : owner[e];          // owner has num_experts + 1 entries; padding -> ntiers
            if (t >= ntiers) continue;
            counts[t]++;
            if (blocking[t] && w[r * k + j] != 0.0f) cnt[t]++;
        }
        for (int t = 0; t < ntiers; t++)
            if (cnt[t]) { nrows[t]++; if (cnt[t] > kmax[t]) kmax[t] = cnt[t]; }
    }
    // jobs of a previous call that was never waited for (the caller raised) may still read their buffers
    for (int i = 0; i < d->n; i++) {
        job_t *jb = &d->jobs[d->posted[i]];
        while (atomic_load_explicit(&jb->state, memory_order_acquire) != 2) { }
        atomic_store(&jb->state, 0);
    }
    d->n = 0;
    for (int t = 0; t < ntiers; t++) {
        if (!nrows[t]) continue;
        job_t *jb = &d->jobs[t];
        reserve(jb, nrows[t], kmax[t], H);
        jb->fn = (moe_tier_fn)fns[t]; jb->ctx = ctxs[t];
        jb->R = 0; jb->H = H; jb->k = kmax[t];
        active[t] = 1; d->posted[njobs++] = t;
    }
    // pass 2: fill each job's rows, x rows and padded [R_t, k_t] pairs
    for (int r = 0; r < R; r++) {
        for (int t = 0; t < ntiers; t++) {
            if (!active[t]) continue;
            job_t *jb = &d->jobs[t];
            int c = 0;
            int64_t *ir = jb->idx + (size_t)jb->R * jb->k;
            float *wr = jb->w + (size_t)jb->R * jb->k;
            for (int j = 0; j < k; j++) {
                int64_t e = idx[r * k + j];
                if (e >= 0 && owner[e] == t && w[r * k + j] != 0.0f) { ir[c] = local[e]; wr[c] = w[r * k + j]; c++; }
            }
            if (!c) continue;
            for (int j = c; j < jb->k; j++) { ir[j] = ir[0]; wr[j] = 0.0f; }
            memcpy(jb->x + (size_t)jb->R * H, x + (size_t)r * H, sizeof(uint16_t) * H);
            jb->rows[jb->R++] = r;
        }
    }
    for (int i = 0; i < njobs; i++) {
        job_t *jb = &d->jobs[d->posted[i]];
        pthread_mutex_lock(&jb->mu);
        atomic_store_explicit(&jb->state, 1, memory_order_release);
        pthread_cond_signal(&jb->cv);
        pthread_mutex_unlock(&jb->mu);
    }
    d->n = njobs;
    return njobs;
}

// Wait for the posted jobs and sum their outputs into acc [R, H] (overwritten). Returns 0, or the first tier error.
int moe_disp_wait(void *dp, int R, int H, float *acc) {
    disp_t *d = (disp_t *)dp;
    int rc = 0;
    memset(acc, 0, sizeof(float) * R * H);
    for (int i = 0; i < d->n; i++) {
        job_t *jb = &d->jobs[d->posted[i]];
        while (atomic_load_explicit(&jb->state, memory_order_acquire) != 2) { }
        atomic_store(&jb->state, 0);
        if (jb->rc && !rc) rc = jb->rc;
        for (int q = 0; q < jb->R; q++) {
            float *dst = acc + (size_t)jb->rows[q] * H, *src = jb->out + (size_t)q * H;
            for (int h = 0; h < H; h++) dst[h] += src[h];
        }
    }
    d->n = 0;
    return rc;
}

// ---------------- built-in tiers ----------------

// native host kernel (csrc/host_experts.c), called through a function pointer from that library
typedef void (*host_fwd_t)(int, int, int, int, int, const uint16_t *, const int64_t *, const float *, const uint16_t *,
                           const uint16_t *, float *, int);
typedef struct { host_fwd_t fwd; const uint16_t *gu, *dn; int E, F, threads; } cpu_ctx_t;

void *moe_cpu_ctx_new(void *fwd, void *gu, void *dn, int E, int F, int threads) {
    cpu_ctx_t *c = malloc(sizeof(cpu_ctx_t));
    c->fwd = (host_fwd_t)fwd; c->gu = gu; c->dn = dn; c->E = E; c->F = F; c->threads = threads;
    return c;
}

int moe_tier_cpu(void *ctx, int R, int H, const uint16_t *x, int k, const int64_t *idx, const float *w, float *out) {
    cpu_ctx_t *c = (cpu_ctx_t *)ctx;
    memset(out, 0, sizeof(float) * R * H);
    c->fwd(R, k, H, c->F, c->E, x, idx, w, c->gu, c->dn, out, c->threads);
    return 0;
}

// expert server over a pipe; one pipe_t per server process (shared by its layers), guarded by a mutex that the
// Python side also takes for load / update
typedef struct { int wfd, rfd, K; pthread_mutex_t mu; } pipe_t;
typedef struct { pipe_t *p; int layer; } pipe_ctx_t;

void *moe_pipe_new(int wfd, int rfd, int K) {
    pipe_t *p = malloc(sizeof(pipe_t));
    p->wfd = wfd; p->rfd = rfd; p->K = K;
    pthread_mutex_init(&p->mu, NULL);
    return p;
}
void moe_pipe_lock(void *p) { pthread_mutex_lock(&((pipe_t *)p)->mu); }
void moe_pipe_unlock(void *p) { pthread_mutex_unlock(&((pipe_t *)p)->mu); }

void *moe_pipe_ctx_new(void *p, int layer) {
    pipe_ctx_t *c = malloc(sizeof(pipe_ctx_t));
    c->p = (pipe_t *)p; c->layer = layer;
    return c;
}

static int write_all(int fd, const void *buf, size_t n) {
    const char *b = buf;
    while (n) { ssize_t m = write(fd, b, n); if (m <= 0) return -1; b += m; n -= m; }
    return 0;
}
static int read_all(int fd, void *buf, size_t n) {
    char *b = buf;
    while (n) { ssize_t m = read(fd, b, n); if (m <= 0) return -1; b += m; n -= m; }
    return 0;
}

int moe_tier_pipe(void *ctx, int R, int H, const uint16_t *x, int k, const int64_t *idx, const float *w, float *out) {
    pipe_ctx_t *c = (pipe_ctx_t *)ctx;
    pipe_t *p = c->p;
    uint32_t msg[3 + 2 * 64];
    float *res = malloc(sizeof(float) * H);
    int rc = 0;
    pthread_mutex_lock(&p->mu);
    for (int r = 0; r < R && !rc; r++) {
        // merge duplicate slots of the row, drop weight-0 padding
        int64_t sl[64]; float sw[64]; int n = 0;
        for (int j = 0; j < k; j++) {
            float a = w[r * k + j];
            if (a == 0.0f) continue;
            int f = -1;
            for (int q = 0; q < n; q++) if (sl[q] == idx[r * k + j]) { f = q; break; }
            if (f >= 0) sw[f] += a; else { sl[n] = idx[r * k + j]; sw[n++] = a; }
        }
        float *o = out + (size_t)r * H;
        memset(o, 0, sizeof(float) * H);
        for (int c0 = 0; c0 < n && !rc; c0 += p->K) {
            int m = n - c0 < p->K ? n - c0 : p->K;
            msg[0] = 4; msg[1] = c->layer; msg[2] = m;
            for (int q = 0; q < m; q++) { msg[3 + q] = (uint32_t)sl[c0 + q]; memcpy(&msg[3 + m + q], &sw[c0 + q], 4); }
            if (write_all(p->wfd, msg, 4 * (3 + 2 * m)) || write_all(p->wfd, x + (size_t)r * H, 2 * H) ||
                read_all(p->rfd, res, 4 * H)) { rc = -1; break; }
            for (int h = 0; h < H; h++) o[h] += res[h];
        }
    }
    pthread_mutex_unlock(&p->mu);
    free(res);
    return rc;
}
