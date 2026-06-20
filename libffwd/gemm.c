/*
 * gemm.c - general Matrix Multiplication
 */

#include "kernels.h"
#include "impl.h"
#include "threadpool.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__AVX512F__) || defined(__AVX2__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#    include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#    include <arm_neon.h>
#endif

#ifdef USE_BLAS
#    ifdef __APPLE__
#        include <Accelerate/Accelerate.h>
#    else
#        include <cblas.h>
#    endif
#endif

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

/* Matrix and vector ops: GEMM, the dtype-dispatch linear kernels, and the
 * BF16-fused matvec/QKV/pair workers, plus the elementwise residual add. */

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void linear(
    float *y, const float *x, const float *W, const float *b, int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    if (seq_len == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim, 1.0f, W, in_dim, x, 1, 0.0f, y, 1);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, seq_len, out_dim, in_dim, 1.0f, x,
                    in_dim, W, in_dim, 0.0f, y, out_dim);
    }
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void linear_nobias(float *y, const float *x, const float *W, int seq_len, int in_dim, int out_dim) {
    linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    size_t i = 0;
    uint32_t *d = (uint32_t *)(void *)dst;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t v = vld1q_u16(src + i);
        uint32x4_t lo = vshlq_n_u32(vmovl_u16(vget_low_u16(v)), 16);
        uint32x4_t hi = vshlq_n_u32(vmovl_u16(vget_high_u16(v)), 16);
        vst1q_u32(d + i, lo);
        vst1q_u32(d + i + 4, hi);
    }
    for (; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
#else
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
#endif
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 */
static void bf16_matvec_fused(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim) {
    bf16_matvec_fused_impl(y, x, W_bf16, bias, in_dim, out_dim);
}

/* Threaded matvec: split output rows across threads */
typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int in_dim;
    int out_dim;
} matvec_task_t;

static void matvec_worker(int tid, int n_threads, void *arg) {
    matvec_task_t *t = (matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim)
        end = t->out_dim;
    if (start >= end)
        return;

    bf16_matvec_fused(t->y + start, t->x, t->W_bf16 + (size_t)start * t->in_dim,
                      t->bias ? t->bias + start : NULL, t->in_dim, end - start);
}

static void bf16_matvec_threaded(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim) {
    if (tp_num_threads() <= 1) {
        bf16_matvec_fused(y, x, W_bf16, bias, in_dim, out_dim);
        return;
    }
    matvec_task_t task = {y, x, W_bf16, bias, in_dim, out_dim};
    parallel_for(matvec_worker, &task);
}

typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int seq_len;
    int in_dim;
    int out_dim;
} bf16_linear_rows_task_t;

static void bf16_linear_rows_worker(int tid, int n_threads, void *arg) {
    bf16_linear_rows_task_t *t = (bf16_linear_rows_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->seq_len)
        end = t->seq_len;
    if (start >= end)
        return;

    for (int s = start; s < end; s++) {
        bf16_matvec_fused(t->y + (size_t)s * t->out_dim, t->x + (size_t)s * t->in_dim, t->W_bf16,
                          t->bias, t->in_dim, t->out_dim);
    }
}

static void bf16_linear_rows(float *y,
                             const float *x,
                             const uint16_t *W_bf16,
                             const float *bias,
                             int seq_len,
                             int in_dim,
                             int out_dim) {
    bf16_linear_rows_task_t task = {
        .y = y,
        .x = x,
        .W_bf16 = W_bf16,
        .bias = bias,
        .seq_len = seq_len,
        .in_dim = in_dim,
        .out_dim = out_dim,
    };

    if (tp_num_threads() > 1 && seq_len >= 2)
        parallel_for(bf16_linear_rows_worker, &task);
    else
        bf16_linear_rows_worker(0, 1, &task);
}

typedef struct {
    float *a;
    float *b;
    const float *x;
    const uint16_t *Wa_bf16;
    const uint16_t *Wb_bf16;
    int seq_len;
    int in_dim;
    int a_dim;
    int b_dim;
    int total_dim;
} pair_matvec_task_t;

static void pair_matvec_range(const pair_matvec_task_t *t, int row, int start, int end) {
    const float *x_row = t->x + (size_t)row * t->in_dim;
    int a_end = t->a_dim;
    int b_end = a_end + t->b_dim;

    if (start < a_end) {
        int s = start;
        int e = end < a_end ? end : a_end;
        if (s < e) {
            bf16_matvec_fused(t->a + (size_t)row * t->a_dim + s, x_row,
                              t->Wa_bf16 + (size_t)s * t->in_dim, NULL, t->in_dim, e - s);
        }
    }

    if (end > a_end && start < b_end) {
        int s = start > a_end ? start - a_end : 0;
        int e_abs = end < b_end ? end : b_end;
        int e = e_abs - a_end;
        if (s < e) {
            bf16_matvec_fused(t->b + (size_t)row * t->b_dim + s, x_row,
                              t->Wb_bf16 + (size_t)s * t->in_dim, NULL, t->in_dim, e - s);
        }
    }
}

static void pair_matvec_worker(int tid, int n_threads, void *arg) {
    pair_matvec_task_t *t = (pair_matvec_task_t *)arg;
    int total_work = t->seq_len * t->total_dim;
    int chunk = (total_work + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > total_work)
        end = total_work;
    if (start >= end)
        return;

    while (start < end) {
        int row = start / t->total_dim;
        int pos = start - row * t->total_dim;
        int row_end = (row + 1) * t->total_dim;
        int stop = end < row_end ? end : row_end;
        pair_matvec_range(t, row, pos, stop - row * t->total_dim);
        start = stop;
    }
}

void linear_nobias_bf16_pair(float *a,
                             float *b,
                             const float *x,
                             const uint16_t *Wa_bf16,
                             const uint16_t *Wb_bf16,
                             int seq_len,
                             int in_dim,
                             int a_dim,
                             int b_dim) {
    if (seq_len <= 0)
        return;

    if (tp_num_threads() <= 1) {
        for (int s = 0; s < seq_len; s++) {
            const float *x_row = x + (size_t)s * in_dim;
            bf16_matvec_fused(a + (size_t)s * a_dim, x_row, Wa_bf16, NULL, in_dim, a_dim);
            bf16_matvec_fused(b + (size_t)s * b_dim, x_row, Wb_bf16, NULL, in_dim, b_dim);
        }
        return;
    }

    pair_matvec_task_t task = {
        .a = a,
        .b = b,
        .x = x,
        .Wa_bf16 = Wa_bf16,
        .Wb_bf16 = Wb_bf16,
        .seq_len = seq_len,
        .in_dim = in_dim,
        .a_dim = a_dim,
        .b_dim = b_dim,
        .total_dim = a_dim + b_dim,
    };
    parallel_for(pair_matvec_worker, &task);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    const float *x;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    int in_dim;
    int q_dim;
    int kv_dim;
    int seq_len;
    int total_dim;
} qkv_matvec_task_t;

static void qkv_matvec_range(const qkv_matvec_task_t *t, int row, int start, int end) {
    const float *x_row = t->x + (size_t)row * t->in_dim;
    int q_end = t->q_dim;
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            bf16_matvec_fused(t->q + (size_t)row * t->q_dim + s, x_row,
                              t->Wq_bf16 + (size_t)s * t->in_dim, NULL, t->in_dim, e - s);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            bf16_matvec_fused(t->k + (size_t)row * t->kv_dim + s, x_row,
                              t->Wk_bf16 + (size_t)s * t->in_dim, NULL, t->in_dim, e - s);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            bf16_matvec_fused(t->v + (size_t)row * t->kv_dim + s, x_row,
                              t->Wv_bf16 + (size_t)s * t->in_dim, NULL, t->in_dim, e - s);
        }
    }
}

static void qkv_matvec_worker(int tid, int n_threads, void *arg) {
    qkv_matvec_task_t *t = (qkv_matvec_task_t *)arg;
    int total_work = t->seq_len * t->total_dim;
    int chunk = (total_work + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > total_work)
        end = total_work;
    if (start >= end)
        return;

    while (start < end) {
        int row = start / t->total_dim;
        int pos = start - row * t->total_dim;
        int row_end = (row + 1) * t->total_dim;
        int stop = end < row_end ? end : row_end;
        qkv_matvec_range(t, row, pos, stop - row * t->total_dim);
        start = stop;
    }
}

void linear_nobias_bf16_qkv(float *q,
                            float *k,
                            float *v,
                            const float *x,
                            const uint16_t *Wq_bf16,
                            const uint16_t *Wk_bf16,
                            const uint16_t *Wv_bf16,
                            int seq_len,
                            int in_dim,
                            int q_dim,
                            int kv_dim) {
    if (seq_len <= 0)
        return;

    if (tp_num_threads() <= 1) {
        for (int s = 0; s < seq_len; s++) {
            const float *x_row = x + (size_t)s * in_dim;
            bf16_matvec_fused(q + (size_t)s * q_dim, x_row, Wq_bf16, NULL, in_dim, q_dim);
            bf16_matvec_fused(k + (size_t)s * kv_dim, x_row, Wk_bf16, NULL, in_dim, kv_dim);
            bf16_matvec_fused(v + (size_t)s * kv_dim, x_row, Wv_bf16, NULL, in_dim, kv_dim);
        }
        return;
    }

    qkv_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x = x,
        .Wq_bf16 = Wq_bf16,
        .Wk_bf16 = Wk_bf16,
        .Wv_bf16 = Wv_bf16,
        .in_dim = in_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .seq_len = seq_len,
        .total_dim = q_dim + 2 * kv_dim,
    };
    parallel_for(qkv_matvec_worker, &task);
}

void linear_nobias_bf16(
    float *y, const float *x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    /* Callers route longer sequences through an F32-widened weight matrix in
     * caller-owned scratch; this per-row path stays correct for any length. */
    bf16_linear_rows(y, x, W_bf16, NULL, seq_len, in_dim, out_dim);
}
