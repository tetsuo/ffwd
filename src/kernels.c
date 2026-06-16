#include "kernels.h"
#include "kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#if (defined(__AVX512F__) || defined(__AVX2__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#if defined(USE_OPENBLAS)
void openblas_set_num_threads(int num_threads);
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

#define EMBED_MAX_THREADS 16

typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[EMBED_MAX_THREADS - 1];
    int tids[EMBED_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

void embed_set_threads(int n) {
    if (n < 1)
        n = 1;
    if (n > EMBED_MAX_THREADS)
        n = EMBED_MAX_THREADS;

#if defined(USE_OPENBLAS)
    /* OpenBLAS owns the heavy F32 GEMM parallelism on Linux. Keep the
     * auxiliary pool serial to avoid nested thread contention. */
    int blas_threads = n;
    openblas_set_num_threads(blas_threads);
    n = 1;
#endif

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
    if (n <= 1) {
        if (embed_verbose >= 2) {
#if defined(USE_OPENBLAS)
            fprintf(stderr, "Thread pool: %d threads, OpenBLAS: %d threads\n", n, blas_threads);
#else
            fprintf(stderr, "Thread pool: %d threads\n", n);
#endif
        }
        return;
    }

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, worker_loop, &tp.tids[i]);
    }

    if (embed_verbose >= 2) {
#if defined(USE_OPENBLAS)
        fprintf(stderr, "Thread pool: %d threads, OpenBLAS: %d threads\n", n, blas_threads);
#else
        fprintf(stderr, "Thread pool: %d threads\n", n);
#endif
    }
}

#ifndef __APPLE__
/* CPU bandwidth this process may actually use, in whole cores, or 0 if there
 * is no cgroup limit. Containers (Docker --cpus, Kubernetes, RunPod) expose
 * every host CPU through sysconf()/cpuset, but the CFS quota caps the CPU time
 * the process really gets. Sizing the BLAS/thread pool to the host count then
 * oversubscribes that quota: the threads spend the period's budget early and
 * the scheduler throttles them for the rest of each period, which adds large,
 * bursty stalls (e.g. a 16-thread default against a 5.1-core quota ran the
 * 0.6B model ~2x slower than a 5-thread pool). Floor, not ceil, so a fully
 * busy pool stays under the quota and never trips throttling. */
static int cgroup_cpu_quota(void) {
    /* cgroup v2: "/sys/fs/cgroup/cpu.max" is "<quota> <period>", or
     * "max <period>" when unlimited. */
    FILE *f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char quota[32];
        long period = 0;
        int got = fscanf(f, "%31s %ld", quota, &period);
        fclose(f);
        if (got == 2 && period > 0 && strcmp(quota, "max") != 0) {
            long q = atol(quota);
            if (q > 0)
                return q / period < 1 ? 1 : (int)(q / period);
        }
    }
    /* cgroup v1: separate files; quota -1 means unlimited. */
    long q = -1, period = 0;
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    if (f) {
        if (fscanf(f, "%ld", &q) != 1)
            q = -1;
        fclose(f);
    }
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
    if (f) {
        if (fscanf(f, "%ld", &period) != 1)
            period = 0;
        fclose(f);
    }
    if (q > 0 && period > 0)
        return q / period < 1 ? 1 : (int)(q / period);
    return 0;
}
#endif

int embed_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int online = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1)
        online = 1;
    /* Honor a cgroup CPU quota when it is tighter than the visible core count,
     * so the default thread pool does not oversubscribe a CPU-limited
     * container and get throttled. */
    int quota = cgroup_cpu_quota();
    if (quota >= 1 && quota < online)
        return quota;
    return online;
#endif
}

/* Dispatch work to all threads; main thread is tid=0 */
static void parallel_for(parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void embed_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void embed_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
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

void embed_linear(float *y,
                 const float *x,
                 const float *W,
                 const float *b,
                 int seq_len,
                 int in_dim,
                 int out_dim) {
#ifdef USE_BLAS
    if (seq_len == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim, 1.0f, W, in_dim, x, 1, 0.0f, y,
                    1);
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

void embed_linear_nobias(
    float *y, const float *x, const float *W, int seq_len, int in_dim, int out_dim) {
    embed_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
void embed_bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
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
    embed_bf16_matvec_fused_impl(y, x, W_bf16, bias, in_dim, out_dim);
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
    if (tp.n_threads <= 1) {
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

    if (tp.n_threads > 1 && seq_len >= 2)
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

void embed_linear_nobias_bf16_pair(float *a,
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

    if (tp.n_threads <= 1) {
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

void embed_linear_nobias_bf16_qkv(float *q,
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

    if (tp.n_threads <= 1) {
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

void embed_linear_nobias_bf16(
    float *y, const float *x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    /* Callers route longer sequences through an F32-widened weight matrix in
     * caller-owned scratch (see embed.c); this per-row path stays correct for
     * any length. */
    bf16_linear_rows(y, x, W_bf16, NULL, seq_len, in_dim, out_dim);
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* Rows split across the pool only when the tensor is big enough to pay
 * for the dispatch. */
#define EMBED_RMS_NORM_PARALLEL_ELEMS 262144

static void embed_rms_norm_range(
    float *out, const float *x, const float *weight, int start, int end, int hidden, float eps) {
    for (int s = start; s < end; s++) {
        const float *x_row = x + (size_t)s * hidden;
        float *out_row = out + (size_t)s * hidden;

#if defined(__AVX512F__) && defined(__FMA__)
        __m512 accv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            __m512 v = _mm512_loadu_ps(x_row + i);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float sum_sq = _mm512_reduce_add_ps(accv);
        for (; i < hidden; i++)
            sum_sq += x_row[i] * x_row[i];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 v = _mm256_loadu_ps(x_row + i);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float sum_sq = _mm_cvtss_f32(acc128);
        for (; i < hidden; i++)
            sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#if defined(__AVX512F__)
        __m512 scale = _mm512_set1_ps(rms_inv);
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 vx = _mm512_loadu_ps(x_row + j);
            __m512 vw = _mm512_loadu_ps(weight + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(_mm512_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++)
            out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
        __m256 scale = _mm256_set1_ps(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x_row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++)
            out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

typedef struct {
    float *out;
    const float *x;
    const float *weight;
    int seq_len;
    int hidden;
    float eps;
} rms_norm_task_t;

static void rms_norm_worker(int tid, int n_threads, void *arg) {
    rms_norm_task_t *t = (rms_norm_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->seq_len)
        end = t->seq_len;
    if (start >= end)
        return;
    embed_rms_norm_range(t->out, t->x, t->weight, start, end, t->hidden, t->eps);
}

void embed_rms_norm(
    float *out, const float *x, const float *weight, int seq_len, int hidden, float eps) {
    if (seq_len <= 0 || hidden <= 0)
        return;

    long long elems = (long long)seq_len * hidden;
    if (tp.n_threads > 1 && elems >= EMBED_RMS_NORM_PARALLEL_ELEMS) {
        rms_norm_task_t task = {
            .out = out,
            .x = x,
            .weight = weight,
            .seq_len = seq_len,
            .hidden = hidden,
            .eps = eps,
        };
        parallel_for(rms_norm_worker, &task);
    } else {
        embed_rms_norm_range(out, x, weight, 0, seq_len, hidden, eps);
    }
}

void embed_rms_norm_per_head(
    float *x, const float *weight, int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            __m512 accv = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v = _mm512_loadu_ps(vec + d);
                accv = _mm512_fmadd_ps(v, v, accv);
            }
            float sum_sq = _mm512_reduce_add_ps(accv);
            for (; d < head_dim; d++)
                sum_sq += vec[d] * vec[d];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 accv = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(vec + d);
                accv = _mm256_fmadd_ps(v, v, accv);
            }
            __m128 acc128 =
                _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
            acc128 = _mm_hadd_ps(acc128, acc128);
            acc128 = _mm_hadd_ps(acc128, acc128);
            float sum_sq = _mm_cvtss_f32(acc128);
            for (; d < head_dim; d++)
                sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#if defined(__AVX512F__)
            __m512 scale = _mm512_set1_ps(rms_inv);
            int j = 0;
            for (; j + 16 <= head_dim; j += 16) {
                __m512 v = _mm512_loadu_ps(vec + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                _mm512_storeu_ps(vec + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++)
                vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
            __m256 scale = _mm256_set1_ps(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 v = _mm256_loadu_ps(vec + j);
                __m256 w = _mm256_loadu_ps(weight + j);
                _mm256_storeu_ps(vec + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++)
                vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* Mean-subtracting LayerNorm with bias. Safe in place (out may alias x): each
 * row's mean and variance are reduced before the row is rewritten. Scalar for
 * now; LayerNorm is not GEMM-bound, so SIMD/threading is a later perf step. */
void embed_layer_norm(float *out,
                     const float *x,
                     const float *gamma,
                     const float *beta,
                     int seq_len,
                     int hidden,
                     float eps) {
    if (seq_len <= 0 || hidden <= 0)
        return;
    float inv_h = 1.0f / (float)hidden;
    for (int s = 0; s < seq_len; s++) {
        const float *row = x + (size_t)s * hidden;
        float *dst = out + (size_t)s * hidden;
        float mean = 0.0f;
        for (int d = 0; d < hidden; d++)
            mean += row[d];
        mean *= inv_h;
        float var = 0.0f;
        for (int d = 0; d < hidden; d++) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        var *= inv_h;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < hidden; d++)
            dst[d] = gamma[d] * (row[d] - mean) * inv_std + beta[d];
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void embed_silu_mul_inplace(float *gate, const float *up, int n) {
#if defined(__APPLE__) && defined(USE_BLAS)
    enum { CHUNK = 4096 };
    float exp_neg[CHUNK];

    for (int base = 0; base < n; base += CHUNK) {
        int len = n - base;
        if (len > CHUNK)
            len = CHUNK;

        for (int i = 0; i < len; i++)
            exp_neg[i] = -gate[base + i];
        vvexpf(exp_neg, exp_neg, &len);

        for (int i = 0; i < len; i++) {
            float g = gate[base + i];
            gate[base + i] = (g / (1.0f + exp_neg[i])) * up[base + i];
        }
    }
#else
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
#endif
}

void embed_gelu_inplace(float *x, int n) {
    /* Exact erf GeLU; 0.70710678... = 1/sqrt(2). */
    for (int i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}

void embed_gelu_tanh_inplace(float *x, int n) {
    /* Tanh-approximation GeLU; 0.79788456... = sqrt(2/pi). */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = 0.79788456080286536f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void embed_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val)
                max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

float embed_dot_f32(const float *a, const float *b, int n) { return embed_dot_f32_impl(a, b, n); }

static inline float embed_dot_f32_fast(const float *a, const float *b, int n) {
    return embed_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void embed_vec_scale_inplace(float *dst, float scale, int n) {
    embed_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void embed_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    embed_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void embed_vec_scale_add(float *dst, const float *src, float correction, int n) {
    embed_vec_scale_add_impl(dst, src, correction, n);
}

#define EMBED_PACKED_ATTN_QUERY_TILE   32
#define EMBED_PACKED_ATTN_BLAS_MIN_SEQ 128

static void embed_bidirectional_gqa_attention_packed_online_rows(float *out,
                                                                const float *Q,
                                                                const float *K,
                                                                const float *V,
                                                                int start,
                                                                int end,
                                                                int n_heads,
                                                                int n_kv_heads,
                                                                int head_dim,
                                                                float scale,
                                                                int h,
                                                                int query_start,
                                                                int query_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    int kv_h = h / heads_per_kv;

    for (int i = query_start; i < query_end; i++) {
        const float *q_row = Q + (size_t)i * q_hidden + h * head_dim;
        float *o_row = out + (size_t)i * q_hidden + h * head_dim;

        float max_score = -1e30f;
        float sum_exp = 0.0f;
        for (int d = 0; d < head_dim; d++)
            o_row[d] = 0.0f;

        for (int j = start; j < end; j++) {
            const float *k_row = K + (size_t)j * kv_hidden + kv_h * head_dim;
            const float *v_row = V + (size_t)j * kv_hidden + kv_h * head_dim;

            float score = embed_dot_f32_fast(q_row, k_row, head_dim) * scale;

            if (score > max_score) {
                float correction = expf(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                embed_vec_scale_add(o_row, v_row, correction, head_dim);
                max_score = score;
            } else {
                float wt = expf(score - max_score);
                sum_exp += wt;
                embed_vec_axpy_inplace(o_row, v_row, wt, head_dim);
            }
        }

        if (sum_exp > 0.0f) {
            float inv_sum = 1.0f / sum_exp;
            embed_vec_scale_inplace(o_row, inv_sum, head_dim);
        }
    }
}

#ifdef USE_BLAS
static void embed_bidirectional_gqa_attention_packed_blas_rows(float *out,
                                                              const float *Q,
                                                              const float *K,
                                                              const float *V,
                                                              int start,
                                                              int end,
                                                              int n_heads,
                                                              int n_kv_heads,
                                                              int head_dim,
                                                              float scale,
                                                              int h,
                                                              int query_start,
                                                              int query_end,
                                                              float *scores) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    int kv_h = h / heads_per_kv;
    int rows = query_end - query_start;
    int keys = end - start;

    const float *q = Q + (size_t)query_start * q_hidden + h * head_dim;
    const float *k = K + (size_t)start * kv_hidden + kv_h * head_dim;
    const float *v = V + (size_t)start * kv_hidden + kv_h * head_dim;
    float *o = out + (size_t)query_start * q_hidden + h * head_dim;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, keys, head_dim, scale, q, q_hidden,
                k, kv_hidden, 0.0f, scores, keys);
    embed_softmax(scores, rows, keys);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, head_dim, keys, 1.0f, scores, keys,
                v, kv_hidden, 0.0f, o, q_hidden);
}
#endif

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    const int *offsets;
    int batch;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    float *scores;
    int max_seq;
} packed_gqa_attn_task_t;

static void packed_gqa_attn_worker(int tid, int n_threads, void *arg) {
    packed_gqa_attn_task_t *t = (packed_gqa_attn_task_t *)arg;
    int task_id = 0;
#ifdef USE_BLAS
    float *scores =
        t->scores ? t->scores + (size_t)tid * EMBED_PACKED_ATTN_QUERY_TILE * t->max_seq : NULL;
#endif

    for (int h = 0; h < t->n_heads; h++) {
        for (int b = 0; b < t->batch; b++) {
            int start = t->offsets[b];
            int end = t->offsets[b + 1];
            for (int q0 = start; q0 < end; q0 += EMBED_PACKED_ATTN_QUERY_TILE) {
                int q1 = q0 + EMBED_PACKED_ATTN_QUERY_TILE;
                if (q1 > end)
                    q1 = end;
                if (task_id++ % n_threads != tid)
                    continue;
#ifdef USE_BLAS
                if (scores && end - start >= EMBED_PACKED_ATTN_BLAS_MIN_SEQ) {
                    embed_bidirectional_gqa_attention_packed_blas_rows(
                        t->out, t->Q, t->K, t->V, start, end, t->n_heads, t->n_kv_heads,
                        t->head_dim, t->scale, h, q0, q1, scores);
                    continue;
                }
#endif
                embed_bidirectional_gqa_attention_packed_online_rows(
                    t->out, t->Q, t->K, t->V, start, end, t->n_heads, t->n_kv_heads, t->head_dim,
                    t->scale, h, q0, q1);
            }
        }
    }
}

size_t embed_bidirectional_gqa_attention_packed_scratch_bytes(const int *offsets, int batch) {
#ifdef USE_BLAS
    if (!offsets || batch <= 0)
        return 0;

    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        int len = offsets[b + 1] - offsets[b];
        if (len > max_seq)
            max_seq = len;
    }
    if (max_seq < EMBED_PACKED_ATTN_BLAS_MIN_SEQ)
        return 0;
    if ((size_t)max_seq >
        SIZE_MAX / (sizeof(float) * EMBED_PACKED_ATTN_QUERY_TILE * (size_t)tp.n_threads))
        return 0;
    return (size_t)tp.n_threads * EMBED_PACKED_ATTN_QUERY_TILE * (size_t)max_seq * sizeof(float);
#else
    (void)offsets;
    (void)batch;
    return 0;
#endif
}

void embed_bidirectional_gqa_attention_packed_with_scratch(float *out,
                                                          const float *Q,
                                                          const float *K,
                                                          const float *V,
                                                          const int *offsets,
                                                          int batch,
                                                          int n_heads,
                                                          int n_kv_heads,
                                                          int head_dim,
                                                          float scale,
                                                          float *scratch,
                                                          size_t scratch_bytes) {
    long long qk_work = 0;
    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        int len = offsets[b + 1] - offsets[b];
        qk_work += (long long)len * len * n_heads;
        if (len > max_seq)
            max_seq = len;
    }

    float *scores = NULL;
#ifdef USE_BLAS
    size_t required = embed_bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    if (required != 0 && scratch && scratch_bytes >= required)
        scores = scratch;
#else
    (void)scratch;
    (void)scratch_bytes;
#endif
    packed_gqa_attn_task_t task = {.out = out,
                                   .Q = Q,
                                   .K = K,
                                   .V = V,
                                   .offsets = offsets,
                                   .batch = batch,
                                   .n_heads = n_heads,
                                   .n_kv_heads = n_kv_heads,
                                   .head_dim = head_dim,
                                   .scale = scale,
                                   .scores = scores,
                                   .max_seq = max_seq};
    if (tp.n_threads > 1 && n_heads >= 2 && qk_work >= 4096) {
        parallel_for(packed_gqa_attn_worker, &task);
    } else {
        packed_gqa_attn_worker(0, 1, &task);
    }
}

void embed_bidirectional_gqa_attention_packed(float *out,
                                             const float *Q,
                                             const float *K,
                                             const float *V,
                                             const int *offsets,
                                             int batch,
                                             int n_heads,
                                             int n_kv_heads,
                                             int head_dim,
                                             float scale) {
    size_t scratch_bytes = embed_bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    float *scores = scratch_bytes ? malloc(scratch_bytes) : NULL;
    embed_bidirectional_gqa_attention_packed_with_scratch(
        out, Q, K, V, offsets, batch, n_heads, n_kv_heads, head_dim, scale, scores, scratch_bytes);
    free(scores);
}

static void embed_causal_gqa_attention_packed_online_rows(float *out,
                                                         const float *Q,
                                                         const float *K,
                                                         const float *V,
                                                         int start,
                                                         int n_heads,
                                                         int n_kv_heads,
                                                         int head_dim,
                                                         float scale,
                                                         int h,
                                                         int query_start,
                                                         int query_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    int kv_h = h / heads_per_kv;

    for (int i = query_start; i < query_end; i++) {
        const float *q_row = Q + (size_t)i * q_hidden + h * head_dim;
        float *o_row = out + (size_t)i * q_hidden + h * head_dim;

        float max_score = -1e30f;
        float sum_exp = 0.0f;
        for (int d = 0; d < head_dim; d++)
            o_row[d] = 0.0f;

        for (int j = start; j <= i; j++) {
            const float *k_row = K + (size_t)j * kv_hidden + kv_h * head_dim;
            const float *v_row = V + (size_t)j * kv_hidden + kv_h * head_dim;
            float score = embed_dot_f32_fast(q_row, k_row, head_dim) * scale;

            if (score > max_score) {
                float correction = expf(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                embed_vec_scale_add(o_row, v_row, correction, head_dim);
                max_score = score;
            } else {
                float wt = expf(score - max_score);
                sum_exp += wt;
                embed_vec_axpy_inplace(o_row, v_row, wt, head_dim);
            }
        }

        if (sum_exp > 0.0f)
            embed_vec_scale_inplace(o_row, 1.0f / sum_exp, head_dim);
    }
}

#ifdef USE_BLAS
static void embed_causal_gqa_attention_packed_blas_rows(float *out,
                                                       const float *Q,
                                                       const float *K,
                                                       const float *V,
                                                       int start,
                                                       int n_heads,
                                                       int n_kv_heads,
                                                       int head_dim,
                                                       float scale,
                                                       int h,
                                                       int query_start,
                                                       int query_end,
                                                       float *scores) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    int kv_h = h / heads_per_kv;
    int rows = query_end - query_start;
    int keys = query_end - start;

    const float *q = Q + (size_t)query_start * q_hidden + h * head_dim;
    const float *k = K + (size_t)start * kv_hidden + kv_h * head_dim;
    const float *v = V + (size_t)start * kv_hidden + kv_h * head_dim;
    float *o = out + (size_t)query_start * q_hidden + h * head_dim;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, keys, head_dim, scale, q, q_hidden,
                k, kv_hidden, 0.0f, scores, keys);
    for (int r = 0; r < rows; r++) {
        int allowed = query_start + r - start + 1;
        float *row = scores + (size_t)r * keys;
        for (int j = allowed; j < keys; j++)
            row[j] = -INFINITY;
    }
    embed_softmax(scores, rows, keys);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, head_dim, keys, 1.0f, scores, keys,
                v, kv_hidden, 0.0f, o, q_hidden);
}
#endif

static void packed_causal_gqa_attn_worker(int tid, int n_threads, void *arg) {
    packed_gqa_attn_task_t *t = (packed_gqa_attn_task_t *)arg;
    int task_id = 0;
#ifdef USE_BLAS
    float *scores =
        t->scores ? t->scores + (size_t)tid * EMBED_PACKED_ATTN_QUERY_TILE * t->max_seq : NULL;
#endif

    for (int h = 0; h < t->n_heads; h++) {
        for (int b = 0; b < t->batch; b++) {
            int start = t->offsets[b];
            int end = t->offsets[b + 1];
            for (int q0 = start; q0 < end; q0 += EMBED_PACKED_ATTN_QUERY_TILE) {
                int q1 = q0 + EMBED_PACKED_ATTN_QUERY_TILE;
                if (q1 > end)
                    q1 = end;
                if (task_id++ % n_threads != tid)
                    continue;
#ifdef USE_BLAS
                if (scores && end - start >= EMBED_PACKED_ATTN_BLAS_MIN_SEQ) {
                    embed_causal_gqa_attention_packed_blas_rows(
                        t->out, t->Q, t->K, t->V, start, t->n_heads, t->n_kv_heads, t->head_dim,
                        t->scale, h, q0, q1, scores);
                    continue;
                }
#endif
                embed_causal_gqa_attention_packed_online_rows(t->out, t->Q, t->K, t->V, start,
                                                             t->n_heads, t->n_kv_heads, t->head_dim,
                                                             t->scale, h, q0, q1);
            }
        }
    }
}

void embed_causal_gqa_attention_packed_with_scratch(float *out,
                                                   const float *Q,
                                                   const float *K,
                                                   const float *V,
                                                   const int *offsets,
                                                   int batch,
                                                   int n_heads,
                                                   int n_kv_heads,
                                                   int head_dim,
                                                   float scale,
                                                   float *scratch,
                                                   size_t scratch_bytes) {
    long long qk_work = 0;
    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        int len = offsets[b + 1] - offsets[b];
        qk_work += (long long)len * (len + 1) / 2 * n_heads;
        if (len > max_seq)
            max_seq = len;
    }

    float *scores = NULL;
#ifdef USE_BLAS
    size_t required = embed_bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    if (required != 0 && scratch && scratch_bytes >= required)
        scores = scratch;
#else
    (void)scratch;
    (void)scratch_bytes;
#endif
    packed_gqa_attn_task_t task = {.out = out,
                                   .Q = Q,
                                   .K = K,
                                   .V = V,
                                   .offsets = offsets,
                                   .batch = batch,
                                   .n_heads = n_heads,
                                   .n_kv_heads = n_kv_heads,
                                   .head_dim = head_dim,
                                   .scale = scale,
                                   .scores = scores,
                                   .max_seq = max_seq};
    if (tp.n_threads > 1 && n_heads >= 2 && qk_work >= 4096)
        parallel_for(packed_causal_gqa_attn_worker, &task);
    else
        packed_causal_gqa_attn_worker(0, 1, &task);
}

void embed_causal_gqa_attention_packed(float *out,
                                      const float *Q,
                                      const float *K,
                                      const float *V,
                                      const int *offsets,
                                      int batch,
                                      int n_heads,
                                      int n_kv_heads,
                                      int head_dim,
                                      float scale) {
    size_t scratch_bytes = embed_bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    float *scores = scratch_bytes ? malloc(scratch_bytes) : NULL;
    embed_causal_gqa_attention_packed_with_scratch(out, Q, K, V, offsets, batch, n_heads, n_kv_heads,
                                                  head_dim, scale, scores, scratch_bytes);
    free(scores);
}

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

void embed_compute_rope_neox(
    float *cos_out, float *sin_out, const int *positions, int seq, int head_dim, float theta) {
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Duplicate for full head_dim */
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

void embed_apply_rope_neox(
    float *x, const float *cos_vals, const float *sin_vals, int seq, int n_heads, int head_dim) {
    /*
     * NeoX split-half style:
     *   x1 = x[..., :half], x2 = x[..., half:]
     *   rotated = cat(-x2, x1)
     *   result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            int d = 0;
            for (; d + 16 <= half; d += 16) {
                __m512 x1 = _mm512_loadu_ps(vec + d);
                __m512 x2 = _mm512_loadu_ps(vec + half + d);
                /* RoPE cache duplicates cos/sin across halves. */
                __m512 cc = _mm512_loadu_ps(c + d);
                __m512 ss = _mm512_loadu_ps(sn + d);
                __m512 new1 = _mm512_fmsub_ps(x1, cc, _mm512_mul_ps(x2, ss));
                __m512 new2 = _mm512_fmadd_ps(x2, cc, _mm512_mul_ps(x1, ss));
                _mm512_storeu_ps(vec + d, new1);
                _mm512_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d] = x1 * c[d] + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#elif defined(__AVX2__) && defined(__FMA__)
            int d = 0;
            for (; d + 8 <= half; d += 8) {
                __m256 x1 = _mm256_loadu_ps(vec + d);
                __m256 x2 = _mm256_loadu_ps(vec + half + d);
                __m256 cc = _mm256_loadu_ps(c + d);
                __m256 ss = _mm256_loadu_ps(sn + d);
                __m256 new1 = _mm256_fmsub_ps(x1, cc, _mm256_mul_ps(x2, ss));
                __m256 new2 = _mm256_fmadd_ps(x2, cc, _mm256_mul_ps(x1, ss));
                _mm256_storeu_ps(vec + d, new1);
                _mm256_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d] = x1 * c[d] + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
            int d = 0;
            for (; d + 4 <= half; d += 4) {
                float32x4_t x1 = vld1q_f32(vec + d);
                float32x4_t x2 = vld1q_f32(vec + half + d);
                float32x4_t cc = vld1q_f32(c + d);
                float32x4_t ss = vld1q_f32(sn + d);
                float32x4_t new1 = vmlsq_f32(vmulq_f32(x1, cc), x2, ss);
                float32x4_t new2 = vfmaq_f32(vmulq_f32(x2, cc), x1, ss);
                vst1q_f32(vec + d, new1);
                vst1q_f32(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d] = x1 * c[d] + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#else
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];        /* first half */
                float x2 = vec[half + d]; /* second half */
                vec[d] = x1 * c[d] + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#endif
        }
    }
}
