#include "kernels.h"
#include "impl.h"
#include "threadpool.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

/* Packed block-diagonal attention: causal (Qwen) and bidirectional (BERT/GTE)
 * GQA, each with an online-softmax and a BLAS-GEMM path. */

/* Attention operations */

float dot_f32(const float *a, const float *b, int n) { return dot_f32_impl(a, b, n); }

static inline float dot_f32_fast(const float *a, const float *b, int n) {
    return dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void vec_scale_inplace(float *dst, float scale, int n) {
    vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void vec_scale_add(float *dst, const float *src, float correction, int n) {
    vec_scale_add_impl(dst, src, correction, n);
}

#define PACKED_ATTN_QUERY_TILE   32
#define PACKED_ATTN_BLAS_MIN_SEQ 128

static void bidirectional_gqa_attention_packed_online_rows(float *out,
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

            float score = dot_f32_fast(q_row, k_row, head_dim) * scale;

            if (score > max_score) {
                float correction = expf(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                vec_scale_add(o_row, v_row, correction, head_dim);
                max_score = score;
            } else {
                float wt = expf(score - max_score);
                sum_exp += wt;
                vec_axpy_inplace(o_row, v_row, wt, head_dim);
            }
        }

        if (sum_exp > 0.0f) {
            float inv_sum = 1.0f / sum_exp;
            vec_scale_inplace(o_row, inv_sum, head_dim);
        }
    }
}

#ifdef USE_BLAS
static void bidirectional_gqa_attention_packed_blas_rows(float *out,
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

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, keys, head_dim, scale, q, q_hidden, k,
                kv_hidden, 0.0f, scores, keys);
    softmax(scores, rows, keys);
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
    float *scores = t->scores ? t->scores + (size_t)tid * PACKED_ATTN_QUERY_TILE * t->max_seq : NULL;
#endif

    for (int h = 0; h < t->n_heads; h++) {
        for (int b = 0; b < t->batch; b++) {
            int start = t->offsets[b];
            int end = t->offsets[b + 1];
            for (int q0 = start; q0 < end; q0 += PACKED_ATTN_QUERY_TILE) {
                int q1 = q0 + PACKED_ATTN_QUERY_TILE;
                if (q1 > end)
                    q1 = end;
                if (task_id++ % n_threads != tid)
                    continue;
#ifdef USE_BLAS
                if (scores && end - start >= PACKED_ATTN_BLAS_MIN_SEQ) {
                    bidirectional_gqa_attention_packed_blas_rows(
                        t->out, t->Q, t->K, t->V, start, end, t->n_heads, t->n_kv_heads, t->head_dim,
                        t->scale, h, q0, q1, scores);
                    continue;
                }
#endif
                bidirectional_gqa_attention_packed_online_rows(t->out, t->Q, t->K, t->V, start, end,
                                                               t->n_heads, t->n_kv_heads, t->head_dim,
                                                               t->scale, h, q0, q1);
            }
        }
    }
}

size_t bidirectional_gqa_attention_packed_scratch_bytes(const int *offsets, int batch) {
#ifdef USE_BLAS
    if (!offsets || batch <= 0)
        return 0;

    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        int len = offsets[b + 1] - offsets[b];
        if (len > max_seq)
            max_seq = len;
    }
    if (max_seq < PACKED_ATTN_BLAS_MIN_SEQ)
        return 0;
    if ((size_t)max_seq >
        SIZE_MAX / (sizeof(float) * PACKED_ATTN_QUERY_TILE * (size_t)tp_num_threads()))
        return 0;
    return (size_t)tp_num_threads() * PACKED_ATTN_QUERY_TILE * (size_t)max_seq * sizeof(float);
#else
    (void)offsets;
    (void)batch;
    return 0;
#endif
}

void bidirectional_gqa_attention_packed_with_scratch(float *out,
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
    size_t required = bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
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
    if (tp_num_threads() > 1 && n_heads >= 2 && qk_work >= 4096) {
        parallel_for(packed_gqa_attn_worker, &task);
    } else {
        packed_gqa_attn_worker(0, 1, &task);
    }
}

void bidirectional_gqa_attention_packed(float *out,
                                        const float *Q,
                                        const float *K,
                                        const float *V,
                                        const int *offsets,
                                        int batch,
                                        int n_heads,
                                        int n_kv_heads,
                                        int head_dim,
                                        float scale) {
    size_t scratch_bytes = bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    float *scores = scratch_bytes ? malloc(scratch_bytes) : NULL;
    bidirectional_gqa_attention_packed_with_scratch(out, Q, K, V, offsets, batch, n_heads, n_kv_heads,
                                                    head_dim, scale, scores, scratch_bytes);
    free(scores);
}

static void causal_gqa_attention_packed_online_rows(float *out,
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
            float score = dot_f32_fast(q_row, k_row, head_dim) * scale;

            if (score > max_score) {
                float correction = expf(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                vec_scale_add(o_row, v_row, correction, head_dim);
                max_score = score;
            } else {
                float wt = expf(score - max_score);
                sum_exp += wt;
                vec_axpy_inplace(o_row, v_row, wt, head_dim);
            }
        }

        if (sum_exp > 0.0f)
            vec_scale_inplace(o_row, 1.0f / sum_exp, head_dim);
    }
}

#ifdef USE_BLAS
static void causal_gqa_attention_packed_blas_rows(float *out,
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

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, keys, head_dim, scale, q, q_hidden, k,
                kv_hidden, 0.0f, scores, keys);
    for (int r = 0; r < rows; r++) {
        int allowed = query_start + r - start + 1;
        float *row = scores + (size_t)r * keys;
        for (int j = allowed; j < keys; j++)
            row[j] = -INFINITY;
    }
    softmax(scores, rows, keys);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, head_dim, keys, 1.0f, scores, keys,
                v, kv_hidden, 0.0f, o, q_hidden);
}
#endif

static void packed_causal_gqa_attn_worker(int tid, int n_threads, void *arg) {
    packed_gqa_attn_task_t *t = (packed_gqa_attn_task_t *)arg;
    int task_id = 0;
#ifdef USE_BLAS
    float *scores = t->scores ? t->scores + (size_t)tid * PACKED_ATTN_QUERY_TILE * t->max_seq : NULL;
#endif

    for (int h = 0; h < t->n_heads; h++) {
        for (int b = 0; b < t->batch; b++) {
            int start = t->offsets[b];
            int end = t->offsets[b + 1];
            for (int q0 = start; q0 < end; q0 += PACKED_ATTN_QUERY_TILE) {
                int q1 = q0 + PACKED_ATTN_QUERY_TILE;
                if (q1 > end)
                    q1 = end;
                if (task_id++ % n_threads != tid)
                    continue;
#ifdef USE_BLAS
                if (scores && end - start >= PACKED_ATTN_BLAS_MIN_SEQ) {
                    causal_gqa_attention_packed_blas_rows(t->out, t->Q, t->K, t->V, start, t->n_heads,
                                                          t->n_kv_heads, t->head_dim, t->scale, h, q0,
                                                          q1, scores);
                    continue;
                }
#endif
                causal_gqa_attention_packed_online_rows(t->out, t->Q, t->K, t->V, start, t->n_heads,
                                                        t->n_kv_heads, t->head_dim, t->scale, h, q0,
                                                        q1);
            }
        }
    }
}

void causal_gqa_attention_packed_with_scratch(float *out,
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
    size_t required = bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
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
    if (tp_num_threads() > 1 && n_heads >= 2 && qk_work >= 4096)
        parallel_for(packed_causal_gqa_attn_worker, &task);
    else
        packed_causal_gqa_attn_worker(0, 1, &task);
}

void causal_gqa_attention_packed(float *out,
                                 const float *Q,
                                 const float *K,
                                 const float *V,
                                 const int *offsets,
                                 int batch,
                                 int n_heads,
                                 int n_kv_heads,
                                 int head_dim,
                                 float scale) {
    size_t scratch_bytes = bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    float *scores = scratch_bytes ? malloc(scratch_bytes) : NULL;
    causal_gqa_attention_packed_with_scratch(out, Q, K, V, offsets, batch, n_heads, n_kv_heads,
                                             head_dim, scale, scores, scratch_bytes);
    free(scores);
}
