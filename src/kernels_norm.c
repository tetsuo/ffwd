#include "kernels.h"
#include "kernels_impl.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__AVX512F__) || defined(__AVX2__)) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Normalization kernels: RMSNorm (Qwen) and LayerNorm (BERT), plus the
 * per-head RMSNorm used for QK-norm. */

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
    if (embed_num_threads() > 1 && elems >= EMBED_RMS_NORM_PARALLEL_ELEMS) {
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
