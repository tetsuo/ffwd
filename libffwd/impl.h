/*
 * impl.h - internal architecture dispatch for hot kernels
 *
 * The small vector primitives (dot product, scale, axpy, scale-add, scaled
 * multiply) are defined here as static inline so they fold directly into the
 * caller's loop: a row of RMSNorm or attention issues no call and the compiler
 * keeps the row in registers across the reduction and the write-back. The
 * generic scalar versions stay out of line in generic.c as a reference for the
 * kernel tests and as the fallback on platforms with no SIMD path.
 *
 * bf16_matvec_fused is large (a full matrix-vector product) and is called once
 * per output tile rather than inside a tight loop, so it stays out of line.
 */

#ifndef KERNELS_IMPL_H
#define KERNELS_IMPL_H

#include <stdint.h>

/* Scalar references (generic.c): the kernel tests compare the SIMD results
 * against these, and the no-SIMD build dispatches to them directly. */
void bf16_matvec_fused_generic(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim);
float dot_f32_generic(const float *a, const float *b, int n);
void vec_scale_inplace_generic(float *dst, float scale, int n);
void vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);
void vec_scale_add_generic(float *dst, const float *src, float correction, int n);

#ifdef __ARM_NEON

#    include <arm_neon.h>

void bf16_matvec_fused_neon(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim);
#    define bf16_matvec_fused_impl bf16_matvec_fused_neon

static inline float dot_f32_impl(const float *a, const float *b, int n) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

static inline void vec_scale_inplace_impl(float *dst, float scale, int n) {
    int i = 0;
    float32x4_t s = vdupq_n_f32(scale);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(vdupq_n_f32(0.0f), d0, s));
        vst1q_f32(dst + i + 4, vfmaq_f32(vdupq_n_f32(0.0f), d1, s));
    }
    for (; i < n; i++)
        dst[i] *= scale;
}

static inline void vec_axpy_inplace_impl(float *dst, const float *src, float alpha, int n) {
    int i = 0;
    float32x4_t a = vdupq_n_f32(alpha);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(d0, s0, a));
        vst1q_f32(dst + i + 4, vfmaq_f32(d1, s1, a));
    }
    for (; i < n; i++)
        dst[i] += alpha * src[i];
}

static inline void vec_scale_add_impl(float *dst, const float *src, float correction, int n) {
    int i = 0;
    float32x4_t c = vdupq_n_f32(correction);
    for (; i + 8 <= n; i += 8) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t s0 = vld1q_f32(src + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t s1 = vld1q_f32(src + i + 4);
        vst1q_f32(dst + i, vfmaq_f32(s0, d0, c));
        vst1q_f32(dst + i + 4, vfmaq_f32(s1, d1, c));
    }
    for (; i < n; i++)
        dst[i] = dst[i] * correction + src[i];
}

#elif defined(__AVX2__) && defined(__FMA__)

#    include <immintrin.h>

void bf16_matvec_fused_avx(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim);
#    define bf16_matvec_fused_impl bf16_matvec_fused_avx

static inline float dot_f32_impl(const float *a, const float *b, int n) {
#    if defined(__AVX512F__)
    int i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#    else
    int i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float sum = _mm_cvtss_f32(r);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#    endif
}

static inline void vec_scale_inplace_impl(float *dst, float scale, int n) {
#    if defined(__AVX512F__)
    int i = 0;
    __m512 s = _mm512_set1_ps(scale);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(dst + i), s));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), s));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), s));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), s));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(dst + i), s));
    }
    for (; i < n; i++)
        dst[i] *= scale;
#    else
    int i = 0;
    __m256 s = _mm256_set1_ps(scale);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), s));
        _mm256_storeu_ps(dst + i + 8, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 8), s));
        _mm256_storeu_ps(dst + i + 16, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 16), s));
        _mm256_storeu_ps(dst + i + 24, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 24), s));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), s));
    }
    for (; i < n; i++)
        dst[i] *= scale;
#    endif
}

static inline void vec_axpy_inplace_impl(float *dst, const float *src, float alpha, int n) {
#    if defined(__AVX512F__)
    int i = 0;
    __m512 a = _mm512_set1_ps(alpha);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), a,
                                                       _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), a,
                                                       _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), a,
                                                       _mm512_loadu_ps(dst + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
    }
    for (; i < n; i++)
        dst[i] += alpha * src[i];
#    else
    int i = 0;
    __m256 a = _mm256_set1_ps(alpha);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), a, _mm256_loadu_ps(dst + i)));
        _mm256_storeu_ps(dst + i + 8, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 8), a,
                                                      _mm256_loadu_ps(dst + i + 8)));
        _mm256_storeu_ps(dst + i + 16, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 16), a,
                                                       _mm256_loadu_ps(dst + i + 16)));
        _mm256_storeu_ps(dst + i + 24, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 24), a,
                                                       _mm256_loadu_ps(dst + i + 24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), a, _mm256_loadu_ps(dst + i)));
    }
    for (; i < n; i++)
        dst[i] += alpha * src[i];
#    endif
}

static inline void vec_scale_add_impl(float *dst, const float *src, float correction, int n) {
#    if defined(__AVX512F__)
    int i = 0;
    __m512 c = _mm512_set1_ps(correction);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 16), c,
                                                       _mm512_loadu_ps(src + i + 16)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 32), c,
                                                       _mm512_loadu_ps(src + i + 32)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 48), c,
                                                       _mm512_loadu_ps(src + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
    }
    for (; i < n; i++)
        dst[i] = dst[i] * correction + src[i];
#    else
    int i = 0;
    __m256 c = _mm256_set1_ps(correction);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(dst + i), c, _mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i + 8, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 8), c,
                                                      _mm256_loadu_ps(src + i + 8)));
        _mm256_storeu_ps(dst + i + 16, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 16), c,
                                                       _mm256_loadu_ps(src + i + 16)));
        _mm256_storeu_ps(dst + i + 24, _mm256_fmadd_ps(_mm256_loadu_ps(dst + i + 24), c,
                                                       _mm256_loadu_ps(src + i + 24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(dst + i), c, _mm256_loadu_ps(src + i)));
    }
    for (; i < n; i++)
        dst[i] = dst[i] * correction + src[i];
#    endif
}

#else /* no SIMD path: dispatch to the scalar references */

#    define bf16_matvec_fused_impl bf16_matvec_fused_generic
#    define dot_f32_impl           dot_f32_generic
#    define vec_scale_inplace_impl vec_scale_inplace_generic
#    define vec_axpy_inplace_impl  vec_axpy_inplace_generic
#    define vec_scale_add_impl     vec_scale_add_generic

#endif

#endif /* KERNELS_IMPL_H */
