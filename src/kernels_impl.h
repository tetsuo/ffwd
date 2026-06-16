/*
 * embed_kernels_impl.h - internal architecture dispatch for hot kernels
 */

#ifndef EMBED_KERNELS_IMPL_H
#define EMBED_KERNELS_IMPL_H

#include <stdint.h>

void embed_bf16_matvec_fused_generic(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim);
float embed_dot_f32_generic(const float *a, const float *b, int n);
void embed_vec_scale_inplace_generic(float *dst, float scale, int n);
void embed_vec_axpy_inplace_generic(float *dst, const float *src, float alpha, int n);
void embed_vec_scale_add_generic(float *dst, const float *src, float correction, int n);

#ifdef __ARM_NEON
void embed_bf16_matvec_fused_neon(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim);
float embed_dot_f32_neon(const float *a, const float *b, int n);
void embed_vec_scale_inplace_neon(float *dst, float scale, int n);
void embed_vec_axpy_inplace_neon(float *dst, const float *src, float alpha, int n);
void embed_vec_scale_add_neon(float *dst, const float *src, float correction, int n);

#define embed_bf16_matvec_fused_impl embed_bf16_matvec_fused_neon
#define embed_dot_f32_impl           embed_dot_f32_neon
#define embed_vec_scale_inplace_impl embed_vec_scale_inplace_neon
#define embed_vec_axpy_inplace_impl  embed_vec_axpy_inplace_neon
#define embed_vec_scale_add_impl     embed_vec_scale_add_neon

#elif defined(__AVX2__) && defined(__FMA__)
void embed_bf16_matvec_fused_avx(
    float *y, const float *x, const uint16_t *W_bf16, const float *bias, int in_dim, int out_dim);
float embed_dot_f32_avx(const float *a, const float *b, int n);
void embed_vec_scale_inplace_avx(float *dst, float scale, int n);
void embed_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n);
void embed_vec_scale_add_avx(float *dst, const float *src, float correction, int n);

#define embed_bf16_matvec_fused_impl embed_bf16_matvec_fused_avx
#define embed_dot_f32_impl           embed_dot_f32_avx
#define embed_vec_scale_inplace_impl embed_vec_scale_inplace_avx
#define embed_vec_axpy_inplace_impl  embed_vec_axpy_inplace_avx
#define embed_vec_scale_add_impl     embed_vec_scale_add_avx

#else
#define embed_bf16_matvec_fused_impl embed_bf16_matvec_fused_generic
#define embed_dot_f32_impl           embed_dot_f32_generic
#define embed_vec_scale_inplace_impl embed_vec_scale_inplace_generic
#define embed_vec_axpy_inplace_impl  embed_vec_axpy_inplace_generic
#define embed_vec_scale_add_impl     embed_vec_scale_add_generic
#endif

#endif /* EMBED_KERNELS_IMPL_H */
