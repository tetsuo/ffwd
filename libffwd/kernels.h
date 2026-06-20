/*
 * kernels.h - tkern public API: transformer inference kernels (CPU, float32)
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void add_inplace(float *a, const float *b, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
void matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out] */
void linear(
    float *y, const float *x, const float *W, const float *b, int seq_len, int in_dim, int out_dim);

void linear_nobias(
    float *y, const float *x, const float *W, int seq_len, int in_dim, int out_dim);

/* bf16 weight variants */

void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n);
void linear_nobias_bf16(
    float *y, const float *x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim);

/* Small-sequence BF16 path: compute two projections with one threaded dispatch. */
void linear_nobias_bf16_pair(float *a,
                                   float *b,
                                   const float *x,
                                   const uint16_t *Wa_bf16,
                                   const uint16_t *Wb_bf16,
                                   int seq_len,
                                   int in_dim,
                                   int a_dim,
                                   int b_dim);

/* Small-sequence BF16 path: compute Q/K/V matvecs with one threaded dispatch. */
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
                                  int kv_dim);

/* Dot product using the best available local SIMD implementation. */
float dot_f32(const float *a, const float *b, int n);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* RMS Normalization: out = x / rms(x) * weight */
void rms_norm(
    float *out, const float *x, const float *weight, int seq_len, int hidden, float eps);

/* Per-head RMS Normalization for Q/K norms in decoder
 * x: [seq, n_heads, head_dim], weight: [head_dim]
 * Normalizes each head independently */
void rms_norm_per_head(
    float *x, const float *weight, int seq_len, int n_heads, int head_dim, float eps);

/* Layer Normalization (BERT family): out = gamma * (x - mean) / sqrt(var + eps)
 * + beta, with mean and biased (population) variance over the hidden axis per
 * row. Unlike RMSNorm this subtracts the mean and adds a bias. */
void layer_norm(float *out,
                      const float *x,
                      const float *gamma,
                      const float *beta,
                      int seq_len,
                      int hidden,
                      float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void softmax(float *x, int rows, int cols);
/* gate = SiLU(gate) * up */
void silu_mul_inplace(float *gate, const float *up, int n);

/* Exact (erf) GeLU in place: x = 0.5 * x * (1 + erf(x / sqrt(2))). This is the
 * "gelu" activation HF BERT/BGE/MiniLM use, not the tanh approximation. */
void gelu_inplace(float *x, int n);

/* Tanh-approximation GeLU in place (HF gelu_new / gelu_pytorch_tanh):
 * x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). */
void gelu_tanh_inplace(float *x, int n);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Bidirectional GQA attention over a packed/ragged batch.
 *
 * Q: [total_seq, n_heads * head_dim]
 * K,V: [total_seq, n_kv_heads * head_dim]
 * offsets: [batch + 1], token spans for each sequence.
 *
 * Attention is block-diagonal: tokens only attend within their own
 * offsets[b]..offsets[b+1] span.
 */
void bidirectional_gqa_attention_packed(float *out,
                                              const float *Q,
                                              const float *K,
                                              const float *V,
                                              const int *offsets,
                                              int batch,
                                              int n_heads,
                                              int n_kv_heads,
                                              int head_dim,
                                              float scale);

/* Optional reusable scratch for the BLAS-tiled packed attention path. */
size_t bidirectional_gqa_attention_packed_scratch_bytes(const int *offsets, int batch);
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
                                                           size_t scratch_bytes);

/*
 * Causal GQA attention over a packed/ragged batch. Query position i attends
 * only to positions at or before i within its sequence.
 */
void causal_gqa_attention_packed(float *out,
                                       const float *Q,
                                       const float *K,
                                       const float *V,
                                       const int *offsets,
                                       int batch,
                                       int n_heads,
                                       int n_kv_heads,
                                       int head_dim,
                                       float scale);
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
                                                    size_t scratch_bytes);

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

/*
 * NeoX-style RoPE: compute cos/sin for positions.
 * cos_out, sin_out: [seq, head_dim]
 * cos[d] and cos[half+d] are the same (duplicated for full head_dim).
 */
void compute_rope_neox(
    float *cos_out, float *sin_out, const int *positions, int seq, int head_dim, float theta);

/*
 * Apply NeoX-style RoPE to Q or K.
 * x: [seq, n_heads * head_dim] (in-place)
 * cos_vals, sin_vals: [seq, head_dim]
 */
void apply_rope_neox(
    float *x, const float *cos_vals, const float *sin_vals, int seq, int n_heads, int head_dim);

#endif /* KERNELS_H */
