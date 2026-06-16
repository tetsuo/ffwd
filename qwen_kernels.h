/*
 * qwen_kernels.h - Math kernels for Qwen3-architecture inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 */

#ifndef QWEN_KERNELS_H
#define QWEN_KERNELS_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying EMBED_API are exported
 * from the shared library; everything else stays internal. */
#ifndef EMBED_API
#if defined(__GNUC__)
#define EMBED_API __attribute__((visibility("default")))
#else
#define EMBED_API
#endif
#endif

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void qwen_add_inplace(float *a, const float *b, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* C = A @ B^T: A[M,K], B[N,K], C[M,N] */
void qwen_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/* y = x @ W^T + b: x[seq,in], W[out,in], b[out], y[seq,out] */
void qwen_linear(
    float *y, const float *x, const float *W, const float *b, int seq_len, int in_dim, int out_dim);

void qwen_linear_nobias(
    float *y, const float *x, const float *W, int seq_len, int in_dim, int out_dim);

/* bf16 weight variants */

void qwen_bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n);
void qwen_linear_nobias_bf16(
    float *y, const float *x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim);

/* Small-sequence BF16 path: compute two projections with one threaded dispatch. */
void qwen_linear_nobias_bf16_pair(float *a,
                                  float *b,
                                  const float *x,
                                  const uint16_t *Wa_bf16,
                                  const uint16_t *Wb_bf16,
                                  int seq_len,
                                  int in_dim,
                                  int a_dim,
                                  int b_dim);

/* Small-sequence BF16 path: compute Q/K/V matvecs with one threaded dispatch. */
void qwen_linear_nobias_bf16_qkv(float *q,
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
float qwen_dot_f32(const float *a, const float *b, int n);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/* RMS Normalization: out = x / rms(x) * weight */
void qwen_rms_norm(
    float *out, const float *x, const float *weight, int seq_len, int hidden, float eps);

/* Per-head RMS Normalization for Q/K norms in decoder
 * x: [seq, n_heads, head_dim], weight: [head_dim]
 * Normalizes each head independently */
void qwen_rms_norm_per_head(
    float *x, const float *weight, int seq_len, int n_heads, int head_dim, float eps);

/* Layer Normalization (BERT family): out = gamma * (x - mean) / sqrt(var + eps)
 * + beta, with mean and biased (population) variance over the hidden axis per
 * row. Unlike RMSNorm this subtracts the mean and adds a bias. */
void qwen_layer_norm(float *out,
                     const float *x,
                     const float *gamma,
                     const float *beta,
                     int seq_len,
                     int hidden,
                     float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void qwen_softmax(float *x, int rows, int cols);
/* gate = SiLU(gate) * up */
void qwen_silu_mul_inplace(float *gate, const float *up, int n);

/* Exact (erf) GeLU in place: x = 0.5 * x * (1 + erf(x / sqrt(2))). This is the
 * "gelu" activation HF BERT/BGE/MiniLM use, not the tanh approximation. */
void qwen_gelu_inplace(float *x, int n);

/* Tanh-approximation GeLU in place (HF gelu_new / gelu_pytorch_tanh):
 * x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). */
void qwen_gelu_tanh_inplace(float *x, int n);

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
void qwen_bidirectional_gqa_attention_packed(float *out,
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
size_t qwen_bidirectional_gqa_attention_packed_scratch_bytes(const int *offsets, int batch);
void qwen_bidirectional_gqa_attention_packed_with_scratch(float *out,
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
void qwen_causal_gqa_attention_packed(float *out,
                                      const float *Q,
                                      const float *K,
                                      const float *V,
                                      const int *offsets,
                                      int batch,
                                      int n_heads,
                                      int n_kv_heads,
                                      int head_dim,
                                      float scale);
void qwen_causal_gqa_attention_packed_with_scratch(float *out,
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
void qwen_compute_rope_neox(
    float *cos_out, float *sin_out, const int *positions, int seq, int head_dim, float theta);

/*
 * Apply NeoX-style RoPE to Q or K.
 * x: [seq, n_heads * head_dim] (in-place)
 * cos_vals, sin_vals: [seq, head_dim]
 */
void qwen_apply_rope_neox(
    float *x, const float *cos_vals, const float *sin_vals, int seq, int n_heads, int head_dim);

/* ========================================================================
 * Threading
 * ======================================================================== */

/* Set number of threads for parallel operations (default: 1).
 * Creates a persistent thread pool. Call before inference. */
EMBED_API void qwen_set_threads(int n);

/* Get number of available CPU cores */
EMBED_API int qwen_get_num_cpus(void);

/* Global verbose flag */
EMBED_API extern int qwen_verbose;

#endif /* QWEN_KERNELS_H */
