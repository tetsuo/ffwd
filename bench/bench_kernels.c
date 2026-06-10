/* bench/bench_kernels.c - regression microbenchmarks for the hot kernels.
 *
 * Shapes are the real pplx-embed-v1-0.6b dimensions (hidden 1024, 16 query /
 * 8 KV heads x head_dim 128, intermediate 3072) so a slowdown here predicts
 * a slowdown in inference. Single-threaded by default for stable numbers;
 * --threads N opts into the pool. Build and record via `make bench`,
 * compare records with scripts/benchstat.py.
 */

#include "bench.h"
#include "qwen_kernels.h"

#include <stdint.h>

int qwen_verbose = 0;   /* defined by embed.c in full builds */

enum {
    HIDDEN = 1024,
    Q_DIM = 2048,        /* 16 heads * 128 */
    KV_DIM = 1024,       /*  8 heads * 128 */
    HEAD_DIM = 128,
    INTER = 3072,
    SEQ = 8,
    ATTN_SEQ = 128,
};

/* Deterministic fill in [-1, 1]; small values keep softmax/silu sane. */
static void fill_f32(float *x, size_t n, unsigned seed)
{
    unsigned s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
}

static uint16_t f32_to_bf16(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (uint16_t)(u >> 16);
}

static void fill_bf16(uint16_t *x, size_t n, unsigned seed)
{
    unsigned s = seed * 2654435761u + 1u;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = f32_to_bf16((float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f);
    }
}

static float *alloc_f32(size_t n, unsigned seed)
{
    float *x = (float *)malloc(n * sizeof(float));
    if (!x) exit(2);
    fill_f32(x, n, seed);
    return x;
}

static uint16_t *alloc_bf16(size_t n, unsigned seed)
{
    uint16_t *x = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!x) exit(2);
    fill_bf16(x, n, seed);
    return x;
}

/* ---- F32 projections (matvec at seq 1, GEMM at seq 8) ---- */

static void bm_linear_f32(bench_state_t *b, int seq, int in, int out)
{
    float *x = alloc_f32((size_t)seq * in, 1);
    float *w = alloc_f32((size_t)out * in, 2);
    float *y = alloc_f32((size_t)seq * out, 3);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_linear_nobias(y, x, w, seq, in, out);
    bench_sink += y[0];
    free(y); free(w); free(x);
}

static void bm_linear_f32_s1_qproj(bench_state_t *b)  { bm_linear_f32(b, 1, HIDDEN, Q_DIM); }
static void bm_linear_f32_s1_gate(bench_state_t *b)   { bm_linear_f32(b, 1, HIDDEN, INTER); }
static void bm_linear_f32_s1_down(bench_state_t *b)   { bm_linear_f32(b, 1, INTER, HIDDEN); }
static void bm_linear_f32_s8_gate(bench_state_t *b)   { bm_linear_f32(b, SEQ, HIDDEN, INTER); }

/* ---- BF16 weights ---- */

static void bm_linear_bf16(bench_state_t *b, int seq, int in, int out)
{
    float *x = alloc_f32((size_t)seq * in, 4);
    uint16_t *w = alloc_bf16((size_t)out * in, 5);
    float *y = alloc_f32((size_t)seq * out, 6);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_linear_nobias_bf16(y, x, w, seq, in, out);
    bench_sink += y[0];
    free(y); free(w); free(x);
}

static void bm_linear_bf16_s1_gate(bench_state_t *b) { bm_linear_bf16(b, 1, HIDDEN, INTER); }
static void bm_linear_bf16_s8_gate(bench_state_t *b) { bm_linear_bf16(b, SEQ, HIDDEN, INTER); }

static void bm_qkv_bf16(bench_state_t *b)
{
    float *x = alloc_f32((size_t)SEQ * HIDDEN, 7);
    uint16_t *wq = alloc_bf16((size_t)Q_DIM * HIDDEN, 8);
    uint16_t *wk = alloc_bf16((size_t)KV_DIM * HIDDEN, 9);
    uint16_t *wv = alloc_bf16((size_t)KV_DIM * HIDDEN, 10);
    float *q = alloc_f32((size_t)SEQ * Q_DIM, 11);
    float *k = alloc_f32((size_t)SEQ * KV_DIM, 12);
    float *v = alloc_f32((size_t)SEQ * KV_DIM, 13);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_linear_nobias_bf16_qkv(q, k, v, x, wq, wk, wv,
                                    SEQ, HIDDEN, Q_DIM, KV_DIM);
    bench_sink += q[0] + k[0] + v[0];
    free(v); free(k); free(q); free(wv); free(wk); free(wq); free(x);
}

static void bm_bf16_widen(bench_state_t *b)
{
    size_t n = (size_t)INTER * HIDDEN;
    uint16_t *src = alloc_bf16(n, 14);
    float *dst = alloc_f32(n, 15);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_bf16_to_f32_buf(dst, src, n);
    bench_sink += dst[0];
    free(dst); free(src);
}

/* ---- attention pieces ---- */

static void bm_attn_scores(bench_state_t *b)
{
    /* one head: scores[seq,seq] = Q[seq,hd] @ K[seq,hd]^T */
    float *qm = alloc_f32((size_t)ATTN_SEQ * HEAD_DIM, 16);
    float *km = alloc_f32((size_t)ATTN_SEQ * HEAD_DIM, 17);
    float *s = alloc_f32((size_t)ATTN_SEQ * ATTN_SEQ, 18);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_matmul_t(s, qm, km, ATTN_SEQ, HEAD_DIM, ATTN_SEQ);
    bench_sink += s[0];
    free(s); free(km); free(qm);
}

static void bm_softmax(bench_state_t *b)
{
    float *x = alloc_f32((size_t)ATTN_SEQ * ATTN_SEQ, 19);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_softmax(x, ATTN_SEQ, ATTN_SEQ);
    bench_sink += x[0];
    free(x);
}

static void bm_rope(bench_state_t *b)
{
    float *x = alloc_f32((size_t)ATTN_SEQ * Q_DIM, 20);
    float *cosv = alloc_f32((size_t)ATTN_SEQ * HEAD_DIM, 21);
    float *sinv = alloc_f32((size_t)ATTN_SEQ * HEAD_DIM, 22);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_apply_rope_neox(x, cosv, sinv, ATTN_SEQ, Q_DIM / HEAD_DIM,
                             HEAD_DIM);
    bench_sink += x[0];
    free(sinv); free(cosv); free(x);
}

/* ---- norms, activations, reductions ---- */

static void bm_rms_norm(bench_state_t *b)
{
    float *x = alloc_f32((size_t)SEQ * HIDDEN, 23);
    float *w = alloc_f32(HIDDEN, 24);
    float *y = alloc_f32((size_t)SEQ * HIDDEN, 25);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_rms_norm(y, x, w, SEQ, HIDDEN, 1e-6f);
    bench_sink += y[0];
    free(y); free(w); free(x);
}

static void bm_swiglu(bench_state_t *b)
{
    float *gate_up = alloc_f32((size_t)SEQ * 2 * INTER, 26);
    float *out = alloc_f32((size_t)SEQ * INTER, 27);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        qwen_swiglu_multiply(out, gate_up, SEQ, INTER);
    bench_sink += out[0];
    free(out); free(gate_up);
}

static void bm_dot(bench_state_t *b)
{
    float *x = alloc_f32(HIDDEN, 28);
    float *y = alloc_f32(HIDDEN, 29);
    bench_begin(b);
    for (long i = 0; i < b->n; i++)
        bench_sink += qwen_dot_f32(x, y, HIDDEN);
    free(y); free(x);
}

static const bench_case_t CASES[] = {
    {"linear_f32/s1_1024x2048_qproj", bm_linear_f32_s1_qproj},
    {"linear_f32/s1_1024x3072_gate", bm_linear_f32_s1_gate},
    {"linear_f32/s1_3072x1024_down", bm_linear_f32_s1_down},
    {"linear_f32/s8_1024x3072_gate", bm_linear_f32_s8_gate},
    {"linear_bf16/s1_1024x3072_gate", bm_linear_bf16_s1_gate},
    {"linear_bf16/s8_1024x3072_gate", bm_linear_bf16_s8_gate},
    {"linear_bf16/s8_qkv", bm_qkv_bf16},
    {"bf16_to_f32/3145728", bm_bf16_widen},
    {"attn/scores_one_head_seq128", bm_attn_scores},
    {"attn/softmax_128x128", bm_softmax},
    {"attn/rope_seq128_16h", bm_rope},
    {"norm/rms_s8_1024", bm_rms_norm},
    {"act/swiglu_s8_3072", bm_swiglu},
    {"reduce/dot_1024", bm_dot},
};

int main(int argc, char **argv)
{
    int threads = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = atoi(argv[i + 1]);
    }
    if (threads < 1) threads = 1;
    qwen_set_threads(threads);

    char meta[128];
#ifdef USE_BLAS
    snprintf(meta, sizeof(meta), "suite=kernels threads=%d blas=1", threads);
#else
    snprintf(meta, sizeof(meta), "suite=kernels threads=%d blas=0", threads);
#endif

    bench_opts_t opts = {0};
    opts.meta = meta;
    bench_parse_args(&opts, argc, argv);
    return bench_main(&opts, CASES, (int)(sizeof(CASES) / sizeof(CASES[0])));
}
