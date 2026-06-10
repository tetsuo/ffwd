/* tests/test_kernels.c - golden tests for the low-level math kernels.
 * Runs via `make test` (generic and BLAS variants) and
 * scripts/check_kernel_golden.py. No model files required. */

#include "qwen_kernels.h"
#include "qwen_kernels_impl.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
int qwen_verbose = 0;

static void expect_close(const char *name, float got, float want, float tol) {
    float diff = fabsf(got - want);
    float scale = fmaxf(1.0f, fmaxf(fabsf(got), fabsf(want)));
    if (diff > tol * scale) {
        fprintf(stderr, "%s: got %.9g want %.9g diff %.9g\n",
                name, got, want, diff);
        failures++;
    }
}

static uint16_t f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (uint16_t)(u >> 16);
}

static float bf16_to_f32(uint16_t x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void test_rms_norm(void) {
    const int seq = 2, hidden = 4;
    const float x[8] = {
        1.0f, 2.0f, -3.0f, 4.0f,
        0.5f, -1.0f, 2.0f, -0.5f
    };
    const float w[4] = {1.0f, 0.5f, -1.0f, 2.0f};
    float out[8];

    qwen_rms_norm(out, x, w, seq, hidden, 1e-6f);

    for (int s = 0; s < seq; s++) {
        float ss = 0.0f;
        for (int d = 0; d < hidden; d++) {
            float v = x[s * hidden + d];
            ss += v * v;
        }
        float inv = 1.0f / sqrtf(ss / hidden + 1e-6f);
        for (int d = 0; d < hidden; d++) {
            float want = x[s * hidden + d] * inv * w[d];
            expect_close("rms_norm", out[s * hidden + d], want, 1e-6f);
        }
    }
}

static void test_rope_neox(void) {
    const int seq = 2, heads = 1, head_dim = 4;
    const int positions[2] = {0, 3};
    float cosv[8], sinv[8];
    float x[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.5f, 2.0f, -0.25f
    };
    float orig[8];
    memcpy(orig, x, sizeof(x));

    qwen_compute_rope_neox(cosv, sinv, positions, seq, head_dim, 10000.0f);
    qwen_apply_rope_neox(x, cosv, sinv, seq, heads, head_dim);

    for (int s = 0; s < seq; s++) {
        for (int d = 0; d < head_dim / 2; d++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * d) / head_dim);
            float angle = (float)positions[s] * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            float x1 = orig[s * head_dim + d];
            float x2 = orig[s * head_dim + head_dim / 2 + d];
            float want1 = x1 * c - x2 * sn;
            float want2 = x2 * c + x1 * sn;
            expect_close("rope_1", x[s * head_dim + d], want1, 1e-6f);
            expect_close("rope_2", x[s * head_dim + head_dim / 2 + d], want2, 1e-6f);
        }
    }
}

static void reference_packed_gqa(float *out, const float *Q, const float *K,
                                 const float *V, const int *offsets, int batch,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 float scale) {
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        for (int b = 0; b < batch; b++) {
            int start = offsets[b], end = offsets[b + 1];
            for (int i = start; i < end; i++) {
                const float *q = Q + i * q_hidden + h * head_dim;
                float *o = out + i * q_hidden + h * head_dim;
                float max_score = -INFINITY;
                for (int j = start; j < end; j++) {
                    const float *k = K + j * kv_hidden + kv_h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) score += q[d] * k[d];
                    score *= scale;
                    if (score > max_score) max_score = score;
                }
                float denom = 0.0f;
                for (int d = 0; d < head_dim; d++) o[d] = 0.0f;
                for (int j = start; j < end; j++) {
                    const float *k = K + j * kv_hidden + kv_h * head_dim;
                    const float *v = V + j * kv_hidden + kv_h * head_dim;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) score += q[d] * k[d];
                    float wt = expf(score * scale - max_score);
                    denom += wt;
                    for (int d = 0; d < head_dim; d++) o[d] += wt * v[d];
                }
                for (int d = 0; d < head_dim; d++) o[d] /= denom;
            }
        }
    }
}

static void test_packed_gqa_attention(void) {
    enum { total = 5, n_heads = 2, n_kv_heads = 1, head_dim = 2 };
    enum { q_hidden = n_heads * head_dim };
    enum { kv_hidden = n_kv_heads * head_dim };
    const int offsets[3] = {0, 2, 5};
    float Q[total * q_hidden];
    float K[total * kv_hidden];
    float V[total * kv_hidden];
    float got[total * q_hidden];
    float want[total * q_hidden];

    for (int i = 0; i < total * q_hidden; i++)
        Q[i] = sinf((float)i * 0.17f) - 0.25f;
    for (int i = 0; i < total * kv_hidden; i++) {
        K[i] = cosf((float)i * 0.11f) + 0.1f;
        V[i] = sinf((float)i * 0.07f) - cosf((float)i * 0.13f);
    }

    qwen_bidirectional_gqa_attention_packed(got, Q, K, V, offsets, 2,
                                            n_heads, n_kv_heads, head_dim,
                                            1.0f / sqrtf((float)head_dim));
    reference_packed_gqa(want, Q, K, V, offsets, 2,
                         n_heads, n_kv_heads, head_dim,
                         1.0f / sqrtf((float)head_dim));

    for (int i = 0; i < total * q_hidden; i++)
        expect_close("packed_gqa", got[i], want[i], 2e-6f);
}

static void test_packed_gqa_attention_long(void) {
    enum { total = 136, n_heads = 4, n_kv_heads = 2, head_dim = 8 };
    enum { q_hidden = n_heads * head_dim };
    enum { kv_hidden = n_kv_heads * head_dim };
    const int offsets[3] = {0, 129, 136};
    float *Q = malloc((size_t)total * q_hidden * sizeof(*Q));
    float *K = malloc((size_t)total * kv_hidden * sizeof(*K));
    float *V = malloc((size_t)total * kv_hidden * sizeof(*V));
    float *got = malloc((size_t)total * q_hidden * sizeof(*got));
    float *want = malloc((size_t)total * q_hidden * sizeof(*want));
    if (!Q || !K || !V || !got || !want) exit(2);

    for (int i = 0; i < total * q_hidden; i++)
        Q[i] = sinf((float)i * 0.017f) - 0.25f;
    for (int i = 0; i < total * kv_hidden; i++) {
        K[i] = cosf((float)i * 0.011f) + 0.1f;
        V[i] = sinf((float)i * 0.007f) - cosf((float)i * 0.013f);
    }

    qwen_set_threads(4);
    qwen_bidirectional_gqa_attention_packed(got, Q, K, V, offsets, 2,
                                            n_heads, n_kv_heads, head_dim,
                                            1.0f / sqrtf((float)head_dim));
    reference_packed_gqa(want, Q, K, V, offsets, 2,
                         n_heads, n_kv_heads, head_dim,
                         1.0f / sqrtf((float)head_dim));
    qwen_set_threads(1);

    for (int i = 0; i < total * q_hidden; i++)
        expect_close("packed_gqa_long", got[i], want[i], 2e-5f);
    free(want);
    free(got);
    free(V);
    free(K);
    free(Q);
}

static void test_bf16_linear(void) {
    enum { seq = 2, in_dim = 3, out_dim = 2 };
    const float x[6] = {
        1.0f, -2.0f, 0.5f,
        -1.0f, 0.25f, 2.0f
    };
    const float wf32[6] = {
        0.5f, -1.0f, 2.0f,
        -0.25f, 0.75f, 1.5f
    };
    uint16_t wbf16[6];
    float got[seq * out_dim];

    for (int i = 0; i < 6; i++) wbf16[i] = f32_to_bf16(wf32[i]);
    qwen_linear_nobias_bf16(got, x, wbf16, seq, in_dim, out_dim);

    for (int s = 0; s < seq; s++) {
        for (int o = 0; o < out_dim; o++) {
            float want = 0.0f;
            for (int i = 0; i < in_dim; i++)
                want += x[s * in_dim + i] * bf16_to_f32(wbf16[o * in_dim + i]);
            expect_close("bf16_linear", got[s * out_dim + o], want, 1e-6f);
        }
    }
}

static void check_bf16_qkv(int n_threads) {
    enum { seq = 3, in_dim = 4, q_dim = 5, kv_dim = 2 };
    float x[seq * in_dim];
    float wq_f32[q_dim * in_dim];
    float wk_f32[kv_dim * in_dim];
    float wv_f32[kv_dim * in_dim];
    uint16_t wq[q_dim * in_dim];
    uint16_t wk[kv_dim * in_dim];
    uint16_t wv[kv_dim * in_dim];
    float got_q[seq * q_dim], got_k[seq * kv_dim], got_v[seq * kv_dim];

    for (int i = 0; i < seq * in_dim; i++)
        x[i] = sinf((float)i * 0.23f) - 0.4f;
    for (int i = 0; i < q_dim * in_dim; i++) {
        wq_f32[i] = cosf((float)i * 0.17f) + 0.1f;
        wq[i] = f32_to_bf16(wq_f32[i]);
    }
    for (int i = 0; i < kv_dim * in_dim; i++) {
        wk_f32[i] = sinf((float)i * 0.31f) - 0.2f;
        wv_f32[i] = cosf((float)i * 0.29f) + 0.3f;
        wk[i] = f32_to_bf16(wk_f32[i]);
        wv[i] = f32_to_bf16(wv_f32[i]);
    }

    qwen_set_threads(n_threads);
    qwen_linear_nobias_bf16_qkv(got_q, got_k, got_v, x, wq, wk, wv,
                                seq, in_dim, q_dim, kv_dim);

    for (int s = 0; s < seq; s++) {
        for (int o = 0; o < q_dim; o++) {
            float want = 0.0f;
            for (int i = 0; i < in_dim; i++)
                want += x[s * in_dim + i] * bf16_to_f32(wq[o * in_dim + i]);
            expect_close("bf16_qkv_q", got_q[s * q_dim + o], want, 1e-6f);
        }
        for (int o = 0; o < kv_dim; o++) {
            float want_k = 0.0f, want_v = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                want_k += x[s * in_dim + i] * bf16_to_f32(wk[o * in_dim + i]);
                want_v += x[s * in_dim + i] * bf16_to_f32(wv[o * in_dim + i]);
            }
            expect_close("bf16_qkv_k", got_k[s * kv_dim + o], want_k, 1e-6f);
            expect_close("bf16_qkv_v", got_v[s * kv_dim + o], want_v, 1e-6f);
        }
    }
}

static void test_bf16_qkv(void) {
    check_bf16_qkv(1);
    check_bf16_qkv(3);
    qwen_set_threads(1);
}

/* seq==1 routes through bf16_matvec_threaded; a pool splits the output
 * rows across matvec_worker calls. Both must match the scalar reference. */
static void check_bf16_matvec(int n_threads) {
    enum { in_dim = 19, out_dim = 23 };
    float x[in_dim];
    uint16_t w[out_dim * in_dim];
    float got[out_dim];

    for (int i = 0; i < in_dim; i++)
        x[i] = sinf((float)i * 0.37f) - 0.2f;
    for (int i = 0; i < out_dim * in_dim; i++)
        w[i] = f32_to_bf16(cosf((float)i * 0.21f) + 0.05f);

    qwen_set_threads(n_threads);
    qwen_linear_nobias_bf16(got, x, w, 1, in_dim, out_dim);

    for (int o = 0; o < out_dim; o++) {
        float want = 0.0f;
        for (int i = 0; i < in_dim; i++)
            want += x[i] * bf16_to_f32(w[o * in_dim + i]);
        expect_close("bf16_matvec", got[o], want, 1e-6f);
    }
}

static void test_bf16_matvec(void) {
    check_bf16_matvec(1);
    check_bf16_matvec(4);
    qwen_set_threads(1);
}

/* Above QWEN_RMS_NORM_PARALLEL_ELEMS (256k elements) rms_norm fans out to
 * rms_norm_worker; row-wise math is identical, so results must match the
 * single-thread run exactly. */
static void test_rms_norm_threaded(void) {
    enum { seq = 96, hidden = 3072 };   /* 294912 elements */
    size_t elems = (size_t)seq * hidden;
    float *x = (float *)malloc(elems * sizeof(float));
    float *w = (float *)malloc((size_t)hidden * sizeof(float));
    float *one = (float *)malloc(elems * sizeof(float));
    float *many = (float *)malloc(elems * sizeof(float));
    if (!x || !w || !one || !many) {
        fprintf(stderr, "rms_norm_threaded: allocation failure\n");
        failures++;
        free(many); free(one); free(w); free(x);
        return;
    }
    unsigned s = 7u;
    for (size_t i = 0; i < elems; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
    for (int i = 0; i < hidden; i++)
        w[i] = 0.8f + 0.01f * (float)(i % 7);

    qwen_set_threads(1);
    qwen_rms_norm(one, x, w, seq, hidden, 1e-6f);
    qwen_set_threads(4);
    qwen_rms_norm(many, x, w, seq, hidden, 1e-6f);
    qwen_set_threads(1);

    for (size_t i = 0; i < elems; i++) {
        if (one[i] != many[i]) {
            fprintf(stderr, "rms_norm_threaded: mismatch at %zu: %.9g %.9g\n",
                    i, one[i], many[i]);
            failures++;
            break;
        }
    }
    free(many); free(one); free(w); free(x);
}

/* bf16->f32 widening is an exact bit shift; check the SIMD body and the
 * remainder lanes against the scalar definition. */
static void test_bf16_widen_buf(void) {
    enum { NMAXW = 67 };
    static const int sizes[] = {1, 7, 8, 15, 64, NMAXW};
    uint16_t src[NMAXW];
    float got[NMAXW];

    for (int i = 0; i < NMAXW; i++)
        src[i] = f32_to_bf16(sinf((float)i * 0.13f) * 3.0f);

    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); t++) {
        int n = sizes[t];
        memset(got, 0, sizeof(got));
        qwen_bf16_to_f32_buf(got, src, (size_t)n);
        for (int i = 0; i < n; i++) {
            float want = bf16_to_f32(src[i]);
            if (memcmp(&got[i], &want, sizeof(float)) != 0) {
                fprintf(stderr, "bf16_widen n=%d at %d: %.9g want %.9g\n",
                        n, i, got[i], want);
                failures++;
            }
        }
    }
}

static void check_bf16_pair(int n_threads) {
    enum { seq = 4, in_dim = 5, a_dim = 3, b_dim = 7 };
    float x[seq * in_dim];
    float wa_f32[a_dim * in_dim];
    float wb_f32[b_dim * in_dim];
    uint16_t wa[a_dim * in_dim];
    uint16_t wb[b_dim * in_dim];
    float got_a[seq * a_dim], got_b[seq * b_dim];

    for (int i = 0; i < seq * in_dim; i++)
        x[i] = cosf((float)i * 0.19f) - 0.3f;
    for (int i = 0; i < a_dim * in_dim; i++) {
        wa_f32[i] = sinf((float)i * 0.13f) + 0.2f;
        wa[i] = f32_to_bf16(wa_f32[i]);
    }
    for (int i = 0; i < b_dim * in_dim; i++) {
        wb_f32[i] = cosf((float)i * 0.21f) - 0.1f;
        wb[i] = f32_to_bf16(wb_f32[i]);
    }

    qwen_set_threads(n_threads);
    qwen_linear_nobias_bf16_pair(got_a, got_b, x, wa, wb,
                                 seq, in_dim, a_dim, b_dim);

    for (int s = 0; s < seq; s++) {
        for (int o = 0; o < a_dim; o++) {
            float want = 0.0f;
            for (int i = 0; i < in_dim; i++)
                want += x[s * in_dim + i] * bf16_to_f32(wa[o * in_dim + i]);
            expect_close("bf16_pair_a", got_a[s * a_dim + o], want, 1e-6f);
        }
        for (int o = 0; o < b_dim; o++) {
            float want = 0.0f;
            for (int i = 0; i < in_dim; i++)
                want += x[s * in_dim + i] * bf16_to_f32(wb[o * in_dim + i]);
            expect_close("bf16_pair_b", got_b[s * b_dim + o], want, 1e-6f);
        }
    }
}

static void test_bf16_pair(void) {
    check_bf16_pair(1);
    check_bf16_pair(4);
    qwen_set_threads(1);
}

/* The dispatch macros in qwen_kernels_impl.h pick the SIMD variant at
 * compile time, so on arm64/x86 the generic C kernels are linked but never
 * called. Run them directly against the selected implementation on
 * remainder-lane-hostile sizes; on a plain build impl == generic and the
 * comparison is trivially exact. */
static void test_generic_vs_impl(void) {
    enum { IN = 67, OUT = 9, NMAX = 1027 };
    static float x[IN], bias[OUT], a[NMAX], b[NMAX], d0[NMAX], d1[NMAX];
    static uint16_t W[(size_t)OUT * IN];
    uint32_t s = 0x12345678u;
    for (int i = 0; i < NMAX; i++) {
        s = s * 1664525u + 1013904223u;
        a[i] = (float)(int32_t)s * 1e-9f;
        s = s * 1664525u + 1013904223u;
        b[i] = (float)(int32_t)s * 1e-9f;
    }
    for (int i = 0; i < IN; i++) x[i] = a[i];
    for (int o = 0; o < OUT; o++) bias[o] = b[o];
    for (size_t i = 0; i < sizeof(W) / sizeof(W[0]); i++) {
        s = s * 1664525u + 1013904223u;
        W[i] = f32_to_bf16((float)(int32_t)s * 1e-9f);
    }

    float yg[OUT], yi[OUT];
    qwen_bf16_matvec_fused_generic(yg, x, W, bias, IN, OUT);
    qwen_bf16_matvec_fused_impl(yi, x, W, bias, IN, OUT);
    for (int o = 0; o < OUT; o++)
        expect_close("matvec_fused generic-vs-impl", yg[o], yi[o], 1e-5f);
    qwen_bf16_matvec_fused_generic(yg, x, W, NULL, IN, OUT);
    qwen_bf16_matvec_fused_impl(yi, x, W, NULL, IN, OUT);
    for (int o = 0; o < OUT; o++)
        expect_close("matvec_fused nobias generic-vs-impl", yg[o], yi[o], 1e-5f);

    int bg = -1, bi = -1;
    float vg = 0.0f, vi = 0.0f;
    qwen_argmax_bf16_range_generic(x, W, IN, 1, OUT, &bg, &vg);
    qwen_argmax_bf16_range_impl(x, W, IN, 1, OUT, &bi, &vi);
    if (bg != bi) {
        fprintf(stderr, "argmax generic-vs-impl: got %d want %d\n", bi, bg);
        failures++;
    }
    expect_close("argmax val generic-vs-impl", vg, vi, 1e-5f);

    const int sizes[] = {1, 7, 64, NMAX};
    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); t++) {
        int n = sizes[t];
        expect_close("dot generic-vs-impl",
                     qwen_dot_f32_generic(a, b, n),
                     qwen_dot_f32_impl(a, b, n), 1e-5f);

        memcpy(d0, a, (size_t)n * sizeof(float));
        memcpy(d1, a, (size_t)n * sizeof(float));
        qwen_vec_scale_inplace_generic(d0, 1.37f, n);
        qwen_vec_scale_inplace_impl(d1, 1.37f, n);
        for (int i = 0; i < n; i++)
            expect_close("scale generic-vs-impl", d0[i], d1[i], 1e-6f);

        memcpy(d0, a, (size_t)n * sizeof(float));
        memcpy(d1, a, (size_t)n * sizeof(float));
        qwen_vec_axpy_inplace_generic(d0, b, -0.61f, n);
        qwen_vec_axpy_inplace_impl(d1, b, -0.61f, n);
        for (int i = 0; i < n; i++)
            expect_close("axpy generic-vs-impl", d0[i], d1[i], 1e-6f);

        memcpy(d0, a, (size_t)n * sizeof(float));
        memcpy(d1, a, (size_t)n * sizeof(float));
        qwen_vec_scale_add_generic(d0, b, 0.83f, n);
        qwen_vec_scale_add_impl(d1, b, 0.83f, n);
        for (int i = 0; i < n; i++)
            expect_close("scale_add generic-vs-impl", d0[i], d1[i], 1e-6f);
    }
}

int main(void) {
    qwen_set_threads(1);
    test_rms_norm();
    test_rope_neox();
    test_packed_gqa_attention();
    test_packed_gqa_attention_long();
    test_bf16_linear();
    test_bf16_qkv();
    test_bf16_matvec();
    test_rms_norm_threaded();
    test_bf16_widen_buf();
    test_bf16_pair();
    test_generic_vs_impl();
    if (failures != 0) return 1;
    puts("ok: kernel golden tests passed");
    return 0;
}
