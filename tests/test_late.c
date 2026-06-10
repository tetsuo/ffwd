/* tests/test_late.c - late-interaction (ColBERT-style MaxSim) API tests.
 * Hermetic: synthesizes a tiny base model, a 1_Dense projection head, and a
 * byte-complete tokenizer fixture. Runs via `make test`. */

#include "embed.h"
#include "qwen_tokenizer.h"
#include "tiny_model.h"
#include "tok_fixture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { TOKEN_DIM = 6 };

static float max_abs_diff(const float *a, const float *b, size_t n)
{
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float row_norm(const float *v, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

/* Parity between the grouped batch scorer and per-document MaxSim. */
static int check_maxsim_batch_parity(const float *query, int query_tokens,
                                     const float *docs, const int *offsets,
                                     int n_docs, int dim, const char *what)
{
    float scores[8];
    if (n_docs > (int)(sizeof(scores) / sizeof(scores[0]))) return 1;
    if (pplx_late_maxsim_batch(query, query_tokens, docs, offsets, n_docs,
                               dim, scores) != 0) {
        fprintf(stderr, "maxsim_batch (%s) failed\n", what);
        return 1;
    }
    for (int i = 0; i < n_docs; i++) {
        float one = pplx_late_maxsim(query, query_tokens,
                                     docs + (size_t)offsets[i] * dim,
                                     offsets[i + 1] - offsets[i], dim);
        if (fabsf(scores[i] - one) > 0.0001f) {
            fprintf(stderr, "maxsim_batch (%s) doc %d: %g vs %g\n",
                    what, i, scores[i], one);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    int rc = 1;
    tm_dims_t dims = {4, 2, 1, 2, 8, TF_VOCAB_SIZE};
    int hidden = dims.hidden;

    qwen_tokenizer_t *tok = NULL;
    pplx_model_t *base = NULL;
    pplx_workspace_t *base_ws = NULL;
    pplx_late_model_t *late = NULL;
    pplx_late_workspace_t *ws = NULL;
    int *ids = NULL;
    float *states = NULL, *raw = NULL, *vecs = NULL, *expected = NULL;
    float *doc_vecs = NULL;
    float *big_q = NULL, *big_d = NULL;
    int *doc_ids[3] = {0};

    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/pplx-late-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(dir) || tf_write_vocab(dir) != 0 ||
        tm_write_model_dims(dir, "F32", &dims) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    /* The base model must load before the projection exists; keep it for
     * the reference forward pass below. */
    base = pplx_model_load(dir);
    if (!base || !(base_ws = pplx_workspace_new(base))) {
        fprintf(stderr, "failed to load base model\n");
        goto fail;
    }

    /* No 1_Dense head yet: not a late model. */
    if (pplx_late_model_load(dir) != NULL) {
        fprintf(stderr, "late load succeeded without 1_Dense\n");
        goto fail;
    }
    if (pplx_late_model_config(NULL) != NULL ||
        pplx_late_model_token_dim(NULL) != 0 ||
        pplx_late_workspace_new(NULL) != NULL) {
        fprintf(stderr, "NULL-argument handling failed\n");
        goto fail;
    }
    pplx_late_workspace_free(NULL);
    pplx_late_model_free(NULL);

    /* A head whose tensor is not named linear.weight must be rejected. */
    if (tm_write_late_projection_named(dir, "F32", TOKEN_DIM, hidden,
                                       "dense.weight") != 0) {
        fprintf(stderr, "failed to write malformed projection\n");
        goto fail;
    }
    if (pplx_late_model_load(dir) != NULL) {
        fprintf(stderr, "late load accepted a head without linear.weight\n");
        goto fail;
    }

    if (tm_write_late_projection(dir, "F32", TOKEN_DIM, hidden) != 0) {
        fprintf(stderr, "failed to write projection\n");
        goto fail;
    }
    late = pplx_late_model_load(dir);
    if (!late) {
        fprintf(stderr, "late model load failed\n");
        goto fail;
    }

    /* With the projection present the dir is no longer a pooled model. */
    if (pplx_model_load(dir) != NULL) {
        fprintf(stderr, "pooled load accepted a late-interaction snapshot\n");
        goto fail;
    }

    const pplx_config_t *cfg = pplx_late_model_config(late);
    if (!cfg || cfg->hidden_size != hidden ||
        pplx_late_model_token_dim(late) != TOKEN_DIM) {
        fprintf(stderr, "late config/token_dim mismatch\n");
        goto fail;
    }

    ws = pplx_late_workspace_new(late);
    if (!ws) {
        fprintf(stderr, "late workspace allocation failed\n");
        goto fail;
    }

    char vocab_path[1280];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", dir);
    tok = qwen_tokenizer_load(vocab_path);
    if (!tok) {
        fprintf(stderr, "tokenizer load failed\n");
        goto fail;
    }

    int n_tokens = 0;
    ids = qwen_tokenizer_encode(tok, "hello world", &n_tokens);
    if (!ids || n_tokens <= 1) {
        fprintf(stderr, "tokenization failed\n");
        goto fail;
    }

    size_t out_count = (size_t)n_tokens * TOKEN_DIM;
    raw = (float *)calloc(out_count, sizeof(float));
    vecs = (float *)calloc(out_count, sizeof(float));
    expected = (float *)calloc(out_count, sizeof(float));
    if (!raw || !vecs || !expected) {
        fprintf(stderr, "allocation failure\n");
        goto fail;
    }

    if (pplx_late_model_encode_tokens(late, ws, NULL, n_tokens, 1, vecs) == 0 ||
        pplx_late_model_encode_tokens(late, ws, ids, 0, 1, vecs) == 0 ||
        pplx_late_model_encode_tokens(NULL, ws, ids, n_tokens, 1, vecs) == 0 ||
        pplx_late_model_encode_tokens(late, NULL, ids, n_tokens, 1, vecs) == 0) {
        fprintf(stderr, "encode accepted invalid arguments\n");
        goto fail;
    }

    if (pplx_late_model_encode_tokens(late, ws, ids, n_tokens, 0, raw) != 0 ||
        pplx_late_model_encode_tokens(late, ws, ids, n_tokens, 1, vecs) != 0) {
        fprintf(stderr, "encode_tokens failed\n");
        goto fail;
    }

    /* Reference: run the same tokens through the standalone base model and
     * apply the projection by hand from the fixture's deterministic values. */
    states = pplx_model_forward(base, base_ws, ids, n_tokens);
    if (!states) {
        fprintf(stderr, "pplx_model_forward failed\n");
        goto fail;
    }
    for (int t = 0; t < n_tokens; t++) {
        for (int r = 0; r < TOKEN_DIM; r++) {
            float acc = 0.0f;
            for (int k = 0; k < hidden; k++)
                acc += states[(size_t)t * hidden + k] *
                       tm_value("linear.weight", (size_t)r * hidden + k);
            expected[(size_t)t * TOKEN_DIM + r] = acc;
        }
    }
    if (max_abs_diff(raw, expected, out_count) > 0.0001f) {
        fprintf(stderr, "projection mismatch vs reference: %g\n",
                max_abs_diff(raw, expected, out_count));
        goto fail;
    }

    /* normalize=1 returns the raw rows scaled to unit length. */
    for (int t = 0; t < n_tokens; t++) {
        const float *nv = vecs + (size_t)t * TOKEN_DIM;
        const float *rv = raw + (size_t)t * TOKEN_DIM;
        if (fabsf(row_norm(nv, TOKEN_DIM) - 1.0f) > 0.0001f) {
            fprintf(stderr, "row %d is not unit norm\n", t);
            goto fail;
        }
        float scale = row_norm(rv, TOKEN_DIM);
        for (int r = 0; r < TOKEN_DIM; r++) {
            if (fabsf(nv[r] * scale - rv[r]) > 0.0001f) {
                fprintf(stderr, "row %d direction changed by normalize\n", t);
                goto fail;
            }
        }
    }

    /* Hand-computed MaxSim: best dot per query token, summed.
     * q0 matches d0 exactly (1.0); q1's best is d0 (0.0). */
    {
        static const float q2[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        static const float d2[4] = {1.0f, 0.0f, 0.0f, -1.0f};
        float s = pplx_late_maxsim(q2, 2, d2, 2, 2);
        if (fabsf(s - 1.0f) > 0.000001f) {
            fprintf(stderr, "maxsim hand check: got %g, want 1\n", s);
            goto fail;
        }
        if (pplx_late_maxsim(NULL, 2, d2, 2, 2) != 0.0f ||
            pplx_late_maxsim(q2, 0, d2, 2, 2) != 0.0f) {
            fprintf(stderr, "maxsim accepted invalid arguments\n");
            goto fail;
        }
    }

    /* Batch scoring parity on real encoded vectors (three small docs share
     * one similarity group). */
    {
        const char *doc_texts[3] = {"hello world", "held", "wor world hello"};
        int lens[3] = {0};
        int offsets[4] = {0};
        for (int i = 0; i < 3; i++) {
            doc_ids[i] = qwen_tokenizer_encode(tok, doc_texts[i], &lens[i]);
            if (!doc_ids[i] || lens[i] <= 0) {
                fprintf(stderr, "doc tokenization failed\n");
                goto fail;
            }
            offsets[i + 1] = offsets[i] + lens[i];
        }
        doc_vecs = (float *)calloc((size_t)offsets[3] * TOKEN_DIM,
                                   sizeof(float));
        if (!doc_vecs) {
            fprintf(stderr, "allocation failure\n");
            goto fail;
        }
        for (int i = 0; i < 3; i++) {
            if (pplx_late_model_encode_tokens(
                    late, ws, doc_ids[i], lens[i], 1,
                    doc_vecs + (size_t)offsets[i] * TOKEN_DIM) != 0) {
                fprintf(stderr, "doc encode failed\n");
                goto fail;
            }
        }
        if (check_maxsim_batch_parity(vecs, n_tokens, doc_vecs, offsets, 3,
                                      TOKEN_DIM, "encoded docs") != 0)
            goto fail;

        float scores[3];
        int bad_first[4] = {1, 2, 3, 4};
        int empty_doc[3] = {0, 2, 2};
        if (pplx_late_maxsim_batch(vecs, n_tokens, doc_vecs, bad_first, 3,
                                   TOKEN_DIM, scores) == 0 ||
            pplx_late_maxsim_batch(vecs, n_tokens, doc_vecs, empty_doc, 2,
                                   TOKEN_DIM, scores) == 0 ||
            pplx_late_maxsim_batch(vecs, n_tokens, doc_vecs, offsets, 3,
                                   TOKEN_DIM, NULL) == 0) {
            fprintf(stderr, "maxsim_batch accepted invalid arguments\n");
            goto fail;
        }
    }

    /* One document larger than the similarity-GEMM token budget must score
     * identically to the scalar path (it runs alone in its own group). */
    {
        enum { BIG_DIM = 4, BIG_Q = 3, BIG_DOC0 = 4100, BIG_DOC1 = 10 };
        int offsets[3] = {0, BIG_DOC0, BIG_DOC0 + BIG_DOC1};
        big_q = (float *)malloc((size_t)BIG_Q * BIG_DIM * sizeof(float));
        big_d = (float *)malloc((size_t)offsets[2] * BIG_DIM * sizeof(float));
        if (!big_q || !big_d) {
            fprintf(stderr, "allocation failure\n");
            goto fail;
        }
        unsigned s = 12345u;
        for (size_t i = 0; i < (size_t)BIG_Q * BIG_DIM; i++) {
            s = s * 1664525u + 1013904223u;
            big_q[i] = (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        for (size_t i = 0; i < (size_t)offsets[2] * BIG_DIM; i++) {
            s = s * 1664525u + 1013904223u;
            big_d[i] = (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        if (check_maxsim_batch_parity(big_q, BIG_Q, big_d, offsets, 2,
                                      BIG_DIM, "oversized doc") != 0)
            goto fail;
    }

    printf("ok: late API token_dim=%d tokens=%d hidden=%d\n",
           TOKEN_DIM, n_tokens, hidden);
    rc = 0;

fail:
    free(big_d);
    free(big_q);
    free(doc_vecs);
    for (int i = 0; i < 3; i++) free(doc_ids[i]);
    free(expected);
    free(vecs);
    free(raw);
    free(states);
    free(ids);
    qwen_tokenizer_free(tok);
    pplx_late_workspace_free(ws);
    pplx_late_model_free(late);
    pplx_workspace_free(base_ws);
    pplx_model_free(base);
    return rc;
}
