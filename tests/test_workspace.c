/* tests/test_workspace.c - model/workspace API lifecycle tests.
 * Hermetic by default (synthesizes a tiny model + tokenizer fixture);
 * pass a MODEL_DIR to run the same checks against real weights, which is
 * what scripts/check_workspace_api.py does. Runs via `make test`. */

#include "embed.h"
#include "tokenizer_bpe.h"
#include "tiny_model.h"
#include "tok_fixture.h"

#include "../src/alloc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

static int check_alloc_helpers(void) {
    int cap = 0;
    if (grow_cap(-1, 1, &cap) == 0) {
        fprintf(stderr, "grow_cap accepted a negative current capacity\n");
        return -1;
    }
    if (grow_cap(33, 17, &cap) != 0 || cap != 33) {
        fprintf(stderr, "grow_cap failed to preserve a larger current capacity\n");
        return -1;
    }
    if (grow_cap(0, 17, &cap) != 0 || cap != 32) {
        fprintf(stderr, "grow_cap failed to round new capacity to granularity\n");
        return -1;
    }
    return 0;
}

static int check_vector_helpers(void) {
    float a[1] = {1.0e19f};
    float b[1] = {1.0e19f};
    float cos = embed_cosine_similarity(a, b, 1);
    if (fabsf(cos - 1.0f) > 1e-5f) {
        fprintf(stderr, "large finite cosine failed: got %.9g\n", cos);
        return -1;
    }

    float z[2] = {0.0f, 0.0f};
    if (embed_l2_normalize(z, 2) == 0 || embed_cosine_similarity(z, a, 1) != 0.0f) {
        fprintf(stderr, "zero-vector helpers accepted invalid input\n");
        return -1;
    }
    return 0;
}

/* The allocating and pooling API variants must agree with the *_into
 * results validated in main. reference is the batched row for ids0. */
static int check_alloc_and_pooling_variants(const embed_model_t *model,
                                            embed_workspace_t *ws,
                                            const int *ids0,
                                            int n0,
                                            const float *states,
                                            const float *reference,
                                            const embed_config_t *cfg) {
    int rc = 1;
    int dim = cfg->hidden_size;
    float *emb = NULL, *fwd = NULL, *pooled = NULL, *span_out = NULL;

    emb = embed_model_encode(model, ws, ids0, n0);
    fwd = embed_model_forward(model, ws, ids0, n0);
    pooled = (float *)calloc((size_t)dim, sizeof(float));
    span_out = (float *)calloc((size_t)2 * dim, sizeof(float));
    if (!emb || !fwd || !pooled || !span_out) {
        fprintf(stderr, "alloc-variant allocation failure\n");
        goto done;
    }

    if (embed_model_encode(NULL, ws, ids0, n0) != NULL ||
        embed_model_encode(model, ws, NULL, n0) != NULL ||
        embed_model_forward(model, ws, ids0, 0) != NULL ||
        embed_pool_batch(cfg, states, NULL, 1, pooled) == 0) {
        fprintf(stderr, "alloc variants accepted invalid arguments\n");
        goto done;
    }

    if (max_abs_diff(emb, reference, dim) > 0.00005f) {
        fprintf(stderr, "embed_model_encode disagrees with batched row\n");
        goto done;
    }
    if (max_abs_diff(fwd, states, n0 * dim) > 0.000001f) {
        fprintf(stderr, "embed_model_forward disagrees with forward_into\n");
        goto done;
    }

    /* Mean-pooling the final states reproduces the embedding. */
    if (embed_pool_batch(cfg, states, &n0, 1, pooled) != 0 ||
        max_abs_diff(pooled, reference, dim) > 0.00005f) {
        fprintf(stderr, "embed_pool_batch disagrees with embedding\n");
        goto done;
    }

    /* One span covering the whole sequence pools to the embedding, and
     * two halves recombine into it by token-count weighting. */
    embed_span_t whole = {0, n0};
    if (embed_model_encode_spans(model, ws, ids0, n0, &whole, 1, span_out) != 0 ||
        max_abs_diff(span_out, reference, dim) > 0.00005f) {
        fprintf(stderr, "whole-sequence span disagrees with embedding\n");
        goto done;
    }
    int h = n0 / 2;
    embed_span_t halves[2] = {{0, h}, {h, n0 - h}};
    if (embed_model_encode_spans(model, ws, ids0, n0, halves, 2, span_out) != 0) {
        fprintf(stderr, "embed_model_encode_spans failed for two spans\n");
        goto done;
    }
    for (int d = 0; d < dim; d++) {
        float combined = ((float)h * span_out[d] + (float)(n0 - h) * span_out[dim + d]) / (float)n0;
        if (fabsf(combined - reference[d]) > 0.00005f) {
            fprintf(stderr, "half spans do not recombine into embedding\n");
            goto done;
        }
    }

    rc = 0;
done:
    free(span_out);
    free(pooled);
    free(fwd);
    free(emb);
    return rc;
}

/* A multi-document contextual batch must match per-document runs.
 * doc0 is chunks 0 and 1 joined by a separator token; doc1 is chunk 2. */
static int check_spans_batch_parity(const embed_model_t *model,
                                    embed_workspace_t *ws,
                                    int *const ids[],
                                    const int *ntok,
                                    int separator_id,
                                    const embed_config_t *cfg) {
    int rc = 1;
    int dim = cfg->hidden_size;
    int n0 = ntok[0] + 1 + ntok[1];
    int *doc0 = (int *)malloc((size_t)n0 * sizeof(int));
    float *expected = (float *)calloc((size_t)3 * dim, sizeof(float));
    float *actual = (float *)calloc((size_t)3 * dim, sizeof(float));
    if (!doc0 || !expected || !actual) {
        fprintf(stderr, "spans-batch allocation failure\n");
        goto done;
    }
    memcpy(doc0, ids[0], (size_t)ntok[0] * sizeof(int));
    doc0[ntok[0]] = separator_id;
    memcpy(doc0 + ntok[0] + 1, ids[1], (size_t)ntok[1] * sizeof(int));

    embed_span_t spans0[2] = {{0, ntok[0]}, {ntok[0] + 1, ntok[1]}};
    embed_span_t span1 = {0, ntok[2]};
    if (embed_model_encode_spans(model, ws, doc0, n0, spans0, 2, expected) != 0 ||
        embed_model_encode_spans(model, ws, ids[2], ntok[2], &span1, 1,
                                 expected + (size_t)2 * dim) != 0) {
        fprintf(stderr, "singleton contextual embedding failed\n");
        goto done;
    }

    embed_context_input_t inputs[2] = {
        {{doc0, n0}, spans0, 2},
        {{ids[2], ntok[2]}, &span1, 1},
    };
    if (embed_model_encode_spans_batch(model, ws, inputs, 2, actual) != 0) {
        fprintf(stderr, "batched contextual embedding failed\n");
        goto done;
    }
    if (embed_model_encode_spans_batch(model, ws, NULL, 2, actual) == 0 ||
        embed_model_encode_spans_batch(model, ws, inputs, 0, actual) == 0) {
        fprintf(stderr, "spans batch accepted invalid arguments\n");
        goto done;
    }

    if (max_abs_diff(expected, actual, 3 * dim) > 0.00005f) {
        fprintf(stderr, "contextual batch disagrees with singletons: %g\n",
                max_abs_diff(expected, actual, 3 * dim));
        goto done;
    }

    rc = 0;
done:
    free(actual);
    free(expected);
    free(doc0);
    return rc;
}

int main(int argc, char **argv) {
    if (check_alloc_helpers() != 0 || check_vector_helpers() != 0)
        return 1;

    char fixture_dir[1024];
    const char *model_dir;
    if (argc == 2) {
        model_dir = argv[1];
    } else if (argc == 1) {
        /* Tiny hidden size keeps it fast; the vocab must cover every id the
         * byte-complete tokenizer fixture can emit. */
        tm_dims_t dims = {4, 2, 1, 2, 8, TF_VOCAB_SIZE};
        snprintf(fixture_dir, sizeof(fixture_dir), "%s/embed-ws-test-XXXXXX",
                 getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
        if (!mkdtemp(fixture_dir) || tf_write_vocab(fixture_dir) != 0 ||
            tm_write_model_dims(fixture_dir, "F32", &dims) != 0) {
            fprintf(stderr, "fixture creation failed\n");
            return 2;
        }
        model_dir = fixture_dir;
    } else {
        fprintf(stderr, "usage: %s [MODEL_DIR]\n", argv[0]);
        return 2;
    }
    char vocab_path[4096];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);

    embed_model_t *model = embed_model_load(model_dir);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const embed_config_t *cfg = embed_model_config(model);
    if (!cfg || cfg->hidden_size <= 0) {
        fprintf(stderr, "invalid model config\n");
        embed_model_free(model);
        return 1;
    }

    embed_workspace_t *batch_ws = embed_workspace_new(model);
    embed_workspace_t *single_ws = embed_workspace_new(model);
    embed_tokenizer_t *tok = embed_tokenizer_load(vocab_path);
    if (!batch_ws || !single_ws || !tok) {
        fprintf(stderr, "failed to allocate workspace or tokenizer\n");
        embed_tokenizer_free(tok);
        embed_workspace_free(single_ws);
        embed_workspace_free(batch_ws);
        embed_model_free(model);
        return 1;
    }

    const char *texts[] = {"query: what is the capital of France?",
                           "document: Paris is the capital of France.",
                           "document: Berlin is the capital of Germany.",
                           "document: Istanbul is a major city in Turkey."};
    const int batch = (int)(sizeof(texts) / sizeof(texts[0]));
    /* Initialized before the first `goto fail` so the cleanup loop never
     * sees indeterminate pointers. */
    int *ids[4] = {0};
    int ntok[4] = {0};
    embed_input_t inputs[4];

    size_t initial_ws_bytes = embed_workspace_nbytes(batch_ws);
    if (initial_ws_bytes == 0) {
        fprintf(stderr, "workspace byte accounting returned zero\n");
        goto fail;
    }

    for (int i = 0; i < batch; i++) {
        ids[i] = embed_tokenizer_encode(tok, texts[i], &ntok[i]);
        if (!ids[i] || ntok[i] <= 0) {
            fprintf(stderr, "failed to tokenize input %d\n", i);
            goto fail;
        }
        inputs[i].ids = ids[i];
        inputs[i].n_tokens = ntok[i];
    }

    int dim = cfg->hidden_size;
    float *batched = (float *)calloc((size_t)batch * dim, sizeof(float));
    float *single = (float *)calloc((size_t)dim, sizeof(float));
    float *normalized = (float *)calloc((size_t)dim, sizeof(float));
    float *states = (float *)calloc((size_t)ntok[0] * dim, sizeof(float));
    if (!batched || !single || !normalized || !states) {
        fprintf(stderr, "allocation failure\n");
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    if (embed_model_encode_batch(model, batch_ws, inputs, batch, batched) != 0) {
        fprintf(stderr, "embed_model_encode_batch failed\n");
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    size_t batch_ws_bytes = embed_workspace_nbytes(batch_ws);
    if (batch_ws_bytes <= initial_ws_bytes) {
        fprintf(stderr, "workspace byte accounting did not grow after embedding\n");
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    float worst_diff = 0.0f;
    float worst_cos = 1.0f;
    for (int i = 0; i < batch; i++) {
        if (embed_model_encode_into(model, single_ws, ids[i], ntok[i], single) != 0) {
            fprintf(stderr, "embed_model_encode_into failed at %d\n", i);
            free(states);
            free(normalized);
            free(single);
            free(batched);
            goto fail;
        }

        const float *row = batched + (size_t)i * dim;
        float diff = max_abs_diff(row, single, dim);
        float cos = embed_cosine_similarity(row, single, dim);
        if (diff > worst_diff)
            worst_diff = diff;
        if (cos < worst_cos)
            worst_cos = cos;
    }

    memcpy(normalized, batched, (size_t)dim * sizeof(float));
    if (embed_l2_normalize(normalized, dim) != 0) {
        fprintf(stderr, "embed_l2_normalize failed\n");
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }
    float normalized_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++)
        normalized_norm_sq += normalized[i] * normalized[i];
    if (fabsf(normalized_norm_sq - 1.0f) > 0.00005f) {
        fprintf(stderr, "embed_l2_normalize produced norm_sq=%g\n", normalized_norm_sq);
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    if (embed_model_forward_into(model, batch_ws, ids[0], ntok[0], states) != 0) {
        fprintf(stderr, "embed_model_forward_into failed\n");
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    if (check_alloc_and_pooling_variants(model, batch_ws, ids[0], ntok[0], states, batched, cfg) !=
        0) {
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    /* Same separator resolution as the server: the fixture vocab defines
     * <|endoftext|>, real snapshots use the reserved id. */
    int sep_id = embed_tokenizer_token_id(tok, "<|endoftext|>");
    if (sep_id < 0)
        sep_id = EMBED_CONTEXT_SEPARATOR_TOKEN_ID;
    if (check_spans_batch_parity(model, batch_ws, ids, ntok, sep_id, cfg) != 0) {
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    free(states);
    free(normalized);
    free(single);
    free(batched);

    for (int i = 0; i < batch; i++)
        free(ids[i]);
    embed_tokenizer_free(tok);
    embed_workspace_free(single_ws);
    embed_workspace_free(batch_ws);
    embed_model_free(model);

    if (worst_diff > 0.00005f || worst_cos < 0.99999f) {
        fprintf(stderr, "workspace API parity failed: max_abs_diff=%g cosine=%g\n", worst_diff,
                worst_cos);
        return 1;
    }

    printf("ok: workspace API dim=%d batch=%d workspace=%zu bytes max_abs_diff=%g cosine=%g\n", dim,
           batch, batch_ws_bytes, worst_diff, worst_cos);
    return 0;

fail:
    for (int i = 0; i < batch; i++)
        free(ids[i]);
    embed_tokenizer_free(tok);
    embed_workspace_free(single_ws);
    embed_workspace_free(batch_ws);
    embed_model_free(model);
    return 1;
}
