/* tests/test_workspace.c - model/workspace API lifecycle tests.
 * Hermetic by default (synthesizes a tiny model + tokenizer fixture);
 * pass a MODEL_DIR to run the same checks against real weights, which is
 * what scripts/check_workspace_api.py does. Runs via `make test`. */

#include "embed.h"
#include "qwen_tokenizer.h"
#include "tiny_model.h"
#include "tok_fixture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static float max_abs_diff(const float *a, const float *b, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* The allocating and pooling API variants must agree with the *_into
 * results validated in main. reference is the batched row for ids0. */
static int check_alloc_and_pooling_variants(const pplx_model_t *model,
                                            pplx_workspace_t *ws,
                                            const int *ids0, int n0,
                                            const float *states,
                                            const float *reference,
                                            const pplx_config_t *cfg)
{
    int rc = 1;
    int dim = cfg->hidden_size;
    float *emb = NULL, *fwd = NULL, *pooled = NULL, *span_out = NULL;

    emb = pplx_model_embed(model, ws, ids0, n0);
    fwd = pplx_model_forward(model, ws, ids0, n0);
    pooled = (float *)calloc((size_t)dim, sizeof(float));
    span_out = (float *)calloc((size_t)2 * dim, sizeof(float));
    if (!emb || !fwd || !pooled || !span_out) {
        fprintf(stderr, "alloc-variant allocation failure\n");
        goto done;
    }

    if (pplx_model_embed(NULL, ws, ids0, n0) != NULL ||
        pplx_model_embed(model, ws, NULL, n0) != NULL ||
        pplx_model_forward(model, ws, ids0, 0) != NULL ||
        pplx_pool_batch(cfg, states, NULL, 1, pooled) == 0) {
        fprintf(stderr, "alloc variants accepted invalid arguments\n");
        goto done;
    }

    if (max_abs_diff(emb, reference, dim) > 0.00005f) {
        fprintf(stderr, "pplx_model_embed disagrees with batched row\n");
        goto done;
    }
    if (max_abs_diff(fwd, states, n0 * dim) > 0.000001f) {
        fprintf(stderr, "pplx_model_forward disagrees with forward_into\n");
        goto done;
    }

    /* Mean-pooling the final states reproduces the embedding. */
    if (pplx_pool_batch(cfg, states, &n0, 1, pooled) != 0 ||
        max_abs_diff(pooled, reference, dim) > 0.00005f) {
        fprintf(stderr, "pplx_pool_batch disagrees with embedding\n");
        goto done;
    }

    /* One span covering the whole sequence pools to the embedding, and
     * two halves recombine into it by token-count weighting. */
    pplx_span_t whole = {0, n0};
    if (pplx_model_embed_spans(model, ws, ids0, n0, &whole, 1,
                               span_out) != 0 ||
        max_abs_diff(span_out, reference, dim) > 0.00005f) {
        fprintf(stderr, "whole-sequence span disagrees with embedding\n");
        goto done;
    }
    int h = n0 / 2;
    pplx_span_t halves[2] = {{0, h}, {h, n0 - h}};
    if (pplx_model_embed_spans(model, ws, ids0, n0, halves, 2,
                               span_out) != 0) {
        fprintf(stderr, "pplx_model_embed_spans failed for two spans\n");
        goto done;
    }
    for (int d = 0; d < dim; d++) {
        float combined = ((float)h * span_out[d] +
                          (float)(n0 - h) * span_out[dim + d]) / (float)n0;
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

int main(int argc, char **argv)
{
    char fixture_dir[1024];
    const char *model_dir;
    if (argc == 2) {
        model_dir = argv[1];
    } else if (argc == 1) {
        /* Tiny hidden size keeps it fast; the vocab must cover every id the
         * byte-complete tokenizer fixture can emit. */
        tm_dims_t dims = {4, 2, 1, 2, 8, TF_VOCAB_SIZE};
        snprintf(fixture_dir, sizeof(fixture_dir), "%s/pplx-ws-test-XXXXXX",
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

    pplx_model_t *model = pplx_model_load(model_dir);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const pplx_config_t *cfg = pplx_model_config(model);
    if (!cfg || cfg->hidden_size <= 0) {
        fprintf(stderr, "invalid model config\n");
        pplx_model_free(model);
        return 1;
    }

    pplx_workspace_t *batch_ws = pplx_workspace_new(model);
    pplx_workspace_t *single_ws = pplx_workspace_new(model);
    qwen_tokenizer_t *tok = qwen_tokenizer_load(vocab_path);
    if (!batch_ws || !single_ws || !tok) {
        fprintf(stderr, "failed to allocate workspace or tokenizer\n");
        qwen_tokenizer_free(tok);
        pplx_workspace_free(single_ws);
        pplx_workspace_free(batch_ws);
        pplx_model_free(model);
        return 1;
    }

    const char *texts[] = {
        "query: what is the capital of France?",
        "document: Paris is the capital of France.",
        "document: Berlin is the capital of Germany.",
        "document: Istanbul is a major city in Turkey."
    };
    const int batch = (int)(sizeof(texts) / sizeof(texts[0]));
    /* Initialized before the first `goto fail` so the cleanup loop never
     * sees indeterminate pointers. */
    int *ids[4] = {0};
    int ntok[4] = {0};
    pplx_input_t inputs[4];

    size_t initial_ws_bytes = pplx_workspace_nbytes(batch_ws);
    if (initial_ws_bytes == 0) {
        fprintf(stderr, "workspace byte accounting returned zero\n");
        goto fail;
    }

    for (int i = 0; i < batch; i++) {
        ids[i] = qwen_tokenizer_encode(tok, texts[i], &ntok[i]);
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

    if (pplx_model_embed_batch(model, batch_ws, inputs, batch, batched) != 0) {
        fprintf(stderr, "pplx_model_embed_batch failed\n");
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    size_t batch_ws_bytes = pplx_workspace_nbytes(batch_ws);
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
        if (pplx_model_embed_into(model, single_ws, ids[i], ntok[i], single) != 0) {
            fprintf(stderr, "pplx_model_embed_into failed at %d\n", i);
            free(states);
            free(normalized);
            free(single);
            free(batched);
            goto fail;
        }

        const float *row = batched + (size_t)i * dim;
        float diff = max_abs_diff(row, single, dim);
        float cos = pplx_cosine_similarity(row, single, dim);
        if (diff > worst_diff) worst_diff = diff;
        if (cos < worst_cos) worst_cos = cos;
    }

    memcpy(normalized, batched, (size_t)dim * sizeof(float));
    if (pplx_l2_normalize(normalized, dim) != 0) {
        fprintf(stderr, "pplx_l2_normalize failed\n");
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
        fprintf(stderr, "pplx_l2_normalize produced norm_sq=%g\n",
                normalized_norm_sq);
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    if (pplx_model_forward_into(model, batch_ws, ids[0], ntok[0], states) != 0) {
        fprintf(stderr, "pplx_model_forward_into failed\n");
        free(states);
        free(normalized);
        free(single);
        free(batched);
        goto fail;
    }

    if (check_alloc_and_pooling_variants(model, batch_ws, ids[0], ntok[0],
                                         states, batched, cfg) != 0) {
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

    for (int i = 0; i < batch; i++) free(ids[i]);
    qwen_tokenizer_free(tok);
    pplx_workspace_free(single_ws);
    pplx_workspace_free(batch_ws);
    pplx_model_free(model);

    if (worst_diff > 0.00005f || worst_cos < 0.99999f) {
        fprintf(stderr, "workspace API parity failed: max_abs_diff=%g cosine=%g\n",
                worst_diff, worst_cos);
        return 1;
    }

    printf("ok: workspace API dim=%d batch=%d workspace=%zu bytes max_abs_diff=%g cosine=%g\n",
           dim, batch, batch_ws_bytes, worst_diff, worst_cos);
    return 0;

fail:
    for (int i = 0; i < batch; i++) free(ids[i]);
    qwen_tokenizer_free(tok);
    pplx_workspace_free(single_ws);
    pplx_workspace_free(batch_ws);
    pplx_model_free(model);
    return 1;
}
