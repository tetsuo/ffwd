/* tests/test_cls.c - hermetic CLS (first-token) pooling across the load path,
 * the encode path, and the public pooling API. */

#include "internal.h"
#include "tiny_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static float max_abs_diff(const float *a, const float *b, int n) {
    float out = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > out)
            out = d;
    }
    return out;
}

int main(void) {
    const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/ffwd-cls-test-XXXXXX", tmp);
    tm_dims_t dims = {4, 2, 1, 2, 8, 16};
    if (!mkdtemp(dir) || tm_write_qwen2_model_pooling(dir, "F32", &dims, 15, 0, "cls") != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    ffwd_model_t *model = ffwd_model_load(dir);
    ffwd_workspace_t *ws = ffwd_workspace_new(model);
    const ffwd_config_t *cfg = ffwd_model_config(model);
    if (!model || !ws || !cfg) {
        fprintf(stderr, "model setup failed\n");
        return 1;
    }
    if (cfg->pooling_mode != FFWD_POOL_CLS || cfg->attention_mode != FFWD_ATTENTION_BIDIRECTIONAL ||
        !cfg->normalize_embeddings) {
        fprintf(stderr, "CLS pooling metadata was not detected\n");
        return 1;
    }

    int ids[] = {1, 2, 15};
    float states[3 * 4], embedding[4], pooled[4];
    if (ffwd_model_forward_into(model, ws, ids, 3, states) != 0 ||
        ffwd_model_encode_into(model, ws, ids, 3, embedding) != 0 ||
        ffwd_pool_batch(cfg, states, (int[]){3}, 1, pooled) != 0) {
        fprintf(stderr, "CLS inference failed\n");
        return 1;
    }

    /* CLS pools the first token's (already final-normed) state. Both the encode
     * path and the public pooling API must return that L2-normalized vector. */
    float expected[4];
    memcpy(expected, states, sizeof(expected));
    if (ffwd_l2_normalize(expected, 4) != 0 || max_abs_diff(embedding, expected, 4) > 1e-5f ||
        max_abs_diff(pooled, expected, 4) > 1e-5f) {
        fprintf(stderr, "CLS pooling did not return the normalized first token\n");
        return 1;
    }

    /* Guard against a symmetric fixture: the first token must differ from the
     * last, so selecting the wrong end would fail this test. */
    float last[4];
    memcpy(last, states + 2 * 4, sizeof(last));
    if (ffwd_l2_normalize(last, 4) != 0 || max_abs_diff(expected, last, 4) < 1e-4f) {
        fprintf(stderr, "fixture too symmetric: first and last tokens coincide\n");
        return 1;
    }

    /* Packed batch parity: each row's CLS vector equals its singleton CLS. */
    int short_ids[] = {4, 15};
    ffwd_input_t inputs[] = {{ids, 3}, {short_ids, 2}};
    float batch[8], single[4];
    if (ffwd_model_encode_batch(model, ws, inputs, 2, batch) != 0 ||
        ffwd_model_encode_into(model, ws, short_ids, 2, single) != 0 ||
        max_abs_diff(batch, embedding, 4) > 1e-5f || max_abs_diff(batch + 4, single, 4) > 1e-5f) {
        fprintf(stderr, "CLS packed batch parity failed\n");
        return 1;
    }

    ffwd_workspace_free(ws);
    ffwd_model_free(model);
    puts("ok: CLS first-token pooling and packed batch parity");
    return 0;
}
