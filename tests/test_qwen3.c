/* tests/test_qwen3.c - hermetic Qwen3 model semantics. */

#include "embed.h"
#include "tiny_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static float max_abs_diff(const float *a, const float *b, int n)
{
    float out = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > out) out = d;
    }
    return out;
}

int main(void)
{
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/embed-qwen3-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    tm_dims_t dims = {4, 2, 1, 2, 8, 16};
    if (!mkdtemp(dir) ||
        tm_write_qwen3_model_dims(dir, "F32", &dims, 15) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    embed_model_t *model = embed_model_load(dir);
    embed_workspace_t *ws = embed_workspace_new(model);
    const embed_config_t *cfg = embed_model_config(model);
    if (!model || !ws || !cfg) {
        fprintf(stderr, "model setup failed\n");
        return 1;
    }
    if (cfg->attention_mode != EMBED_ATTENTION_CAUSAL ||
        cfg->pooling_mode != EMBED_POOL_LAST_TOKEN ||
        !cfg->normalize_embeddings || !cfg->append_terminal_token) {
        fprintf(stderr, "Qwen3 config semantics were not detected\n");
        return 1;
    }

    int ids[] = {1, 2, 3, 15};
    float states[4 * 4];
    float embedding[4];
    float pooled[4];
    if (embed_model_forward_into(model, ws, ids, 4, states) != 0 ||
        embed_model_encode_into(model, ws, ids, 4, embedding) != 0 ||
        embed_pool_batch(cfg, states, (int[]){4}, 1, pooled) != 0) {
        fprintf(stderr, "Qwen3 inference failed\n");
        return 1;
    }

    float expected[4];
    memcpy(expected, states + 3 * 4, sizeof(expected));
    if (embed_l2_normalize(expected, 4) != 0 ||
        max_abs_diff(embedding, expected, 4) > 1e-5f ||
        max_abs_diff(pooled, expected, 4) > 1e-5f) {
        fprintf(stderr, "last-token pooling mismatch\n");
        return 1;
    }

    float norm_sq = 0.0f;
    for (int d = 0; d < 4; d++) norm_sq += embedding[d] * embedding[d];
    if (fabsf(norm_sq - 1.0f) > 1e-5f) {
        fprintf(stderr, "Qwen3 embedding is not normalized: %.9g\n", norm_sq);
        return 1;
    }

    int short_ids[] = {4, 15};
    embed_input_t inputs[] = {{ids, 4}, {short_ids, 2}};
    float batch[8];
    float single[4];
    if (embed_model_encode_batch(model, ws, inputs, 2, batch) != 0 ||
        embed_model_encode_into(model, ws, short_ids, 2, single) != 0 ||
        max_abs_diff(batch, embedding, 4) > 1e-5f ||
        max_abs_diff(batch + 4, single, 4) > 1e-5f) {
        fprintf(stderr, "Qwen3 packed batch parity failed\n");
        return 1;
    }

    embed_workspace_free(ws);
    embed_model_free(model);
    puts("ok: Qwen3 causal attention and last-token pooling");
    return 0;
}
