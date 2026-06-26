/* Hermetic GTE-Qwen2 block and task semantics. */

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
    char dir[1024], zero_dir[1024];
    snprintf(dir, sizeof(dir), "%s/ffwd-qwen2-test-XXXXXX", tmp);
    snprintf(zero_dir, sizeof(zero_dir), "%s/ffwd-qwen2-zero-bias-test-XXXXXX", tmp);
    tm_dims_t dims = {4, 2, 1, 2, 8, 16};
    if (!mkdtemp(dir) || !mkdtemp(zero_dir) ||
        tm_write_qwen2_model_dims(dir, "F32", &dims, 15, 0) != 0 ||
        tm_write_qwen2_model_dims(zero_dir, "F32", &dims, 15, 1) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    ffwd_model_t *model = ffwd_model_load(dir);
    ffwd_model_t *zero_model = ffwd_model_load(zero_dir);
    ffwd_workspace_t *ws = ffwd_workspace_new(model);
    ffwd_workspace_t *zero_ws = ffwd_workspace_new(zero_model);
    const ffwd_config_t *cfg = ffwd_model_config(model);
    if (!model || !zero_model || !ws || !zero_ws || !cfg) {
        fprintf(stderr, "model setup failed\n");
        return 1;
    }
    if (cfg->qk_norm || !cfg->qkv_bias || cfg->attention_mode != FFWD_ATTENTION_BIDIRECTIONAL ||
        cfg->pooling_mode != FFWD_POOL_LAST_TOKEN || !cfg->normalize_embeddings ||
        !cfg->append_terminal_token) {
        fprintf(stderr, "Qwen2 config or Sentence Transformers metadata was not detected\n");
        return 1;
    }

    int ids[] = {1, 2, 15};
    float states[3 * 4], embedding[4], pooled[4], zero_embedding[4];
    if (ffwd_model_forward_into(model, ws, ids, 3, states) != 0 ||
        ffwd_model_encode_into(model, ws, ids, 3, embedding) != 0 ||
        ffwd_model_encode_into(zero_model, zero_ws, ids, 3, zero_embedding) != 0 ||
        ffwd_pool_batch(cfg, states, (int[]){3}, 1, pooled) != 0) {
        fprintf(stderr, "Qwen2 inference failed\n");
        return 1;
    }

    float expected[4];
    memcpy(expected, states + 2 * 4, sizeof(expected));
    if (ffwd_l2_normalize(expected, 4) != 0 || max_abs_diff(embedding, expected, 4) > 1e-5f ||
        max_abs_diff(pooled, expected, 4) > 1e-5f) {
        fprintf(stderr, "last-token pooling mismatch\n");
        return 1;
    }
    if (max_abs_diff(embedding, zero_embedding, 4) < 1e-4f) {
        fprintf(stderr, "QKV projection bias did not affect the embedding\n");
        return 1;
    }

    int changed_ids[] = {1, 3, 15};
    float changed_states[3 * 4];
    if (ffwd_model_forward_into(model, ws, changed_ids, 3, changed_states) != 0 ||
        max_abs_diff(states, changed_states, 4) < 1e-6f) {
        fprintf(stderr, "bidirectional attention did not affect the first token\n");
        return 1;
    }

    int short_ids[] = {4, 15};
    ffwd_input_t inputs[] = {{ids, 3}, {short_ids, 2}};
    float batch[8], single[4];
    if (ffwd_model_encode_batch(model, ws, inputs, 2, batch) != 0 ||
        ffwd_model_encode_into(model, ws, short_ids, 2, single) != 0 ||
        max_abs_diff(batch, embedding, 4) > 1e-5f || max_abs_diff(batch + 4, single, 4) > 1e-5f) {
        fprintf(stderr, "Qwen2 packed batch parity failed\n");
        return 1;
    }

    ffwd_workspace_free(zero_ws);
    ffwd_workspace_free(ws);
    ffwd_model_free(zero_model);
    ffwd_model_free(model);
    puts("ok: Qwen2 bias, bidirectional attention, and GTE pooling semantics");
    return 0;
}
