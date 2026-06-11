/* tests/test_bf16_model.c - F32 vs BF16 loader/dispatch parity on a tiny
 * synthetic model. Hermetic: builds its own model dirs. Runs via
 * `make test`. */

#include "embed.h"
#include "tiny_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int main(void) {
    char root[1024], f32_dir[1088], bf16_dir[1088];
    snprintf(root, sizeof(root), "%s/embed-bf16-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(root)) { perror("mkdtemp"); return 2; }
    snprintf(f32_dir, sizeof(f32_dir), "%s/f32", root);
    snprintf(bf16_dir, sizeof(bf16_dir), "%s/bf16", root);
    if (mkdir(f32_dir, 0755) != 0 || mkdir(bf16_dir, 0755) != 0 ||
        tm_write_model(f32_dir, "F32") != 0 ||
        tm_write_model(bf16_dir, "BF16") != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    int ids[] = {1, 2, 3, 4};
    embed_model_t *mf32 = embed_model_load(f32_dir);
    embed_model_t *mbf16 = embed_model_load(bf16_dir);
    if (!mf32 || !mbf16) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    const embed_config_t *cfg = embed_model_config(mf32);
    int dim = cfg->hidden_size;
    float *a = (float *)malloc((size_t)dim * sizeof(float));
    float *b = (float *)malloc((size_t)dim * sizeof(float));
    embed_workspace_t *wf32 = embed_workspace_new(mf32);
    embed_workspace_t *wbf16 = embed_workspace_new(mbf16);
    if (!a || !b || !wf32 || !wbf16) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    if (embed_model_encode_into(mf32, wf32, ids, 4, a) != 0 ||
        embed_model_encode_into(mbf16, wbf16, ids, 4, b) != 0) {
        fprintf(stderr, "embedding failed\n");
        return 1;
    }

    float max_diff = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            fprintf(stderr, "non-finite embedding value\n");
            return 1;
        }
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff) max_diff = d;
        norm += b[i] * b[i];
    }
    norm = sqrtf(norm);
    /* Embeddings are intentionally unnormalized; require a sane nonzero
     * magnitude and tight F32-vs-BF16 parity, not unit norm. */
    if (norm <= 1e-6f || max_diff > 2e-5f) {
        fprintf(stderr, "bad parity: norm=%.9g max_diff=%.9g\n", norm, max_diff);
        return 1;
    }

    /* Sequences longer than 16 tokens take the widen-to-F32 weight path
     * (linear_nobias_weight); parity must hold there too. */
    enum { LONG_N = 24 };
    int long_ids[LONG_N];
    for (int i = 0; i < LONG_N; i++) long_ids[i] = (i % 14) + 1;
    if (embed_model_encode_into(mf32, wf32, long_ids, LONG_N, a) != 0 ||
        embed_model_encode_into(mbf16, wbf16, long_ids, LONG_N, b) != 0) {
        fprintf(stderr, "long-sequence embedding failed\n");
        return 1;
    }
    float long_diff = 0.0f;
    for (int i = 0; i < dim; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            fprintf(stderr, "non-finite long-sequence value\n");
            return 1;
        }
        float d = fabsf(a[i] - b[i]);
        if (d > long_diff) long_diff = d;
    }
    if (long_diff > 2e-5f) {
        fprintf(stderr, "bad widen-path parity: max_diff=%.9g\n", long_diff);
        return 1;
    }

    printf("ok: bf16 model parity dim=%d max_abs_diff=%.9g norm=%.9g\n",
           dim, max_diff, norm);

    embed_workspace_free(wf32);
    embed_workspace_free(wbf16);
    embed_model_free(mf32);
    embed_model_free(mbf16);
    free(a);
    free(b);
    return 0;
}
