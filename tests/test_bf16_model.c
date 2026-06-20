/* tests/test_bf16_model.c - F32 vs BF16 vs F16 loader/dispatch parity on a tiny
 * synthetic model. Hermetic: builds its own model dirs. Runs via `make test`.
 * F16 has no fused kernel: the loader widens it to F32 once at load, so an F16
 * copy must match the F32 copy as closely as the BF16 copy does. */

#include "internal.h"
#include "tiny_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

int main(void) {
    char root[1024], f32_dir[1088], bf16_dir[1088], f16_dir[1088], prefixed_dir[1088];
    snprintf(root, sizeof(root), "%s/ffwd-bf16-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(root)) {
        perror("mkdtemp");
        return 2;
    }
    snprintf(f32_dir, sizeof(f32_dir), "%s/f32", root);
    snprintf(bf16_dir, sizeof(bf16_dir), "%s/bf16", root);
    snprintf(f16_dir, sizeof(f16_dir), "%s/f16", root);
    snprintf(prefixed_dir, sizeof(prefixed_dir), "%s/prefixed", root);
    static const tm_dims_t dims = {4, 2, 1, 2, 8, 16};
    if (mkdir(f32_dir, 0755) != 0 || mkdir(bf16_dir, 0755) != 0 || mkdir(f16_dir, 0755) != 0 ||
        mkdir(prefixed_dir, 0755) != 0 || tm_write_model(f32_dir, "F32") != 0 ||
        tm_write_model(bf16_dir, "BF16") != 0 || tm_write_model(f16_dir, "F16") != 0 ||
        tm_write_prefixed_model_dims(prefixed_dir, "F32", &dims) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    int ids[] = {1, 2, 3, 4};
    ffwd_model_t *mf32 = ffwd_model_load(f32_dir);
    ffwd_model_t *mbf16 = ffwd_model_load(bf16_dir);
    ffwd_model_t *mf16 = ffwd_model_load(f16_dir);
    ffwd_model_t *mprefixed = ffwd_model_load(prefixed_dir);
    if (!mf32 || !mbf16 || !mf16 || !mprefixed) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    const ffwd_config_t *cfg = ffwd_model_config(mf32);
    int dim = cfg->hidden_size;
    float *a = (float *)malloc((size_t)dim * sizeof(float));
    float *b = (float *)malloc((size_t)dim * sizeof(float));
    float *c = (float *)malloc((size_t)dim * sizeof(float));
    float *e = (float *)malloc((size_t)dim * sizeof(float));
    ffwd_workspace_t *wf32 = ffwd_workspace_new(mf32);
    ffwd_workspace_t *wbf16 = ffwd_workspace_new(mbf16);
    ffwd_workspace_t *wf16 = ffwd_workspace_new(mf16);
    ffwd_workspace_t *wprefixed = ffwd_workspace_new(mprefixed);
    if (!a || !b || !c || !e || !wf32 || !wbf16 || !wf16 || !wprefixed) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    if (ffwd_model_encode_into(mf32, wf32, ids, 4, a) != 0 ||
        ffwd_model_encode_into(mbf16, wbf16, ids, 4, b) != 0 ||
        ffwd_model_encode_into(mf16, wf16, ids, 4, e) != 0 ||
        ffwd_model_encode_into(mprefixed, wprefixed, ids, 4, c) != 0) {
        fprintf(stderr, "embedding failed\n");
        return 1;
    }

    float max_diff = 0.0f;
    float f16_diff = 0.0f;
    float prefix_diff = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(e[i]) || !isfinite(c[i])) {
            fprintf(stderr, "non-finite embedding value\n");
            return 1;
        }
        float d = fabsf(a[i] - b[i]);
        if (d > max_diff)
            max_diff = d;
        d = fabsf(a[i] - e[i]);
        if (d > f16_diff)
            f16_diff = d;
        d = fabsf(a[i] - c[i]);
        if (d > prefix_diff)
            prefix_diff = d;
        norm += b[i] * b[i];
    }
    norm = sqrtf(norm);
    /* Embeddings are intentionally unnormalized; require a sane nonzero
     * magnitude and tight F32-vs-BF16/F16 parity, not unit norm. F16 carries
     * more mantissa bits than BF16, so the same 2e-5 bound holds. */
    if (norm <= 1e-6f || max_diff > 2e-5f || f16_diff > 2e-5f || prefix_diff != 0.0f) {
        fprintf(stderr, "bad parity: norm=%.9g max_diff=%.9g f16_diff=%.9g prefix_diff=%.9g\n", norm,
                max_diff, f16_diff, prefix_diff);
        return 1;
    }

    /* Sequences longer than 16 tokens take the widen-to-F32 weight path
     * (linear_nobias_weight); parity must hold there too. */
    enum { LONG_N = 24 };
    int long_ids[LONG_N];
    for (int i = 0; i < LONG_N; i++)
        long_ids[i] = (i % 14) + 1;
    if (ffwd_model_encode_into(mf32, wf32, long_ids, LONG_N, a) != 0 ||
        ffwd_model_encode_into(mbf16, wbf16, long_ids, LONG_N, b) != 0 ||
        ffwd_model_encode_into(mf16, wf16, long_ids, LONG_N, e) != 0) {
        fprintf(stderr, "long-sequence embedding failed\n");
        return 1;
    }
    float long_diff = 0.0f, long_f16_diff = 0.0f;
    for (int i = 0; i < dim; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(e[i])) {
            fprintf(stderr, "non-finite long-sequence value\n");
            return 1;
        }
        float d = fabsf(a[i] - b[i]);
        if (d > long_diff)
            long_diff = d;
        d = fabsf(a[i] - e[i]);
        if (d > long_f16_diff)
            long_f16_diff = d;
    }
    if (long_diff > 2e-5f || long_f16_diff > 2e-5f) {
        fprintf(stderr, "bad widen-path parity: bf16=%.9g f16=%.9g\n", long_diff, long_f16_diff);
        return 1;
    }

    printf("ok: bf16/f16 and model-prefix parity dim=%d bf16_diff=%.9g f16_diff=%.9g norm=%.9g\n",
           dim, max_diff, f16_diff, norm);

    ffwd_workspace_free(wf32);
    ffwd_workspace_free(wbf16);
    ffwd_workspace_free(wf16);
    ffwd_workspace_free(wprefixed);
    ffwd_model_free(mf32);
    ffwd_model_free(mbf16);
    ffwd_model_free(mf16);
    ffwd_model_free(mprefixed);
    free(a);
    free(b);
    free(c);
    free(e);
    return 0;
}
