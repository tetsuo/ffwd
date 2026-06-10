/* tests/test_bf16_model.c - F32 vs BF16 loader/dispatch parity on a tiny
 * synthetic model. Runs via `make test`. */

#include "embed.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s f32_model bf16_model\n", argv[0]);
        return 2;
    }

    int ids[] = {1, 2, 3, 4};
    pplx_model_t *mf32 = pplx_model_load(argv[1]);
    pplx_model_t *mbf16 = pplx_model_load(argv[2]);
    if (!mf32 || !mbf16) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    const pplx_config_t *cfg = pplx_model_config(mf32);
    int dim = cfg->hidden_size;
    float *a = (float *)malloc((size_t)dim * sizeof(float));
    float *b = (float *)malloc((size_t)dim * sizeof(float));
    pplx_workspace_t *wf32 = pplx_workspace_new(mf32);
    pplx_workspace_t *wbf16 = pplx_workspace_new(mbf16);
    if (!a || !b || !wf32 || !wbf16) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    if (pplx_model_embed_into(mf32, wf32, ids, 4, a) != 0 ||
        pplx_model_embed_into(mbf16, wbf16, ids, 4, b) != 0) {
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

    printf("ok: bf16 model parity dim=%d max_abs_diff=%.9g norm=%.9g\n",
           dim, max_diff, norm);

    pplx_workspace_free(wf32);
    pplx_workspace_free(wbf16);
    pplx_model_free(mf32);
    pplx_model_free(mbf16);
    free(a);
    free(b);
    return 0;
}
