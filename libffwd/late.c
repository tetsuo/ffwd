/*
 * Late-interaction model.
 *
 * A late-interaction model is a pooled base model plus a 1_Dense linear
 * projection from hidden states to token vectors.
 *
 * Instead of pooling to one vector, it keeps one projected vector per kept token
 * and scores query/doc pairs with MaxSim.
 *
 * Reuses the base model's packed forward pass and workspace. This layer only
 * owns the projection, token-vector buffer, and scoring.
 */

#include "ffwd.h"
#include "model_internal.h"
#include "kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>

struct ffwd_late_model {
    ffwd_model_t *base;
    safetensors_file_t *dense_sf;
    ffwd_weight_ref_t projection; /* [token_dim, hidden] */
    int token_dim;
};

static const safetensor_t *find_single_safetensor(const safetensors_file_t *sf, const char *name) {
    if (!sf || !name)
        return NULL;
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0)
            return &sf->tensors[i];
    }
    return NULL;
}

ffwd_late_model_t *ffwd_late_model_load(const char *model_dir) {
    if (!model_dir_has_late_projection(model_dir)) {
        fprintf(stderr, "ffwd_late_model_load: missing 1_Dense projection\n");
        return NULL;
    }

    ffwd_late_model_t *late = (ffwd_late_model_t *)calloc(1, sizeof(*late));
    if (!late)
        return NULL;

    late->base = model_load_range_ex(model_dir, 0, -1, 1);
    if (!late->base)
        goto fail;

    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/1_Dense/model.safetensors", model_dir);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ffwd_late_model_load: model path too long\n");
        goto fail;
    }

    late->dense_sf = safetensors_open(path);
    if (!late->dense_sf) {
        fprintf(stderr, "ffwd_late_model_load: cannot open %s\n", path);
        goto fail;
    }

    const safetensor_t *t = find_single_safetensor(late->dense_sf, "linear.weight");
    if (!t) {
        fprintf(stderr, "ffwd_late_model_load: missing linear.weight\n");
        goto fail;
    }
    if (t->ndim != 2 || t->shape[0] <= 0 || t->shape[0] > INT_MAX) {
        fprintf(stderr, "ffwd_late_model_load: bad projection shape\n");
        goto fail;
    }

    int64_t shape[2] = {t->shape[0], late->base->config.hidden_size};
    if (!tensor_has_supported_shape(late->dense_sf, t, "linear.weight", shape, 2, 1))
        goto fail;

    /* Same F32/BF16-borrow, F16-widen contract as the backbone weights. */
    late->projection = weight_ref_from_tensor(late->dense_sf, t);
    if (!late->projection.data)
        goto fail;
    late->token_dim = (int)t->shape[0];
    return late;

fail:
    ffwd_late_model_free(late);
    return NULL;
}

void ffwd_late_model_free(ffwd_late_model_t *model) {
    if (!model)
        return;
    ffwd_model_free(model->base);
    /* Free the projection before the mmap: only an F16-widened ref owns its
     * buffer; a borrowed F32/BF16 ref points into dense_sf and must not be freed. */
    if (model->projection.owned)
        free((void *)model->projection.data);
    safetensors_close(model->dense_sf);
    free(model);
}

const ffwd_config_t *ffwd_late_model_config(const ffwd_late_model_t *model) {
    return model && model->base ? &model->base->config : NULL;
}

int ffwd_late_model_token_dim(const ffwd_late_model_t *model) { return model ? model->token_dim : 0; }

/* Internal (internal.h):
 * expose the base model and projection so GPU backends can upload them to the
 * device without seeing struct ffwd_late_model. */
ffwd_model_t *ffwd_late_model_base(const ffwd_late_model_t *model) {
    return model ? model->base : NULL;
}

const ffwd_weight_ref_t *ffwd_late_model_projection(const ffwd_late_model_t *model) {
    return model ? &model->projection : NULL;
}

float ffwd_late_maxsim(
    const float *query_vectors, int query_tokens, const float *doc_vectors, int doc_tokens, int dim) {
    if (!query_vectors || !doc_vectors || query_tokens <= 0 || doc_tokens <= 0 || dim <= 0)
        return 0.0f;

    float score = 0.0f;
    for (int qi = 0; qi < query_tokens; qi++) {
        const float *q = query_vectors + (size_t)qi * dim;
        float best = -FLT_MAX;
        for (int di = 0; di < doc_tokens; di++) {
            const float *d = doc_vectors + (size_t)di * dim;
            float dot = dot_f32(q, d, dim);
            if (dot > best)
                best = dot;
        }
        score += best;
    }
    return score;
}

int ffwd_late_maxsim_batch(const float *query_vectors,
                           int query_tokens,
                           const float *doc_vectors,
                           const int *doc_offsets,
                           int docs,
                           int dim,
                           float *scores) {
    if (!query_vectors || !doc_vectors || !doc_offsets || !scores || query_tokens <= 0 || docs <= 0 ||
        dim <= 0 || doc_offsets[0] != 0)
        return -1;

    for (int i = 0; i < docs; i++) {
        if (doc_offsets[i] < 0 || doc_offsets[i + 1] <= doc_offsets[i])
            return -1;
    }

#ifdef USE_BLAS
    /* One similarity GEMM per group of whole candidates:
     * S[query_tokens, group_tokens] = Q @ D_group^T, then per-document
     * row-max plus sum. The group token budget bounds the scratch matrix;
     * one oversized document still runs alone in a group of its own. */
    enum { LATE_GROUP_TOKEN_BUDGET = 4096 };
    int total = doc_offsets[docs];
    int budget = LATE_GROUP_TOKEN_BUDGET;
    size_t cap = (size_t)query_tokens * (size_t)(total < budget ? total : budget);
    int largest = 0;
    for (int i = 0; i < docs; i++) {
        int len = doc_offsets[i + 1] - doc_offsets[i];
        if (len > largest)
            largest = len;
    }
    if ((size_t)query_tokens * (size_t)largest > cap)
        cap = (size_t)query_tokens * (size_t)largest;
    float *sim = (float *)malloc(cap * sizeof(float));
    if (sim) {
        int doc = 0;
        while (doc < docs) {
            int first = doc;
            int start = doc_offsets[first];
            doc++;
            while (doc < docs && doc_offsets[doc + 1] - start <= budget &&
                   doc_offsets[doc + 1] - start <= (int)(cap / (size_t)query_tokens))
                doc++;
            int group_tokens = doc_offsets[doc] - start;

            matmul_t(sim, query_vectors, doc_vectors + (size_t)start * dim, query_tokens, dim,
                     group_tokens);

            for (int i = first; i < doc; i++) {
                int s0 = doc_offsets[i] - start;
                int s1 = doc_offsets[i + 1] - start;
                float score = 0.0f;
                for (int qi = 0; qi < query_tokens; qi++) {
                    const float *row = sim + (size_t)qi * group_tokens;
                    float best = row[s0];
                    for (int t = s0 + 1; t < s1; t++)
                        if (row[t] > best)
                            best = row[t];
                    score += best;
                }
                scores[i] = score;
            }
        }
        free(sim);
        return 0;
    }
    /* Allocation failure: fall through to the scalar path. */
#endif

    for (int i = 0; i < docs; i++) {
        int start = doc_offsets[i];
        int end = doc_offsets[i + 1];
        scores[i] = ffwd_late_maxsim(query_vectors, query_tokens, doc_vectors + (size_t)start * dim,
                                     end - start, dim);
    }
    return 0;
}
