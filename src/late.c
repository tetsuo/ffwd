#include "embed.h"
#include "engine.h"
#include "kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>

/* ========================================================================
 * Late-interaction model
 *
 * A late-interaction model is a pooled base model plus a 1_Dense linear
 * projection from hidden states to token vectors. Instead of pooling to one
 * vector, it keeps one projected vector per kept token and scores query/doc
 * pairs with MaxSim. It reuses the base model's packed forward pass and
 * workspace; only the projection, the token-vector buffer, and the scoring
 * live here.
 * ======================================================================== */

struct embed_late_model {
    embed_model_t *base;
    safetensors_file_t *dense_sf;
    embed_weight_ref_t projection; /* [token_dim, hidden] */
    int token_dim;
};

struct embed_late_workspace {
    const embed_late_model_t *model;
    embed_workspace_t *base_ws;
    float *states;
    int states_seq_cap;
};

static int ensure_late_state_buffer(embed_late_workspace_t *ws, int n_tokens) {
    if (!ws || !ws->model || n_tokens <= 0)
        return -1;
    if (n_tokens <= ws->states_seq_cap)
        return 0;
    int hidden = ws->model->base->config.hidden_size;
    if (realloc_floats_2d(&ws->states, n_tokens, hidden) != 0)
        return -1;
    ws->states_seq_cap = n_tokens;
    return 0;
}

static const safetensor_t *find_single_safetensor(const safetensors_file_t *sf, const char *name) {
    if (!sf || !name)
        return NULL;
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0)
            return &sf->tensors[i];
    }
    return NULL;
}

embed_late_model_t *embed_late_model_load(const char *model_dir) {
    if (!model_dir_has_late_projection(model_dir)) {
        fprintf(stderr, "embed_late_model_load: missing 1_Dense projection\n");
        return NULL;
    }

    embed_late_model_t *late = (embed_late_model_t *)calloc(1, sizeof(*late));
    if (!late)
        return NULL;

    late->base = model_load_range_ex(model_dir, 0, -1, 1);
    if (!late->base)
        goto fail;

    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/1_Dense/model.safetensors", model_dir);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "embed_late_model_load: model path too long\n");
        goto fail;
    }

    late->dense_sf = safetensors_open(path);
    if (!late->dense_sf) {
        fprintf(stderr, "embed_late_model_load: cannot open %s\n", path);
        goto fail;
    }

    const safetensor_t *t = find_single_safetensor(late->dense_sf, "linear.weight");
    if (!t) {
        fprintf(stderr, "embed_late_model_load: missing linear.weight\n");
        goto fail;
    }
    if (t->ndim != 2 || t->shape[0] <= 0 || t->shape[0] > INT_MAX) {
        fprintf(stderr, "embed_late_model_load: bad projection shape\n");
        goto fail;
    }

    int64_t shape[2] = {t->shape[0], late->base->config.hidden_size};
    if (!tensor_has_supported_shape(late->dense_sf, t, "linear.weight", shape, 2, 1))
        goto fail;

    late->projection.dtype = t->dtype;
    late->projection.data = safetensors_data(late->dense_sf, t);
    late->token_dim = (int)t->shape[0];
    return late;

fail:
    embed_late_model_free(late);
    return NULL;
}

void embed_late_model_free(embed_late_model_t *model) {
    if (!model)
        return;
    embed_model_free(model->base);
    safetensors_close(model->dense_sf);
    free(model);
}

embed_late_workspace_t *embed_late_workspace_new(const embed_late_model_t *model) {
    if (!model || !model->base)
        return NULL;
    embed_late_workspace_t *ws = (embed_late_workspace_t *)calloc(1, sizeof(*ws));
    if (!ws)
        return NULL;
    ws->model = model;
    ws->base_ws = embed_workspace_new(model->base);
    if (!ws->base_ws) {
        free(ws);
        return NULL;
    }
    return ws;
}

void embed_late_workspace_free(embed_late_workspace_t *ws) {
    if (!ws)
        return;
    embed_workspace_free(ws->base_ws);
    free(ws->states);
    free(ws);
}

const embed_config_t *embed_late_model_config(const embed_late_model_t *model) {
    return model && model->base ? &model->base->config : NULL;
}

int embed_late_model_token_dim(const embed_late_model_t *model) {
    return model ? model->token_dim : 0;
}

/* Internal (embed_internal.h): expose the base model and projection so GPU
 * backends can upload them to device without seeing struct embed_late_model. */
embed_model_t *embed_late_model_base(const embed_late_model_t *model) {
    return model ? model->base : NULL;
}

const embed_weight_ref_t *embed_late_model_projection(const embed_late_model_t *model) {
    return model ? &model->projection : NULL;
}

int embed_late_model_encode_tokens(const embed_late_model_t *model,
                                   embed_late_workspace_t *ws,
                                   const int *token_ids,
                                   int n_tokens,
                                   int normalize,
                                   float *out_vectors) {
    if (!model || !ws || ws->model != model || !token_ids || n_tokens <= 0 || !out_vectors ||
        model->token_dim <= 0)
        return -1;

    int hidden = model->base->config.hidden_size;
    int dim = model->token_dim;
    if (ensure_late_state_buffer(ws, n_tokens) != 0)
        return -1;
    if (embed_model_forward_into(model->base, ws->base_ws, token_ids, n_tokens, ws->states) != 0)
        return -1;

    linear_nobias_weight(ws->base_ws, out_vectors, ws->states, &model->projection, n_tokens, hidden,
                         dim);
    if (normalize) {
        for (int i = 0; i < n_tokens; i++) {
            if (embed_l2_normalize(out_vectors + (size_t)i * dim, dim) != 0)
                return -1;
        }
    }
    return 0;
}

float embed_late_maxsim(const float *query_vectors,
                        int query_tokens,
                        const float *doc_vectors,
                        int doc_tokens,
                        int dim) {
    if (!query_vectors || !doc_vectors || query_tokens <= 0 || doc_tokens <= 0 || dim <= 0)
        return 0.0f;

    float score = 0.0f;
    for (int qi = 0; qi < query_tokens; qi++) {
        const float *q = query_vectors + (size_t)qi * dim;
        float best = -FLT_MAX;
        for (int di = 0; di < doc_tokens; di++) {
            const float *d = doc_vectors + (size_t)di * dim;
            float dot = embed_dot_f32(q, d, dim);
            if (dot > best)
                best = dot;
        }
        score += best;
    }
    return score;
}

int embed_late_maxsim_batch(const float *query_vectors,
                            int query_tokens,
                            const float *doc_vectors,
                            const int *doc_offsets,
                            int docs,
                            int dim,
                            float *scores) {
    if (!query_vectors || !doc_vectors || !doc_offsets || !scores || query_tokens <= 0 ||
        docs <= 0 || dim <= 0 || doc_offsets[0] != 0)
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

            embed_matmul_t(sim, query_vectors, doc_vectors + (size_t)start * dim, query_tokens, dim,
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
        scores[i] = embed_late_maxsim(query_vectors, query_tokens,
                                      doc_vectors + (size_t)start * dim, end - start, dim);
    }
    return 0;
}

/* Late-interaction batch encoder. Lives here, beside the per-document encoder,
 * and reuses the base model's packed/ragged batch forward pass. */
int embed_late_model_encode_docs(const embed_late_model_t *model,
                                 embed_late_workspace_t *ws,
                                 const int *const *doc_ids,
                                 const int *n_tokens,
                                 const int *const *keep,
                                 const int *n_keep,
                                 int n_docs,
                                 int normalize,
                                 float *out_vectors,
                                 int *out_offsets) {
    if (!model || !ws || ws->model != model || !doc_ids || !n_tokens || !keep || !n_keep ||
        n_docs <= 0 || !out_vectors || !out_offsets || model->token_dim <= 0)
        return -1;

    int hidden = model->base->config.hidden_size;
    int dim = model->token_dim;

    /* Pack all candidates for one block-diagonal forward. inputs/offsets index
     * the full token stream; out_offsets is the prefix sum of kept tokens. */
    embed_input_t *inputs = (embed_input_t *)malloc((size_t)n_docs * sizeof(*inputs));
    int *offsets = (int *)malloc((size_t)(n_docs + 1) * sizeof(*offsets));
    if (!inputs || !offsets) {
        free(inputs);
        free(offsets);
        return -1;
    }

    int total_keep = 0;
    out_offsets[0] = 0;
    for (int d = 0; d < n_docs; d++) {
        if (!doc_ids[d] || n_tokens[d] <= 0 || !keep[d] || n_keep[d] <= 0 ||
            n_keep[d] > n_tokens[d] || total_keep > INT_MAX - n_keep[d]) {
            free(inputs);
            free(offsets);
            return -1;
        }
        for (int k = 0; k < n_keep[d]; k++) {
            if (keep[d][k] < 0 || keep[d][k] >= n_tokens[d]) {
                free(inputs);
                free(offsets);
                return -1;
            }
        }
        inputs[d].ids = doc_ids[d];
        inputs[d].n_tokens = n_tokens[d];
        total_keep += n_keep[d];
        out_offsets[d + 1] = total_keep;
    }

    int total_seq = 0, max_seq = 0;
    int rc = build_offsets(inputs, n_docs, 1, offsets, &total_seq, &max_seq);
    if (rc == 0)
        rc = forward_packed_inplace(model->base, ws->base_ws, inputs, n_docs, offsets, total_seq,
                                    max_seq, 1);
    free(inputs);
    if (rc != 0) {
        free(offsets);
        return -1;
    }

    /* Gather each document's kept hidden rows into ws->states (its natural
     * [seq, hidden] shape), then project the packed [total_keep, hidden] block
     * to [total_keep, dim] in one GEMM, matching the per-document path. */
    if (ensure_late_state_buffer(ws, total_keep) != 0) {
        free(offsets);
        return -1;
    }
    const float *states = ws->base_ws->x;
    int pos = 0;
    for (int d = 0; d < n_docs; d++) {
        int start = offsets[d];
        for (int k = 0; k < n_keep[d]; k++) {
            memcpy(ws->states + (size_t)pos * hidden,
                   states + (size_t)(start + keep[d][k]) * hidden, (size_t)hidden * sizeof(float));
            pos++;
        }
    }
    free(offsets);

    linear_nobias_weight(ws->base_ws, out_vectors, ws->states, &model->projection, total_keep,
                         hidden, dim);
    if (normalize) {
        for (int i = 0; i < total_keep; i++) {
            if (embed_l2_normalize(out_vectors + (size_t)i * dim, dim) != 0)
                return -1;
        }
    }
    return 0;
}
