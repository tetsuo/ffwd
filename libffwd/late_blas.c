#include "ffwd.h"
#include "model_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* CPU late-interaction encoding. The late model loader and MaxSim scoring live
 * in late.c because CUDA reuses those without linking the CPU transformer. */

struct ffwd_late_workspace {
    const ffwd_late_model_t *model;
    ffwd_workspace_t *base_ws;
    float *states;
    int states_seq_cap;
};

static int ensure_late_state_buffer(ffwd_late_workspace_t *ws, int n_tokens) {
    if (!ws || !ws->model || n_tokens <= 0)
        return -1;
    if (n_tokens <= ws->states_seq_cap)
        return 0;
    ffwd_model_t *base = ffwd_late_model_base(ws->model);
    if (!base)
        return -1;
    int hidden = base->config.hidden_size;
    if (realloc_floats_2d(&ws->states, n_tokens, hidden) != 0)
        return -1;
    ws->states_seq_cap = n_tokens;
    return 0;
}

ffwd_late_workspace_t *ffwd_late_workspace_new(const ffwd_late_model_t *model) {
    ffwd_model_t *base = ffwd_late_model_base(model);
    if (!model || !base)
        return NULL;
    ffwd_late_workspace_t *ws = (ffwd_late_workspace_t *)calloc(1, sizeof(*ws));
    if (!ws)
        return NULL;
    ws->model = model;
    ws->base_ws = ffwd_workspace_new(base);
    if (!ws->base_ws) {
        free(ws);
        return NULL;
    }
    return ws;
}

void ffwd_late_workspace_free(ffwd_late_workspace_t *ws) {
    if (!ws)
        return;
    ffwd_workspace_free(ws->base_ws);
    free(ws->states);
    free(ws);
}

int ffwd_late_model_encode_tokens(const ffwd_late_model_t *model,
                                     ffwd_late_workspace_t *ws,
                                     const int *token_ids,
                                     int n_tokens,
                                     int normalize,
                                     float *out_vectors) {
    int dim = ffwd_late_model_token_dim(model);
    ffwd_model_t *base = ffwd_late_model_base(model);
    const ffwd_weight_ref_t *projection = ffwd_late_model_projection(model);
    if (!model || !ws || ws->model != model || !base || !projection || !projection->data ||
        !token_ids || n_tokens <= 0 || !out_vectors || dim <= 0)
        return -1;

    int hidden = base->config.hidden_size;
    if (ensure_late_state_buffer(ws, n_tokens) != 0)
        return -1;
    if (ffwd_model_forward_into(base, ws->base_ws, token_ids, n_tokens, ws->states) != 0)
        return -1;

    linear_nobias_weight(ws->base_ws, out_vectors, ws->states, projection, n_tokens, hidden, dim);
    if (normalize) {
        for (int i = 0; i < n_tokens; i++) {
            if (ffwd_l2_normalize(out_vectors + (size_t)i * dim, dim) != 0)
                return -1;
        }
    }
    return 0;
}

/* Late-interaction batch encoder. Reuses the base model's packed/ragged batch
 * forward pass, then gathers kept rows and projects them in one GEMM. */
int ffwd_late_model_encode_docs(const ffwd_late_model_t *model,
                                   ffwd_late_workspace_t *ws,
                                   const int *const *doc_ids,
                                   const int *n_tokens,
                                   const int *const *keep,
                                   const int *n_keep,
                                   int n_docs,
                                   int normalize,
                                   float *out_vectors,
                                   int *out_offsets) {
    int dim = ffwd_late_model_token_dim(model);
    ffwd_model_t *base = ffwd_late_model_base(model);
    const ffwd_weight_ref_t *projection = ffwd_late_model_projection(model);
    if (!model || !ws || ws->model != model || !base || !projection || !projection->data ||
        !doc_ids || !n_tokens || !keep || !n_keep || n_docs <= 0 || !out_vectors || !out_offsets ||
        dim <= 0)
        return -1;

    int hidden = base->config.hidden_size;

    ffwd_input_t *inputs = (ffwd_input_t *)malloc((size_t)n_docs * sizeof(*inputs));
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
        rc =
            forward_packed_inplace(base, ws->base_ws, inputs, n_docs, offsets, total_seq, max_seq, 1);
    free(inputs);
    if (rc != 0) {
        free(offsets);
        return -1;
    }

    if (ensure_late_state_buffer(ws, total_keep) != 0) {
        free(offsets);
        return -1;
    }
    const float *states = ws->base_ws->x;
    int pos = 0;
    for (int d = 0; d < n_docs; d++) {
        int start = offsets[d];
        for (int k = 0; k < n_keep[d]; k++) {
            memcpy(ws->states + (size_t)pos * hidden, states + (size_t)(start + keep[d][k]) * hidden,
                   (size_t)hidden * sizeof(float));
            pos++;
        }
    }
    free(offsets);

    linear_nobias_weight(ws->base_ws, out_vectors, ws->states, projection, total_keep, hidden, dim);
    if (normalize) {
        for (int i = 0; i < total_keep; i++) {
            if (ffwd_l2_normalize(out_vectors + (size_t)i * dim, dim) != 0)
                return -1;
        }
    }
    return 0;
}
