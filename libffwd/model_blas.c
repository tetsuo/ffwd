#include "ffwd.h"
#include "model_internal.h"
#include "kernels.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* CPU workspace ownership and scratch growth. GPU builds use their own
 * backend contexts and do not link this translation unit. */

ffwd_workspace_t *ffwd_workspace_new(const ffwd_model_t *model) {
    if (!model)
        return NULL;
    ffwd_workspace_t *ws = (ffwd_workspace_t *)calloc(1, sizeof(ffwd_workspace_t));
    if (!ws)
        return NULL;
    ws->model = model;
    return ws;
}

void ffwd_workspace_free(ffwd_workspace_t *ws) {
    if (!ws)
        return;
    free(ws->x);
    free(ws->x_norm);
    free(ws->q);
    free(ws->k);
    free(ws->v);
    free(ws->attn_out);
    free(ws->proj_out);
    free(ws->ffn_gate);
    free(ws->ffn_up);
    free(ws->attn_scores);
    free(ws->offsets);
    free(ws->rope_cos);
    free(ws->rope_sin);
    free(ws->bf16_widen);
    free(ws);
}

int ensure_buffers(ffwd_workspace_t *ws, const ffwd_config_t *c, int seq) {
    if (seq <= ws->buf_seq_cap)
        return 0;

    int cap;
    if (grow_cap(ws->buf_seq_cap, seq, &cap) != 0)
        return -1;

#define R(ptr, n)                                     \
    do {                                              \
        if (realloc_floats_2d(&(ptr), cap, (n)) != 0) \
            return -1;                                \
    } while (0)

    R(ws->x, c->hidden_size);
    R(ws->x_norm, c->hidden_size);
    R(ws->q, c->q_dim);
    R(ws->k, c->kv_dim);
    R(ws->v, c->kv_dim);
    R(ws->attn_out, c->q_dim);
    R(ws->proj_out, c->hidden_size);
    R(ws->ffn_gate, c->intermediate_size);
    R(ws->ffn_up, c->intermediate_size);

#undef R

    ws->buf_seq_cap = cap;
    return 0;
}

void ensure_attention_scores(ffwd_workspace_t *ws, const int *offsets, int batch) {
    size_t bytes = bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    if (bytes <= ws->attn_scores_bytes)
        return;

    void *p = realloc(ws->attn_scores, bytes);
    if (!p)
        return;
    ws->attn_scores = (float *)p;
    ws->attn_scores_bytes = bytes;
}

int ensure_rope_cache(ffwd_workspace_t *ws, const ffwd_config_t *cfg, int n_pos) {
    if (n_pos <= ws->rope_cache_cap)
        return 0;

    int cap = ws->rope_cache_cap > 0 ? ws->rope_cache_cap : 512;
    while (cap < n_pos) {
        if (cap > INT_MAX / 2)
            return -1;
        cap *= 2;
    }

    int head_dim = cfg->head_dim;
    int half = head_dim / 2;
    float theta = cfg->rope_theta;

    size_t count;
    if (mul_size((size_t)cap, (size_t)head_dim, &count) != 0)
        return -1;
    if (realloc_floats(&ws->rope_cos, count) != 0)
        return -1;
    if (realloc_floats(&ws->rope_sin, count) != 0)
        return -1;

    for (int pos = ws->rope_cache_cap; pos < cap; pos++) {
        float fp = (float)pos;
        float *c = ws->rope_cos + (size_t)pos * head_dim;
        float *s = ws->rope_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float inv = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = fp * inv;
            float cv = cosf(angle), sv = sinf(angle);
            c[d] = cv;
            c[half + d] = cv;
            s[d] = sv;
            s[half + d] = sv;
        }
    }

    ws->rope_cache_cap = cap;
    return 0;
}

int ensure_offsets(ffwd_workspace_t *ws, int batch) {
    if (!ws || batch <= 0 || batch == INT_MAX)
        return -1;

    int needed = batch + 1;
    if (needed <= ws->offsets_cap)
        return 0;

    if (realloc_ints(&ws->offsets, (size_t)needed) != 0)
        return -1;
    ws->offsets_cap = needed;
    return 0;
}

size_t ffwd_workspace_nbytes(const ffwd_workspace_t *ws) {
    if (!ws)
        return 0;

    size_t total = sizeof(*ws);
    const ffwd_model_t *model = ws->model;
    if (!model)
        return total;

    const ffwd_config_t *c = &model->config;
    size_t row_floats = 0;
    const int dims[] = {
        c->hidden_size,       /* x */
        c->hidden_size,       /* x_norm */
        c->q_dim,             /* q */
        c->kv_dim,            /* k */
        c->kv_dim,            /* v */
        c->q_dim,             /* attn_out */
        c->hidden_size,       /* proj_out */
        c->intermediate_size, /* ffn_gate */
        c->intermediate_size, /* ffn_up */
    };

    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        if (dims[i] < 0 || add_size(row_floats, (size_t)dims[i], &row_floats) != 0)
            return SIZE_MAX;
    }

    size_t buf_floats = 0;
    if (ws->buf_seq_cap > 0 && mul_size((size_t)ws->buf_seq_cap, row_floats, &buf_floats) != 0)
        return SIZE_MAX;

    size_t rope_floats = 0;
    if (ws->rope_cache_cap > 0) {
        size_t rope_rows = 0;
        if (mul_size((size_t)ws->rope_cache_cap, (size_t)c->head_dim, &rope_rows) != 0 ||
            mul_size(rope_rows, 2, &rope_floats) != 0)
            return SIZE_MAX;
    }

    size_t all_floats = 0, bytes = 0;
    if (add_size(buf_floats, rope_floats, &all_floats) != 0 ||
        mul_size(all_floats, sizeof(float), &bytes) != 0 || add_size(total, bytes, &total) != 0)
        return SIZE_MAX;

    size_t offset_bytes = 0;
    if (ws->offsets_cap > 0 && mul_size((size_t)ws->offsets_cap, sizeof(int), &offset_bytes) != 0)
        return SIZE_MAX;
    if (add_size(total, offset_bytes, &total) != 0)
        return SIZE_MAX;
    if (add_size(total, ws->attn_scores_bytes, &total) != 0)
        return SIZE_MAX;

    size_t widen_bytes = 0;
    if (ws->bf16_widen_count > 0 && mul_size(ws->bf16_widen_count, sizeof(float), &widen_bytes) != 0)
        return SIZE_MAX;
    if (add_size(total, widen_bytes, &total) != 0)
        return SIZE_MAX;

    return total;
}
