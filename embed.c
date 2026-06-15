#include "embed.h"
#include "embed_internal.h"
#include "qwen_kernels.h"
#include "qwen_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>

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

/* ========================================================================
 * Globals
 * ======================================================================== */

int embed_verbose = 0;
int qwen_verbose = 0;

#define EMBED_BF16_QKV_FUSE_MAX_SEQ     16
#define EMBED_BF16_PAIR_FUSE_MAX_SEQ    4
#define EMBED_MIN_WORKSPACE_SEQ_CAP     16
#define EMBED_WORKSPACE_SEQ_GRANULARITY 16

/* ========================================================================
 * Size helpers
 * ======================================================================== */

static int mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a)
        return -1;
    *out = a * b;
    return 0;
}

static int add_size(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a)
        return -1;
    *out = a + b;
    return 0;
}

static int grow_cap(int current, int needed, int *out) {
    if (needed <= 0)
        return -1;

    int cap = needed;
    if (cap < EMBED_MIN_WORKSPACE_SEQ_CAP)
        cap = EMBED_MIN_WORKSPACE_SEQ_CAP;

    int rem = cap % EMBED_WORKSPACE_SEQ_GRANULARITY;
    if (rem != 0) {
        int add = EMBED_WORKSPACE_SEQ_GRANULARITY - rem;
        if (cap > INT_MAX - add)
            return -1;
        cap += add;
    }

    if (cap < current)
        cap = current;

    *out = cap;
    return 0;
}

static int realloc_floats(float **ptr, size_t count) {
    size_t bytes;
    if (mul_size(count, sizeof(float), &bytes) != 0)
        return -1;

    void *p = realloc(*ptr, bytes);
    if (!p && bytes != 0)
        return -1;
    *ptr = (float *)p;
    return 0;
}

static float *malloc_floats(size_t count) {
    size_t bytes;
    if (mul_size(count, sizeof(float), &bytes) != 0)
        return NULL;
    return (float *)malloc(bytes);
}

static int model_dir_has_file(const char *model_dir, const char *rel) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", model_dir, rel);
    if (n < 0 || (size_t)n >= sizeof(path))
        return 0;
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    fclose(f);
    return 1;
}

static int model_dir_has_late_projection(const char *model_dir) {
    return model_dir_has_file(model_dir, "1_Dense/config.json") &&
           model_dir_has_file(model_dir, "1_Dense/model.safetensors");
}

static int realloc_floats_2d(float **ptr, int rows, int cols) {
    if (rows < 0 || cols < 0)
        return -1;

    size_t count;
    if (mul_size((size_t)rows, (size_t)cols, &count) != 0)
        return -1;
    return realloc_floats(ptr, count);
}

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

static int realloc_ints(int **ptr, size_t count) {
    size_t bytes;
    if (mul_size(count, sizeof(int), &bytes) != 0)
        return -1;

    void *p = realloc(*ptr, bytes);
    if (!p && bytes != 0)
        return -1;
    *ptr = (int *)p;
    return 0;
}

/* ========================================================================
 * Minimal JSON helpers for config.json parsing
 * ======================================================================== */

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
        p++;
    return p;
}

/* Find "key": <value> in a flat JSON object. Returns pointer to value start. */
static const char *json_find_key(const char *json, const char *key) {
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strstr(p, key)) != NULL) {
        /* Check it's actually a quoted key: "key" */
        if (p > json && *(p - 1) == '"') {
            const char *after = p + klen;
            if (*after == '"') {
                after = skip_ws(after + 1);
                if (*after == ':')
                    return skip_ws(after + 1);
            }
        }
        p += klen;
    }
    return NULL;
}

static int json_get_int(const char *json, const char *key, int fallback) {
    const char *v = json_find_key(json, key);
    if (!v)
        return fallback;
    return atoi(v);
}

static double json_get_double(const char *json, const char *key, double fallback) {
    const char *v = json_find_key(json, key);
    if (!v)
        return fallback;
    return atof(v);
}

static int json_string_equals(const char *json, const char *key, const char *expected) {
    const char *v = json_find_key(json, key);
    if (!v || *v != '"')
        return 0;
    v++;
    size_t len = strlen(expected);
    return strncmp(v, expected, len) == 0 && v[len] == '"';
}

static int parse_config(embed_config_t *cfg, const char *model_dir) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/config.json", model_dir);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "embed_model_load: model path too long\n");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "embed_model_load: cannot open %s\n", path);
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long sz = ftell(f);
    if (sz < 0 || (size_t)sz == SIZE_MAX || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        return -1;
    }
    fclose(f);
    buf[sz] = '\0';

    cfg->hidden_size = json_get_int(buf, "hidden_size", 0);
    cfg->n_layers = json_get_int(buf, "num_hidden_layers", 0);
    cfg->n_heads = json_get_int(buf, "num_attention_heads", 0);
    cfg->n_kv_heads = json_get_int(buf, "num_key_value_heads", 0);
    cfg->head_dim = json_get_int(buf, "head_dim", EMBED_HEAD_DIM);
    cfg->intermediate_size = json_get_int(buf, "intermediate_size", 0);
    cfg->vocab_size = json_get_int(buf, "vocab_size", EMBED_VOCAB_SIZE);
    cfg->rms_norm_eps = (float)json_get_double(buf, "rms_norm_eps", 1e-6);
    cfg->rope_theta = (float)json_get_double(buf, "rope_theta", 1000000.0);
    cfg->attention_mode = EMBED_ATTENTION_BIDIRECTIONAL;
    cfg->pooling_mode = EMBED_POOL_MEAN;
    cfg->normalize_embeddings = 0;
    cfg->append_terminal_token = 0;

    if (json_string_equals(buf, "model_type", "qwen3")) {
        cfg->attention_mode = EMBED_ATTENTION_CAUSAL;
        cfg->pooling_mode = EMBED_POOL_LAST_TOKEN;
        cfg->normalize_embeddings = 1;
        cfg->append_terminal_token = 1;
    }

    cfg->q_dim = 0;
    cfg->kv_dim = 0;
    if (cfg->n_heads > 0 && cfg->head_dim > 0 && cfg->n_heads <= INT_MAX / cfg->head_dim)
        cfg->q_dim = cfg->n_heads * cfg->head_dim;
    if (cfg->n_kv_heads > 0 && cfg->head_dim > 0 && cfg->n_kv_heads <= INT_MAX / cfg->head_dim)
        cfg->kv_dim = cfg->n_kv_heads * cfg->head_dim;

    free(buf);

    /* Sanity checks */
    if (cfg->hidden_size <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0 || cfg->n_kv_heads <= 0 ||
        cfg->head_dim <= 0 || cfg->intermediate_size <= 0 || cfg->vocab_size <= 0 ||
        cfg->q_dim <= 0 || cfg->kv_dim <= 0 || (cfg->head_dim & 1) ||
        cfg->n_heads % cfg->n_kv_heads != 0 || !isfinite(cfg->rms_norm_eps) ||
        cfg->rms_norm_eps <= 0.0f || !isfinite(cfg->rope_theta) || cfg->rope_theta <= 0.0f) {
        fprintf(stderr,
                "embed_model_load: invalid config in %s "
                "(hidden=%d, layers=%d, heads=%d/%d, head_dim=%d, inter=%d)\n",
                path, cfg->hidden_size, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim,
                cfg->intermediate_size);
        return -1;
    }
    if (cfg->n_layers > EMBED_MAX_LAYERS) {
        fprintf(stderr, "embed_model_load: too many layers (%d > %d)\n", cfg->n_layers,
                EMBED_MAX_LAYERS);
        return -1;
    }

    if (embed_verbose >= 1)
        fprintf(stderr,
                "config: hidden=%d, layers=%d, heads=%d/%d, "
                "inter=%d, head_dim=%d, attention=%s, pooling=%s\n",
                cfg->hidden_size, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
                cfg->intermediate_size, cfg->head_dim,
                cfg->attention_mode == EMBED_ATTENTION_CAUSAL ? "causal" : "bidirectional",
                cfg->pooling_mode == EMBED_POOL_LAST_TOKEN ? "last-token" : "mean");

    return 0;
}

/* ========================================================================
 * Weight loading (direct mmap pointers into safetensors)
 * ======================================================================== */

static size_t dtype_size(safetensor_dtype_t dtype) {
    switch (dtype) {
    case DTYPE_F32:
        return sizeof(float);
    case DTYPE_BF16:
        return sizeof(uint16_t);
    default:
        return 0;
    }
}

static const char *dtype_name(safetensor_dtype_t dtype) {
    switch (dtype) {
    case DTYPE_F32:
        return "F32";
    case DTYPE_F16:
        return "F16";
    case DTYPE_BF16:
        return "BF16";
    case DTYPE_I32:
        return "I32";
    case DTYPE_I64:
        return "I64";
    case DTYPE_BOOL:
        return "BOOL";
    default:
        return "UNKNOWN";
    }
}

static int tensor_has_supported_shape(const safetensors_file_t *sf,
                                      const safetensor_t *t,
                                      const char *name,
                                      const int64_t *shape,
                                      int ndim,
                                      int bf16_ok) {
    if (t->dtype != DTYPE_F32 && !(bf16_ok && t->dtype == DTYPE_BF16)) {
        fprintf(stderr, "embed: unsupported dtype for %s: got %s, expected %s\n", name,
                dtype_name(t->dtype), bf16_ok ? "F32 or BF16" : "F32");
        return 0;
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "embed: bad rank for %s: got %d, expected %d\n", name, t->ndim, ndim);
        return 0;
    }

    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (t->shape[i] != shape[i]) {
            fprintf(stderr,
                    "embed: bad shape for %s at dim %d: "
                    "got %lld, expected %lld\n",
                    name, i, (long long)t->shape[i], (long long)shape[i]);
            return 0;
        }
        if (shape[i] < 0 || (uint64_t)shape[i] > SIZE_MAX / numel) {
            fprintf(stderr, "embed: shape too large for %s\n", name);
            return 0;
        }
        numel *= (size_t)shape[i];
    }

    size_t elem_size = dtype_size(t->dtype);
    if (elem_size == 0) {
        fprintf(stderr, "embed: unsupported dtype for %s: got %s\n", name, dtype_name(t->dtype));
        return 0;
    }

    size_t bytes;
    if (mul_size(numel, elem_size, &bytes) != 0) {
        fprintf(stderr, "embed: tensor data too large for %s\n", name);
        return 0;
    }
    if (t->data_size != bytes) {
        fprintf(stderr, "embed: bad data size for %s: got %zu, expected %zu\n", name, t->data_size,
                bytes);
        return 0;
    }

    size_t data_start = 8 + sf->header_size;
    if (data_start > sf->file_size || t->data_offset > sf->file_size - data_start ||
        t->data_size > sf->file_size - data_start - t->data_offset) {
        fprintf(stderr, "embed: tensor data out of bounds for %s\n", name);
        return 0;
    }
    return 1;
}

static embed_weight_ref_t
load_weight_direct(multi_safetensors_t *ms, const char *name, const int64_t *shape, int ndim) {
    embed_weight_ref_t ref = {DTYPE_UNKNOWN, NULL};
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "embed: tensor not found: %s\n", name);
        return ref;
    }
    if (!tensor_has_supported_shape(sf, t, name, shape, ndim, 1))
        return ref;

    ref.dtype = t->dtype;
    ref.data = safetensors_data(sf, t);
    return ref;
}

static float *
load_norm_f32(multi_safetensors_t *ms, const char *name, const int64_t *shape, int ndim) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "embed: tensor not found: %s\n", name);
        return NULL;
    }
    if (!tensor_has_supported_shape(sf, t, name, shape, ndim, 1))
        return NULL;
    return safetensors_get_f32(sf, t);
}

static void bf16_row_to_f32(float *dst, const uint16_t *src, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t u = ((uint32_t)src[i]) << 16;
        memcpy(dst + i, &u, sizeof(float));
    }
}

static void copy_weight_row(float *dst, const embed_weight_ref_t *w, size_t row, int row_width) {
    if (w->dtype == DTYPE_F32) {
        const float *src = (const float *)w->data + row * (size_t)row_width;
        memcpy(dst, src, (size_t)row_width * sizeof(float));
    } else {
        const uint16_t *src = (const uint16_t *)w->data + row * (size_t)row_width;
        bf16_row_to_f32(dst, src, row_width);
    }
}

static int ensure_bf16_widen(embed_workspace_t *ws, size_t count) {
    if (ws->bf16_widen_count >= count)
        return 0;
    float *p = (float *)realloc(ws->bf16_widen, count * sizeof(float));
    if (!p)
        return -1;
    ws->bf16_widen = p;
    ws->bf16_widen_count = count;
    return 0;
}

static void linear_nobias_weight(embed_workspace_t *ws,
                                 float *y,
                                 const float *x,
                                 const embed_weight_ref_t *w,
                                 int seq_len,
                                 int in_dim,
                                 int out_dim) {
    if (w->dtype == DTYPE_F32) {
        qwen_linear_nobias(y, x, (const float *)w->data, seq_len, in_dim, out_dim);
        return;
    }
    if (seq_len > 16) {
        /* Widen the weight matrix once into workspace scratch and use the
         * BLAS-backed F32 path; below that the fused BF16 row kernels win. */
        size_t count = (size_t)out_dim * (size_t)in_dim;
        if (ensure_bf16_widen(ws, count) == 0) {
            qwen_bf16_to_f32_buf(ws->bf16_widen, (const uint16_t *)w->data, count);
            qwen_linear_nobias(y, x, ws->bf16_widen, seq_len, in_dim, out_dim);
            return;
        }
    }
    qwen_linear_nobias_bf16(y, x, (const uint16_t *)w->data, seq_len, in_dim, out_dim);
}

static void linear_qkv_weight(embed_workspace_t *ws,
                              float *q,
                              float *k,
                              float *v,
                              const float *x,
                              const embed_weight_ref_t *wq,
                              const embed_weight_ref_t *wk,
                              const embed_weight_ref_t *wv,
                              int seq_len,
                              int in_dim,
                              int q_dim,
                              int kv_dim) {
    if (wq->dtype == DTYPE_BF16 && wk->dtype == DTYPE_BF16 && wv->dtype == DTYPE_BF16 &&
        seq_len <= EMBED_BF16_QKV_FUSE_MAX_SEQ) {
        qwen_linear_nobias_bf16_qkv(q, k, v, x, (const uint16_t *)wq->data,
                                    (const uint16_t *)wk->data, (const uint16_t *)wv->data, seq_len,
                                    in_dim, q_dim, kv_dim);
        return;
    }

    linear_nobias_weight(ws, q, x, wq, seq_len, in_dim, q_dim);
    linear_nobias_weight(ws, k, x, wk, seq_len, in_dim, kv_dim);
    linear_nobias_weight(ws, v, x, wv, seq_len, in_dim, kv_dim);
}

static void linear_pair_weight(embed_workspace_t *ws,
                               float *a,
                               float *b,
                               const float *x,
                               const embed_weight_ref_t *wa,
                               const embed_weight_ref_t *wb,
                               int seq_len,
                               int in_dim,
                               int a_dim,
                               int b_dim) {
    if (wa->dtype == DTYPE_BF16 && wb->dtype == DTYPE_BF16 &&
        seq_len <= EMBED_BF16_PAIR_FUSE_MAX_SEQ) {
        qwen_linear_nobias_bf16_pair(a, b, x, (const uint16_t *)wa->data,
                                     (const uint16_t *)wb->data, seq_len, in_dim, a_dim, b_dim);
        return;
    }

    linear_nobias_weight(ws, a, x, wa, seq_len, in_dim, a_dim);
    linear_nobias_weight(ws, b, x, wb, seq_len, in_dim, b_dim);
}

/* ========================================================================
 * Working buffer management
 * ======================================================================== */

static int ensure_buffers(embed_workspace_t *ws, const embed_config_t *c, int seq) {
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

static void ensure_attention_scores(embed_workspace_t *ws, const int *offsets, int batch) {
    size_t bytes = qwen_bidirectional_gqa_attention_packed_scratch_bytes(offsets, batch);
    if (bytes <= ws->attn_scores_bytes)
        return;

    void *p = realloc(ws->attn_scores, bytes);
    if (!p)
        return;
    ws->attn_scores = (float *)p;
    ws->attn_scores_bytes = bytes;
}

/* ========================================================================
 * RoPE cache
 * ======================================================================== */

static int ensure_rope_cache(embed_workspace_t *ws, const embed_config_t *cfg, int n_pos) {
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

static int ensure_offsets(embed_workspace_t *ws, int batch) {
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

/* ========================================================================
 * Model / workspace load and free
 * ======================================================================== */

static embed_model_t *
model_load_range_ex(const char *model_dir, int layer_start, int layer_end, int allow_late) {
    if (!allow_late && model_dir_has_late_projection(model_dir)) {
        fprintf(stderr, "embed_model_load: late-interaction models require "
                        "token-level MaxSim support and are not valid pooled "
                        "embedding models yet\n");
        return NULL;
    }

    embed_model_t *model = (embed_model_t *)calloc(1, sizeof(embed_model_t));
    if (!model)
        return NULL;

    /* Parse config.json */
    if (parse_config(&model->config, model_dir) != 0) {
        free(model);
        return NULL;
    }

    const embed_config_t *cfg = &model->config;
    if (layer_end == -1)
        layer_end = cfg->n_layers;
    if (layer_start < 0 || layer_start >= layer_end || layer_end > cfg->n_layers) {
        fprintf(stderr,
                "embed_model_load: invalid layer range [%d, %d) "
                "for %d-layer model\n",
                layer_start, layer_end, cfg->n_layers);
        free(model);
        return NULL;
    }
    model->layer_start = layer_start;
    model->layer_end = layer_end;
    model->attention = cfg->attention_mode == EMBED_ATTENTION_CAUSAL
                           ? qwen_causal_gqa_attention_packed_with_scratch
                           : qwen_bidirectional_gqa_attention_packed_with_scratch;

    /* Open safetensors (handles single file or multi-shard) */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "embed_model_load: failed to open safetensors in %s\n", model_dir);
        free(model);
        return NULL;
    }
    model->safetensors = ms;

    if (embed_verbose >= 1)
        fprintf(stderr, "embed_model_load: loading weights from %s\n", model_dir);

    /* Only the first stage needs token embeddings. */
    embed_weights_t *w = &model->weights;
    if (layer_start == 0) {
        const int64_t embed_shape[2] = {cfg->vocab_size, cfg->hidden_size};
        w->embed_tokens = load_weight_direct(ms, "embed_tokens.weight", embed_shape, 2);
        if (!w->embed_tokens.data)
            goto fail;
    }

    /* Allocate layer array */
    w->layers = (embed_layer_t *)calloc(cfg->n_layers, sizeof(embed_layer_t));
    if (!w->layers)
        goto fail;

    /* Per-layer weights */
    char name[256];
    for (int i = layer_start; i < layer_end; i++) {
        embed_layer_t *l = &w->layers[i];

#define LOAD_NORM(field, fmt, d0)                      \
    do {                                               \
        const int64_t expect[1] = {(d0)};              \
        snprintf(name, sizeof(name), fmt, i);          \
        l->field = load_norm_f32(ms, name, expect, 1); \
        if (!l->field)                                 \
            goto fail;                                 \
    } while (0)

#define LOAD2(field, fmt, d0, d1)                           \
    do {                                                    \
        const int64_t expect[2] = {(d0), (d1)};             \
        snprintf(name, sizeof(name), fmt, i);               \
        l->field = load_weight_direct(ms, name, expect, 2); \
        if (!l->field.data)                                 \
            goto fail;                                      \
    } while (0)

        LOAD2(wq, "layers.%d.self_attn.q_proj.weight", cfg->q_dim, cfg->hidden_size);
        LOAD2(wk, "layers.%d.self_attn.k_proj.weight", cfg->kv_dim, cfg->hidden_size);
        LOAD2(wv, "layers.%d.self_attn.v_proj.weight", cfg->kv_dim, cfg->hidden_size);
        LOAD2(wo, "layers.%d.self_attn.o_proj.weight", cfg->hidden_size, cfg->q_dim);
        LOAD_NORM(q_norm, "layers.%d.self_attn.q_norm.weight", cfg->head_dim);
        LOAD_NORM(k_norm, "layers.%d.self_attn.k_norm.weight", cfg->head_dim);
        LOAD_NORM(input_norm, "layers.%d.input_layernorm.weight", cfg->hidden_size);
        LOAD_NORM(post_attn_norm, "layers.%d.post_attention_layernorm.weight", cfg->hidden_size);
        LOAD2(gate_proj, "layers.%d.mlp.gate_proj.weight", cfg->intermediate_size,
              cfg->hidden_size);
        LOAD2(up_proj, "layers.%d.mlp.up_proj.weight", cfg->intermediate_size, cfg->hidden_size);
        LOAD2(down_proj, "layers.%d.mlp.down_proj.weight", cfg->hidden_size,
              cfg->intermediate_size);

#undef LOAD2
#undef LOAD_NORM

        if (embed_verbose >= 2)
            fprintf(stderr, "  layer %d loaded\n", i);
    }

    /* Only the final stage applies the output RMSNorm. */
    if (layer_end == cfg->n_layers) {
        const int64_t norm_shape[1] = {cfg->hidden_size};
        w->norm = load_norm_f32(ms, "norm.weight", norm_shape, 1);
        if (!w->norm)
            goto fail;
    }

    if (embed_verbose >= 1)
        fprintf(stderr,
                "embed_model_load: layers [%d, %d) loaded "
                "(%d-dim embeddings)\n",
                layer_start, layer_end, cfg->hidden_size);

    return model;

fail:
    embed_model_free(model);
    return NULL;
}

embed_model_t *embed_model_load(const char *model_dir) {
    return model_load_range_ex(model_dir, 0, -1, 0);
}

void embed_model_free(embed_model_t *model) {
    if (!model)
        return;
    if (model->weights.layers) {
        for (int i = 0; i < model->config.n_layers; i++) {
            embed_layer_t *l = &model->weights.layers[i];
            free((void *)l->q_norm);
            free((void *)l->k_norm);
            free((void *)l->input_norm);
            free((void *)l->post_attn_norm);
        }
    }
    free((void *)model->weights.norm);
    if (model->safetensors)
        multi_safetensors_close((multi_safetensors_t *)model->safetensors);
    free(model->weights.layers);
    free(model);
}

embed_workspace_t *embed_workspace_new(const embed_model_t *model) {
    if (!model)
        return NULL;
    embed_workspace_t *ws = (embed_workspace_t *)calloc(1, sizeof(embed_workspace_t));
    if (!ws)
        return NULL;
    ws->model = model;
    return ws;
}

void embed_workspace_free(embed_workspace_t *ws) {
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

const embed_config_t *embed_model_config(const embed_model_t *model) {
    return model ? &model->config : NULL;
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
            float dot = qwen_dot_f32(q, d, dim);
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

            qwen_matmul_t(sim, query_vectors, doc_vectors + (size_t)start * dim, query_tokens, dim,
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

size_t embed_workspace_nbytes(const embed_workspace_t *ws) {
    if (!ws)
        return 0;

    size_t total = sizeof(*ws);
    const embed_model_t *model = ws->model;
    if (!model)
        return total;

    const embed_config_t *c = &model->config;
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
    if (ws->bf16_widen_count > 0 &&
        mul_size(ws->bf16_widen_count, sizeof(float), &widen_bytes) != 0)
        return SIZE_MAX;
    if (add_size(total, widen_bytes, &total) != 0)
        return SIZE_MAX;

    return total;
}

/* ========================================================================
 * Packed forward pass
 * ======================================================================== */

static int build_offsets(const embed_input_t *inputs,
                         int batch,
                         int require_ids,
                         int *offsets,
                         int *total_out,
                         int *max_out) {
    if (!inputs || batch <= 0 || !offsets || !total_out || !max_out)
        return -1;

    int total = 0;
    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        if ((require_ids && !inputs[b].ids) || inputs[b].n_tokens <= 0)
            return -1;
        if (total > INT32_MAX - inputs[b].n_tokens)
            return -1;
        offsets[b] = total;
        total += inputs[b].n_tokens;
        if (inputs[b].n_tokens > max_seq)
            max_seq = inputs[b].n_tokens;
    }
    offsets[batch] = total;
    *total_out = total;
    *max_out = max_seq;
    return 0;
}

static int forward_packed_slice_inplace(const embed_model_t *model,
                                        embed_workspace_t *ws,
                                        const embed_input_t *inputs,
                                        int batch,
                                        const int *offsets,
                                        int total_seq,
                                        int max_seq,
                                        const float *input_states,
                                        int layer_start,
                                        int layer_end,
                                        int apply_final_norm) {
    if (!model || !ws || ws->model != model || !inputs || batch <= 0 || !offsets ||
        total_seq <= 0 || max_seq <= 0)
        return -1;

    const embed_config_t *cfg = &model->config;
    const embed_weights_t *w = &model->weights;
    if (layer_start < 0 || layer_start > layer_end || layer_end > cfg->n_layers ||
        layer_start < model->layer_start || layer_end > model->layer_end ||
        (layer_start == 0 && input_states) || (layer_start != 0 && !input_states) ||
        (layer_start == 0 && !w->embed_tokens.data) ||
        (apply_final_norm && (layer_end != cfg->n_layers || !w->norm)))
        return -1;

    int hidden = cfg->hidden_size;
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim = cfg->head_dim;
    int q_dim = cfg->q_dim;
    int kv_dim = cfg->kv_dim;
    int inter = cfg->intermediate_size;
    float eps = cfg->rms_norm_eps;

    if (ensure_buffers(ws, cfg, total_seq) != 0)
        return -1;
    if (ensure_rope_cache(ws, cfg, max_seq) != 0)
        return -1;
    ensure_attention_scores(ws, offsets, batch);

    const float *rope_cos = ws->rope_cos;
    const float *rope_sin = ws->rope_sin;

    float *x = ws->x;
    float *x_norm = ws->x_norm;
    float *q_buf = ws->q;
    float *k_buf = ws->k;
    float *v_buf = ws->v;
    float *attn_out = ws->attn_out;
    float *proj_out = ws->proj_out;
    float *ffn_gate = ws->ffn_gate;
    float *ffn_up = ws->ffn_up;

    /* A first stage performs embedding lookup. Later stages resume from a
     * packed hidden-state tensor received from the preceding stage. */
    if (layer_start == 0) {
        for (int b = 0; b < batch; b++) {
            int start = offsets[b];
            for (int i = 0; i < inputs[b].n_tokens; i++) {
                int id = inputs[b].ids[i];
                if (id < 0 || id >= cfg->vocab_size) {
                    fprintf(stderr, "embed: invalid token id %d at input %d position %d\n", id, b,
                            i);
                    return -1;
                }
                copy_weight_row(x + (size_t)(start + i) * hidden, &w->embed_tokens, (size_t)id,
                                hidden);
            }
        }
    } else {
        memcpy(x, input_states, (size_t)total_seq * hidden * sizeof(float));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    /* 2. Transformer layers. Linear/MLP ops run over all packed tokens;
     * attention is explicitly block-diagonal by sequence offsets. */
    for (int layer = layer_start; layer < layer_end; layer++) {
        const embed_layer_t *l = &w->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, total_seq, hidden, eps);

        linear_qkv_weight(ws, q_buf, k_buf, v_buf, x_norm, &l->wq, &l->wk, &l->wv, total_seq,
                          hidden, q_dim, kv_dim);

        qwen_rms_norm_per_head(q_buf, l->q_norm, total_seq, n_heads, head_dim, eps);
        qwen_rms_norm_per_head(k_buf, l->k_norm, total_seq, n_kv_heads, head_dim, eps);

        for (int b = 0; b < batch; b++) {
            int start = offsets[b];
            int len = offsets[b + 1] - start;
            qwen_apply_rope_neox(q_buf + (size_t)start * q_dim, rope_cos, rope_sin, len, n_heads,
                                 head_dim);
            qwen_apply_rope_neox(k_buf + (size_t)start * kv_dim, rope_cos, rope_sin, len,
                                 n_kv_heads, head_dim);
        }

        model->attention(attn_out, q_buf, k_buf, v_buf, offsets, batch, n_heads, n_kv_heads,
                         head_dim, scale, ws->attn_scores, ws->attn_scores_bytes);

        linear_nobias_weight(ws, proj_out, attn_out, &l->wo, total_seq, q_dim, hidden);
        qwen_add_inplace(x, proj_out, total_seq * hidden);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, total_seq, hidden, eps);

        linear_pair_weight(ws, ffn_gate, ffn_up, x_norm, &l->gate_proj, &l->up_proj, total_seq,
                           hidden, inter, inter);
        qwen_silu_mul_inplace(ffn_gate, ffn_up, total_seq * inter);
        linear_nobias_weight(ws, proj_out, ffn_gate, &l->down_proj, total_seq, inter, hidden);
        qwen_add_inplace(x, proj_out, total_seq * hidden);
    }

    if (apply_final_norm)
        qwen_rms_norm(x, x, w->norm, total_seq, hidden, eps);
    return 0;
}

static int forward_packed_inplace(const embed_model_t *model,
                                  embed_workspace_t *ws,
                                  const embed_input_t *inputs,
                                  int batch,
                                  const int *offsets,
                                  int total_seq,
                                  int max_seq,
                                  int apply_final_norm) {
    return forward_packed_slice_inplace(model, ws, inputs, batch, offsets, total_seq, max_seq, NULL,
                                        0, model->config.n_layers, apply_final_norm);
}

static int pool_embeddings(const embed_model_t *model,
                           const embed_workspace_t *ws,
                           const int *offsets,
                           int batch,
                           float *out_embeddings) {
    const embed_config_t *cfg = &model->config;
    int hidden = cfg->hidden_size;
    float eps = cfg->rms_norm_eps;
    const float *norm_weight = model->weights.norm;
    const float *x = ws->x;
    if (hidden <= 0 || !norm_weight || !x)
        return -1;

    for (int b = 0; b < batch; b++) {
        int start = offsets[b];
        int end = offsets[b + 1];
        int len = end - start;
        float *emb = out_embeddings + (size_t)b * hidden;

        memset(emb, 0, (size_t)hidden * sizeof(float));
        if (cfg->pooling_mode == EMBED_POOL_LAST_TOKEN) {
            const float *row = x + (size_t)(end - 1) * hidden;
            float sum_sq = 0.0f;
            for (int d = 0; d < hidden; d++)
                sum_sq += row[d] * row[d];
            float rms_inv = 1.0f / sqrtf(sum_sq / (float)hidden + eps);
            for (int d = 0; d < hidden; d++)
                emb[d] = row[d] * rms_inv * norm_weight[d];
        } else {
            for (int i = start; i < end; i++) {
                const float *row = x + (size_t)i * hidden;

                float sum_sq = 0.0f;
                for (int d = 0; d < hidden; d++)
                    sum_sq += row[d] * row[d];

                float rms_inv = 1.0f / sqrtf(sum_sq / (float)hidden + eps);
                for (int d = 0; d < hidden; d++)
                    emb[d] += row[d] * rms_inv * norm_weight[d];
            }
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < hidden; d++)
                emb[d] *= inv_len;
        }

        if (cfg->normalize_embeddings && embed_l2_normalize(emb, hidden) != 0)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Public forward/embed APIs
 * ======================================================================== */

int embed_model_forward_into(const embed_model_t *model,
                             embed_workspace_t *ws,
                             const int *token_ids,
                             int n_tokens,
                             float *out_states) {
    if (!model || !ws || !token_ids || n_tokens <= 0 || !out_states)
        return -1;

    size_t count, bytes;
    if (mul_size((size_t)n_tokens, (size_t)model->config.hidden_size, &count) != 0 ||
        mul_size(count, sizeof(float), &bytes) != 0)
        return -1;

    embed_input_t input = {token_ids, n_tokens};
    int offsets[2] = {0, n_tokens};
    if (forward_packed_inplace(model, ws, &input, 1, offsets, n_tokens, n_tokens, 1) != 0)
        return -1;

    memcpy(out_states, ws->x, bytes);
    return 0;
}

int embed_pool_spans(const embed_config_t *cfg,
                     const float *states,
                     int n_tokens,
                     const embed_span_t *spans,
                     int n_spans,
                     float *out_embeddings) {
    if (!cfg || !states || n_tokens <= 0 || !spans || n_spans <= 0 || !out_embeddings)
        return -1;

    int hidden = cfg->hidden_size;
    if (hidden <= 0)
        return -1;

    for (int s = 0; s < n_spans; s++) {
        int start = spans[s].start;
        int len = spans[s].n_tokens;
        if (start < 0 || len <= 0 || start > n_tokens || len > n_tokens - start)
            return -1;

        float *emb = out_embeddings + (size_t)s * hidden;
        memset(emb, 0, (size_t)hidden * sizeof(float));

        for (int t = 0; t < len; t++) {
            const float *row = states + (size_t)(start + t) * hidden;
            for (int d = 0; d < hidden; d++)
                emb[d] += row[d];
        }

        float inv_len = 1.0f / (float)len;
        for (int d = 0; d < hidden; d++) {
            emb[d] *= inv_len;
        }
    }
    return 0;
}

int embed_model_encode_spans_batch(const embed_model_t *model,
                                   embed_workspace_t *ws,
                                   const embed_context_input_t *inputs,
                                   int batch,
                                   float *out_embeddings) {
    if (!model || !ws || !inputs || batch <= 0 || !out_embeddings)
        return -1;

    if (ensure_offsets(ws, batch) != 0)
        return -1;
    if ((size_t)batch > SIZE_MAX / sizeof(embed_input_t))
        return -1;
    embed_input_t *packed_inputs = malloc((size_t)batch * sizeof(*packed_inputs));
    if (!packed_inputs)
        return -1;
    int *offsets = ws->offsets;
    int total_seq = 0;
    int max_seq = 0;
    int total_spans = 0;
    for (int b = 0; b < batch; b++) {
        const embed_context_input_t *input = &inputs[b];
        int n_tokens = input->input.n_tokens;
        if (!input->input.ids || n_tokens <= 0 || !input->spans || input->n_spans <= 0 ||
            total_seq > INT_MAX - n_tokens || total_spans > INT_MAX - input->n_spans) {
            free(packed_inputs);
            return -1;
        }
        packed_inputs[b] = input->input;
        offsets[b] = total_seq;
        total_seq += n_tokens;
        total_spans += input->n_spans;
        if (n_tokens > max_seq)
            max_seq = n_tokens;
        for (int s = 0; s < input->n_spans; s++) {
            int start = input->spans[s].start;
            int len = input->spans[s].n_tokens;
            if (start < 0 || len <= 0 || start > n_tokens || len > n_tokens - start) {
                free(packed_inputs);
                return -1;
            }
        }
    }
    offsets[batch] = total_seq;

    if (forward_packed_inplace(model, ws, packed_inputs, batch, offsets, total_seq, max_seq, 1) !=
        0) {
        free(packed_inputs);
        return -1;
    }
    free(packed_inputs);

    int span_offset = 0;
    int hidden = model->config.hidden_size;
    for (int b = 0; b < batch; b++) {
        const embed_context_input_t *input = &inputs[b];
        if (embed_pool_spans(&model->config, ws->x + (size_t)offsets[b] * hidden,
                             input->input.n_tokens, input->spans, input->n_spans,
                             out_embeddings + (size_t)span_offset * hidden) != 0)
            return -1;
        span_offset += input->n_spans;
    }
    return 0;
}

int embed_model_encode_spans(const embed_model_t *model,
                             embed_workspace_t *ws,
                             const int *token_ids,
                             int n_tokens,
                             const embed_span_t *spans,
                             int n_spans,
                             float *out_embeddings) {
    embed_context_input_t input = {
        .input = {token_ids, n_tokens},
        .spans = spans,
        .n_spans = n_spans,
    };
    return embed_model_encode_spans_batch(model, ws, &input, 1, out_embeddings);
}

float *embed_model_forward(const embed_model_t *model,
                           embed_workspace_t *ws,
                           const int *token_ids,
                           int n_tokens) {
    if (!model || !ws || !token_ids || n_tokens <= 0)
        return NULL;

    size_t n;
    if (mul_size((size_t)n_tokens, (size_t)model->config.hidden_size, &n) != 0)
        return NULL;
    float *out = malloc_floats(n);
    if (!out)
        return NULL;

    if (embed_model_forward_into(model, ws, token_ids, n_tokens, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}

int embed_model_encode_batch(const embed_model_t *model,
                             embed_workspace_t *ws,
                             const embed_input_t *inputs,
                             int batch,
                             float *out_embeddings) {
    if (!model || !ws || !inputs || batch <= 0 || !out_embeddings)
        return -1;

    if (ensure_offsets(ws, batch) != 0)
        return -1;
    int *offsets = ws->offsets;

    int total_seq = 0;
    int max_seq = 0;
    if (build_offsets(inputs, batch, 1, offsets, &total_seq, &max_seq) != 0)
        return -1;

    int rc = forward_packed_inplace(model, ws, inputs, batch, offsets, total_seq, max_seq, 0);
    if (rc == 0)
        rc = pool_embeddings(model, ws, offsets, batch, out_embeddings);

    return rc;
}

int embed_pool_batch(const embed_config_t *cfg,
                     const float *states,
                     const int *seq_lengths,
                     int batch,
                     float *out_embeddings) {
    if (!cfg || !states || !seq_lengths || batch <= 0 || !out_embeddings || cfg->hidden_size <= 0)
        return -1;

    int offset = 0;
    for (int b = 0; b < batch; b++) {
        int len = seq_lengths[b];
        if (len <= 0 || offset > INT_MAX - len)
            return -1;

        float *emb = out_embeddings + (size_t)b * cfg->hidden_size;
        if (cfg->pooling_mode == EMBED_POOL_LAST_TOKEN) {
            const float *row = states + (size_t)(offset + len - 1) * cfg->hidden_size;
            memcpy(emb, row, (size_t)cfg->hidden_size * sizeof(float));
        } else {
            memset(emb, 0, (size_t)cfg->hidden_size * sizeof(float));
            for (int t = 0; t < len; t++) {
                const float *row = states + (size_t)(offset + t) * cfg->hidden_size;
                for (int d = 0; d < cfg->hidden_size; d++)
                    emb[d] += row[d];
            }
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < cfg->hidden_size; d++)
                emb[d] *= inv_len;
        }
        if (cfg->normalize_embeddings && embed_l2_normalize(emb, cfg->hidden_size) != 0)
            return -1;
        offset += len;
    }
    return 0;
}

int embed_model_encode_into(const embed_model_t *model,
                            embed_workspace_t *ws,
                            const int *token_ids,
                            int n_tokens,
                            float *out_embedding) {
    embed_input_t input = {token_ids, n_tokens};
    return embed_model_encode_batch(model, ws, &input, 1, out_embedding);
}

float *embed_model_encode(const embed_model_t *model,
                          embed_workspace_t *ws,
                          const int *token_ids,
                          int n_tokens) {
    if (!model || !ws || !token_ids || n_tokens <= 0)
        return NULL;

    int hidden = model->config.hidden_size;
    if (hidden <= 0)
        return NULL;
    float *emb = malloc_floats((size_t)hidden);
    if (!emb)
        return NULL;

    if (embed_model_encode_into(model, ws, token_ids, n_tokens, emb) != 0) {
        free(emb);
        return NULL;
    }
    return emb;
}

/* ========================================================================
 * Vector helpers
 * ======================================================================== */

int embed_l2_normalize(float *vec, int dim) {
    if (!vec || dim <= 0)
        return -1;

    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++)
        norm_sq += vec[i] * vec[i];

    if (!(norm_sq > 0.0f) || !isfinite(norm_sq))
        return -1;

    float inv_norm = 1.0f / sqrtf(norm_sq);
    for (int i = 0; i < dim; i++)
        vec[i] *= inv_norm;
    return 0;
}

float embed_cosine_similarity(const float *a, const float *b, int dim) {
    if (!a || !b || dim <= 0)
        return 0.0f;

    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    if (!(norm_a_sq > 0.0f) || !(norm_b_sq > 0.0f) || !isfinite(norm_a_sq) || !isfinite(norm_b_sq))
        return 0.0f;
    return dot / sqrtf(norm_a_sq * norm_b_sq);
}
