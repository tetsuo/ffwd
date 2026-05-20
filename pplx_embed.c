#include "pplx_embed.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Globals
 * ======================================================================== */

int pplx_verbose = 0;
int qwen_verbose = 0;

/* ========================================================================
 * Minimal JSON helpers for config.json parsing
 * ======================================================================== */

static const char *skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

/* Find "key": <value> in a flat JSON object. Returns pointer to value start. */
static const char *json_find_key(const char *json, const char *key)
{
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strstr(p, key)) != NULL) {
        /* Check it's actually a quoted key: "key" */
        if (p > json && *(p - 1) == '"') {
            const char *after = p + klen;
            if (*after == '"') {
                after = skip_ws(after + 1);
                if (*after == ':') return skip_ws(after + 1);
            }
        }
        p += klen;
    }
    return NULL;
}

static int json_get_int(const char *json, const char *key, int fallback)
{
    const char *v = json_find_key(json, key);
    if (!v) return fallback;
    return atoi(v);
}

static double json_get_double(const char *json, const char *key, double fallback)
{
    const char *v = json_find_key(json, key);
    if (!v) return fallback;
    return atof(v);
}

static int parse_config(pplx_config_t *cfg, const char *model_dir)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "pplx_load: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf || fread(buf, 1, sz, f) != (size_t)sz) {
        fclose(f); free(buf); return -1;
    }
    fclose(f);
    buf[sz] = '\0';

    cfg->hidden_size      = json_get_int(buf, "hidden_size", 0);
    cfg->n_layers         = json_get_int(buf, "num_hidden_layers", 0);
    cfg->n_heads          = json_get_int(buf, "num_attention_heads", 0);
    cfg->n_kv_heads       = json_get_int(buf, "num_key_value_heads", 0);
    cfg->head_dim         = json_get_int(buf, "head_dim", PPLX_HEAD_DIM);
    cfg->intermediate_size = json_get_int(buf, "intermediate_size", 0);
    cfg->vocab_size       = json_get_int(buf, "vocab_size", PPLX_VOCAB_SIZE);
    cfg->rms_norm_eps     = (float)json_get_double(buf, "rms_norm_eps", 1e-6);
    cfg->rope_theta       = (float)json_get_double(buf, "rope_theta", 1000000.0);

    cfg->q_dim  = cfg->n_heads    * cfg->head_dim;
    cfg->kv_dim = cfg->n_kv_heads * cfg->head_dim;

    free(buf);

    /* Sanity checks */
    if (cfg->hidden_size <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0) {
        fprintf(stderr, "pplx_load: invalid config in %s "
                "(hidden=%d, layers=%d, heads=%d)\n",
                path, cfg->hidden_size, cfg->n_layers, cfg->n_heads);
        return -1;
    }
    if (cfg->n_layers > PPLX_MAX_LAYERS) {
        fprintf(stderr, "pplx_load: too many layers (%d > %d)\n",
                cfg->n_layers, PPLX_MAX_LAYERS);
        return -1;
    }

    if (pplx_verbose >= 1)
        fprintf(stderr, "config: hidden=%d, layers=%d, heads=%d/%d, "
                "inter=%d, head_dim=%d\n",
                cfg->hidden_size, cfg->n_layers,
                cfg->n_heads, cfg->n_kv_heads,
                cfg->intermediate_size, cfg->head_dim);

    return 0;
}

/* ========================================================================
 * Weight loading (direct mmap pointers into safetensors)
 * ======================================================================== */

static const float *load_f32_direct(multi_safetensors_t *ms, const char *name)
{
    safetensors_file_t *sf = NULL;
    const safetensor_t *t  = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "pplx: tensor not found: %s\n", name);
        return NULL;
    }
    if (t->dtype != DTYPE_F32) {
        fprintf(stderr, "pplx: expected F32 for %s, got dtype=%d\n",
                name, (int)t->dtype);
        return NULL;
    }
    return (const float *)safetensors_data(sf, t);
}

/* ========================================================================
 * Working buffer management
 * ======================================================================== */

static int ensure_buffers(pplx_ctx_t *ctx, int seq)
{
    if (seq <= ctx->buf_seq_cap) return 0;

    int cap = ctx->buf_seq_cap > 0 ? ctx->buf_seq_cap : 64;
    while (cap < seq) cap *= 2;

    const pplx_config_t *c = &ctx->config;

#define R(ptr, n) do {                                          \
    void *_t = realloc((ptr), (size_t)(n) * sizeof(float));     \
    if (!_t) return -1;                                          \
    (ptr) = (float *)_t;                                         \
} while (0)

    R(ctx->x,        cap * c->hidden_size);
    R(ctx->x_norm,   cap * c->hidden_size);
    R(ctx->q,        cap * c->q_dim);
    R(ctx->k,        cap * c->kv_dim);
    R(ctx->v,        cap * c->kv_dim);
    R(ctx->attn_out, cap * c->q_dim);
    R(ctx->proj_out, cap * c->hidden_size);
    R(ctx->ffn_gate, cap * c->intermediate_size);
    R(ctx->ffn_up,   cap * c->intermediate_size);
    R(ctx->ffn_out,  cap * c->hidden_size);

#undef R

    ctx->buf_seq_cap = cap;
    return 0;
}

/* ========================================================================
 * RoPE cache
 * ======================================================================== */

static int ensure_rope_cache(pplx_ctx_t *ctx, int n_pos)
{
    if (n_pos <= ctx->rope_cache_cap) return 0;

    int cap = ctx->rope_cache_cap > 0 ? ctx->rope_cache_cap : 512;
    while (cap < n_pos) cap *= 2;

    int   head_dim = ctx->config.head_dim;
    int   half     = head_dim / 2;
    float theta    = ctx->config.rope_theta;

    size_t n = (size_t)cap * head_dim;
    float *nc = (float *)realloc(ctx->rope_cos, n * sizeof(float));
    float *ns = (float *)realloc(ctx->rope_sin, n * sizeof(float));
    if (!nc || !ns) {
        if (nc) ctx->rope_cos = nc;
        if (ns) ctx->rope_sin = ns;
        return -1;
    }
    ctx->rope_cos = nc;
    ctx->rope_sin = ns;

    for (int pos = ctx->rope_cache_cap; pos < cap; pos++) {
        float fp = (float)pos;
        float *c = ctx->rope_cos + (size_t)pos * head_dim;
        float *s = ctx->rope_sin + (size_t)pos * head_dim;
        for (int d = 0; d < half; d++) {
            float inv = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = fp * inv;
            float cv = cosf(angle), sv = sinf(angle);
            c[d] = cv;  c[half + d] = cv;
            s[d] = sv;  s[half + d] = sv;
        }
    }

    ctx->rope_cache_cap = cap;
    return 0;
}

/* ========================================================================
 * Load
 * ======================================================================== */

pplx_ctx_t *pplx_load(const char *model_dir)
{
    pplx_ctx_t *ctx = (pplx_ctx_t *)calloc(1, sizeof(pplx_ctx_t));
    if (!ctx) return NULL;

    /* Parse config.json */
    if (parse_config(&ctx->config, model_dir) != 0) {
        free(ctx);
        return NULL;
    }

    const pplx_config_t *cfg = &ctx->config;

    /* Open safetensors (handles single file or multi-shard) */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "pplx_load: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    if (pplx_verbose >= 1)
        fprintf(stderr, "pplx_load: loading weights from %s\n", model_dir);

    /* Token embeddings */
    pplx_weights_t *w = &ctx->weights;
    w->embed_tokens = load_f32_direct(ms, "embed_tokens.weight");
    if (!w->embed_tokens) goto fail;

    /* Allocate layer array */
    w->layers = (pplx_layer_t *)calloc(cfg->n_layers, sizeof(pplx_layer_t));
    if (!w->layers) goto fail;

    /* Per-layer weights */
    char name[256];
    for (int i = 0; i < cfg->n_layers; i++) {
        pplx_layer_t *l = &w->layers[i];

#define LOAD(field, fmt) do {                               \
    snprintf(name, sizeof(name), fmt, i);                   \
    l->field = load_f32_direct(ms, name);                   \
    if (!l->field) goto fail;                               \
} while (0)

        LOAD(wq,             "layers.%d.self_attn.q_proj.weight");
        LOAD(wk,             "layers.%d.self_attn.k_proj.weight");
        LOAD(wv,             "layers.%d.self_attn.v_proj.weight");
        LOAD(wo,             "layers.%d.self_attn.o_proj.weight");
        LOAD(q_norm,         "layers.%d.self_attn.q_norm.weight");
        LOAD(k_norm,         "layers.%d.self_attn.k_norm.weight");
        LOAD(input_norm,     "layers.%d.input_layernorm.weight");
        LOAD(post_attn_norm, "layers.%d.post_attention_layernorm.weight");
        LOAD(gate_proj,      "layers.%d.mlp.gate_proj.weight");
        LOAD(up_proj,        "layers.%d.mlp.up_proj.weight");
        LOAD(down_proj,      "layers.%d.mlp.down_proj.weight");

#undef LOAD

        if (pplx_verbose >= 2)
            fprintf(stderr, "  layer %d loaded\n", i);
    }

    /* Final RMSNorm */
    w->norm = load_f32_direct(ms, "norm.weight");
    if (!w->norm) goto fail;

    if (pplx_verbose >= 1)
        fprintf(stderr, "pplx_load: %d layers loaded (%d-dim embeddings)\n",
                cfg->n_layers, cfg->hidden_size);

    return ctx;

fail:
    pplx_free(ctx);
    return NULL;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void pplx_free(pplx_ctx_t *ctx)
{
    if (!ctx) return;
    if (ctx->safetensors)
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    free(ctx->weights.layers);
    free(ctx->x);       free(ctx->x_norm);
    free(ctx->q);       free(ctx->k);        free(ctx->v);
    free(ctx->attn_out); free(ctx->proj_out);
    free(ctx->ffn_gate); free(ctx->ffn_up);   free(ctx->ffn_out);
    free(ctx->rope_cos); free(ctx->rope_sin);
    free(ctx);
}

/* ========================================================================
 * Packed forward pass
 * ======================================================================== */

static int build_offsets(const pplx_input_t *inputs, int batch,
                         int *offsets, int *total_out, int *max_out)
{
    if (!inputs || batch <= 0 || !offsets || !total_out || !max_out)
        return -1;

    int total = 0;
    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        if (!inputs[b].ids || inputs[b].n_tokens <= 0)
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

static int forward_packed_inplace(pplx_ctx_t *ctx, const pplx_input_t *inputs,
                                  int batch, const int *offsets,
                                  int total_seq, int max_seq)
{
    if (!ctx || !inputs || batch <= 0 || !offsets || total_seq <= 0 || max_seq <= 0)
        return -1;

    const pplx_config_t  *cfg = &ctx->config;
    const pplx_weights_t *w   = &ctx->weights;

    int hidden     = cfg->hidden_size;
    int n_heads    = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim   = cfg->head_dim;
    int q_dim      = cfg->q_dim;
    int kv_dim     = cfg->kv_dim;
    int inter      = cfg->intermediate_size;
    float eps      = cfg->rms_norm_eps;

    if (ensure_buffers(ctx, total_seq) != 0) return -1;
    if (ensure_rope_cache(ctx, max_seq) != 0) return -1;

    const float *rope_cos = ctx->rope_cos;
    const float *rope_sin = ctx->rope_sin;

    float *x        = ctx->x;
    float *x_norm   = ctx->x_norm;
    float *q_buf    = ctx->q;
    float *k_buf    = ctx->k;
    float *v_buf    = ctx->v;
    float *attn_out = ctx->attn_out;
    float *proj_out = ctx->proj_out;
    float *ffn_gate = ctx->ffn_gate;
    float *ffn_up   = ctx->ffn_up;
    float *ffn_out  = ctx->ffn_out;

    /* 1. Token embedding lookup into packed [total_seq, hidden]. */
    for (int b = 0; b < batch; b++) {
        int start = offsets[b];
        for (int i = 0; i < inputs[b].n_tokens; i++) {
            int id = inputs[b].ids[i];
            if (id < 0 || id >= cfg->vocab_size) {
                fprintf(stderr, "pplx: invalid token id %d at input %d position %d\n",
                        id, b, i);
                return -1;
            }
            memcpy(x + (size_t)(start + i) * hidden,
                   w->embed_tokens + (size_t)id * hidden,
                   (size_t)hidden * sizeof(float));
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    /* 2. Transformer layers. Linear/MLP ops run over all packed tokens;
     * attention is explicitly block-diagonal by sequence offsets. */
    for (int layer = 0; layer < cfg->n_layers; layer++) {
        const pplx_layer_t *l = &w->layers[layer];

        qwen_rms_norm(x_norm, x, l->input_norm, total_seq, hidden, eps);

        qwen_linear_nobias(q_buf, x_norm, l->wq, total_seq, hidden, q_dim);
        qwen_linear_nobias(k_buf, x_norm, l->wk, total_seq, hidden, kv_dim);
        qwen_linear_nobias(v_buf, x_norm, l->wv, total_seq, hidden, kv_dim);

        qwen_rms_norm_per_head(q_buf, l->q_norm, total_seq, n_heads,    head_dim, eps);
        qwen_rms_norm_per_head(k_buf, l->k_norm, total_seq, n_kv_heads, head_dim, eps);

        for (int b = 0; b < batch; b++) {
            int start = offsets[b];
            int len = offsets[b + 1] - start;
            qwen_apply_rope_neox(q_buf + (size_t)start * q_dim,
                                 rope_cos, rope_sin, len, n_heads, head_dim);
            qwen_apply_rope_neox(k_buf + (size_t)start * kv_dim,
                                 rope_cos, rope_sin, len, n_kv_heads, head_dim);
        }

        qwen_bidirectional_gqa_attention_packed(attn_out, q_buf, k_buf, v_buf,
                                                offsets, batch, n_heads,
                                                n_kv_heads, head_dim, scale);

        qwen_linear_nobias(proj_out, attn_out, l->wo, total_seq, q_dim, hidden);
        qwen_add_inplace(x, proj_out, total_seq * hidden);

        qwen_rms_norm(x_norm, x, l->post_attn_norm, total_seq, hidden, eps);

        qwen_linear_nobias(ffn_gate, x_norm, l->gate_proj, total_seq, hidden, inter);
        qwen_linear_nobias(ffn_up,   x_norm, l->up_proj,   total_seq, hidden, inter);
        qwen_silu(ffn_gate, total_seq * inter);
        qwen_mul_inplace(ffn_gate, ffn_up, total_seq * inter);
        qwen_linear_nobias(ffn_out, ffn_gate, l->down_proj, total_seq, inter, hidden);
        qwen_add_inplace(x, ffn_out, total_seq * hidden);
    }

    qwen_rms_norm(x, x, w->norm, total_seq, hidden, eps);
    return 0;
}

static void pool_embeddings(const pplx_ctx_t *ctx, const int *offsets,
                            int batch, float *out_embeddings)
{
    int hidden = ctx->config.hidden_size;
    const float *x = ctx->x;

    for (int b = 0; b < batch; b++) {
        int start = offsets[b];
        int end = offsets[b + 1];
        int len = end - start;
        float *emb = out_embeddings + (size_t)b * hidden;

        memset(emb, 0, (size_t)hidden * sizeof(float));
        for (int i = start; i < end; i++) {
            const float *row = x + (size_t)i * hidden;
            for (int d = 0; d < hidden; d++)
                emb[d] += row[d];
        }

        float inv_len = 1.0f / (float)len;
        for (int d = 0; d < hidden; d++)
            emb[d] *= inv_len;

        float norm_sq = 0.0f;
        for (int d = 0; d < hidden; d++)
            norm_sq += emb[d] * emb[d];
        float nv = sqrtf(norm_sq);
        if (nv > 1e-8f) {
            float inv = 1.0f / nv;
            for (int d = 0; d < hidden; d++)
                emb[d] *= inv;
        }
    }
}

/* ========================================================================
 * Public forward/embed APIs
 * ======================================================================== */

int pplx_forward_into(pplx_ctx_t *ctx, const int *token_ids, int n_tokens,
                      float *out_states)
{
    if (!ctx || !token_ids || n_tokens <= 0 || !out_states) return -1;

    pplx_input_t input = { token_ids, n_tokens };
    int offsets[2] = { 0, n_tokens };
    if (forward_packed_inplace(ctx, &input, 1, offsets, n_tokens, n_tokens) != 0)
        return -1;

    size_t out_size = (size_t)n_tokens * ctx->config.hidden_size * sizeof(float);
    memcpy(out_states, ctx->x, out_size);
    return 0;
}

float *pplx_forward(pplx_ctx_t *ctx, const int *token_ids, int n_tokens)
{
    if (!ctx || !token_ids || n_tokens <= 0) return NULL;

    size_t n = (size_t)n_tokens * ctx->config.hidden_size;
    float *out = (float *)malloc(n * sizeof(float));
    if (!out) return NULL;

    if (pplx_forward_into(ctx, token_ids, n_tokens, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}

int pplx_embed_batch(pplx_ctx_t *ctx, const pplx_input_t *inputs, int batch,
                     float *out_embeddings)
{
    if (!ctx || !inputs || batch <= 0 || !out_embeddings) return -1;

    int *offsets = (int *)malloc((size_t)(batch + 1) * sizeof(int));
    if (!offsets) return -1;

    int total_seq = 0;
    int max_seq = 0;
    if (build_offsets(inputs, batch, offsets, &total_seq, &max_seq) != 0) {
        free(offsets);
        return -1;
    }

    int rc = forward_packed_inplace(ctx, inputs, batch, offsets, total_seq, max_seq);
    if (rc == 0)
        pool_embeddings(ctx, offsets, batch, out_embeddings);

    free(offsets);
    return rc;
}

int pplx_embed_into(pplx_ctx_t *ctx, const int *token_ids, int n_tokens,
                    float *out_embedding)
{
    pplx_input_t input = { token_ids, n_tokens };
    return pplx_embed_batch(ctx, &input, 1, out_embedding);
}

float *pplx_embed(pplx_ctx_t *ctx, const int *token_ids, int n_tokens)
{
    if (!ctx || !token_ids || n_tokens <= 0) return NULL;

    int hidden = ctx->config.hidden_size;
    float *emb = (float *)malloc((size_t)hidden * sizeof(float));
    if (!emb) return NULL;

    if (pplx_embed_into(ctx, token_ids, n_tokens, emb) != 0) {
        free(emb);
        return NULL;
    }
    return emb;
}

/* ========================================================================
 * Cosine similarity
 * ======================================================================== */

float pplx_cosine_similarity(const float *a, const float *b, int dim)
{
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) dot += a[i] * b[i];
    return dot;
}
