#include "pplx_embed_mlx.h"
#include "pplx_embed.h"
#include "qwen_asr_safetensors.h"

#include <mlx/c/mlx.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* ========================================================================
 * Per-layer MLX weight arrays
 * ======================================================================== */

typedef struct {
    mlx_array wq, wk, wv, wo;
    mlx_array q_norm, k_norm;
    mlx_array input_norm, post_attn_norm;
    mlx_array gate_proj, up_proj, down_proj;
} mlx_layer_t;

struct pplx_mlx_ctx {
    pplx_config_t config;
    mlx_array embed_tokens;
    mlx_layer_t *layers;        /* heap-allocated [n_layers] */
    mlx_array norm;
    mlx_stream stream;
    multi_safetensors_t *ms;
};

/* ========================================================================
 * Helpers
 * ======================================================================== */

static int arr_ok(mlx_array a) { return a.ctx != NULL; }

static mlx_array load_tensor(multi_safetensors_t *ms, const char *name,
                              const int *shape, int ndim)
{
    safetensors_file_t *sf = NULL;
    const safetensor_t *t  = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "mlx: tensor not found: %s\n", name);
        return (mlx_array){0};
    }
    mlx_dtype dtype;
    if (t->dtype == DTYPE_F32) {
        dtype = MLX_FLOAT32;
    } else if (t->dtype == DTYPE_BF16) {
        dtype = MLX_BFLOAT16;
    } else {
        fprintf(stderr, "mlx: expected F32 or BF16 for %s\n", name);
        return (mlx_array){0};
    }
    return mlx_array_new_data(safetensors_data(sf, t), shape, ndim, dtype);
}

/* y = x @ W^T */
static mlx_array linear(mlx_array x, mlx_array W, mlx_stream s)
{
    mlx_array Wt  = mlx_array_new();
    mlx_transpose(&Wt, W, s);
    mlx_array res = mlx_array_new();
    mlx_matmul(&res, x, Wt, s);
    mlx_array_free(Wt);
    return res;
}

const pplx_config_t *pplx_mlx_config(const pplx_mlx_ctx_t *ctx)
{
    return ctx ? &ctx->config : NULL;
}

/* ========================================================================
 * Load
 * ======================================================================== */

/* We duplicate a minimal config parser here so MLX can load standalone.
 * (Same logic as pplx_embed.c:parse_config, kept inline for simplicity.) */

static const char *skip_ws_m(const char *p)
{
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    return p;
}

static const char *json_find_m(const char *json, const char *key)
{
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strstr(p, key)) != NULL) {
        if (p > json && *(p - 1) == '"') {
            const char *a = p + klen;
            if (*a == '"') {
                a = skip_ws_m(a + 1);
                if (*a == ':') return skip_ws_m(a + 1);
            }
        }
        p += klen;
    }
    return NULL;
}

static int mlx_parse_config(pplx_config_t *cfg, const char *model_dir)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "mlx: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf || fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return -1; }
    fclose(f);
    buf[sz] = '\0';

#define GI(k, fb) do { const char *v = json_find_m(buf, k); cfg->fb = v ? atoi(v) : 0; } while(0)
#define GF(k, fb, def) do { const char *v = json_find_m(buf, k); cfg->fb = v ? (float)atof(v) : def; } while(0)
    GI("hidden_size",         hidden_size);
    GI("num_hidden_layers",   n_layers);
    GI("num_attention_heads",  n_heads);
    GI("num_key_value_heads",  n_kv_heads);
    GI("intermediate_size",   intermediate_size);
    cfg->head_dim  = PPLX_HEAD_DIM;
    cfg->vocab_size = PPLX_VOCAB_SIZE;
    GF("rms_norm_eps",  rms_norm_eps, 1e-6f);
    GF("rope_theta",    rope_theta, 1000000.0f);
#undef GI
#undef GF

    cfg->q_dim  = cfg->n_heads    * cfg->head_dim;
    cfg->kv_dim = cfg->n_kv_heads * cfg->head_dim;
    free(buf);

    if (cfg->hidden_size <= 0 || cfg->n_layers <= 0) {
        fprintf(stderr, "mlx: bad config (hidden=%d layers=%d)\n",
                cfg->hidden_size, cfg->n_layers);
        return -1;
    }
    return 0;
}

pplx_mlx_ctx_t *pplx_mlx_load(const char *model_dir)
{
    pplx_mlx_ctx_t *ctx = calloc(1, sizeof(pplx_mlx_ctx_t));
    if (!ctx) return NULL;

    if (mlx_parse_config(&ctx->config, model_dir) != 0) { free(ctx); return NULL; }

    const pplx_config_t *c = &ctx->config;
    ctx->stream = mlx_default_gpu_stream_new();

    ctx->ms = multi_safetensors_open(model_dir);
    if (!ctx->ms) {
        fprintf(stderr, "mlx: failed to open safetensors in %s\n", model_dir);
        free(ctx); return NULL;
    }

    int h = c->hidden_size, qd = c->q_dim, kvd = c->kv_dim;
    int inter = c->intermediate_size, hd = c->head_dim;

    /* Embedding table */
    int emb_shape[] = {c->vocab_size, h};
    ctx->embed_tokens = load_tensor(ctx->ms, "embed_tokens.weight", emb_shape, 2);
    if (!arr_ok(ctx->embed_tokens)) goto fail;

    /* Layers */
    ctx->layers = calloc(c->n_layers, sizeof(mlx_layer_t));
    if (!ctx->layers) goto fail;

    char name[256];
    for (int i = 0; i < c->n_layers; i++) {
        mlx_layer_t *l = &ctx->layers[i];

#define LD(fld, fmt, ...) do {                                    \
    snprintf(name, sizeof(name), fmt, i);                         \
    int sh[] = {__VA_ARGS__};                                     \
    l->fld = load_tensor(ctx->ms, name, sh, sizeof(sh)/sizeof(sh[0])); \
    if (!arr_ok(l->fld)) goto fail;                               \
} while (0)

        LD(wq,             "layers.%d.self_attn.q_proj.weight", qd,  h);
        LD(wk,             "layers.%d.self_attn.k_proj.weight", kvd, h);
        LD(wv,             "layers.%d.self_attn.v_proj.weight", kvd, h);
        LD(wo,             "layers.%d.self_attn.o_proj.weight", h,   qd);
        LD(q_norm,         "layers.%d.self_attn.q_norm.weight", hd);
        LD(k_norm,         "layers.%d.self_attn.k_norm.weight", hd);
        LD(input_norm,     "layers.%d.input_layernorm.weight", h);
        LD(post_attn_norm, "layers.%d.post_attention_layernorm.weight", h);
        LD(gate_proj,      "layers.%d.mlp.gate_proj.weight", inter, h);
        LD(up_proj,        "layers.%d.mlp.up_proj.weight", inter, h);
        LD(down_proj,      "layers.%d.mlp.down_proj.weight", h, inter);
#undef LD

        if (pplx_verbose >= 2) fprintf(stderr, "  mlx layer %d loaded\n", i);
    }

    int ns[] = {h};
    ctx->norm = load_tensor(ctx->ms, "norm.weight", ns, 1);
    if (!arr_ok(ctx->norm)) goto fail;

    if (pplx_verbose >= 1)
        fprintf(stderr, "mlx: %d layers loaded (%d-dim)\n", c->n_layers, h);

    return ctx;

fail:
    pplx_mlx_free(ctx);
    return NULL;
}

/* ========================================================================
 * Free
 * ======================================================================== */

static void free_mlx_layer(mlx_layer_t *l)
{
    mlx_array_free(l->wq); mlx_array_free(l->wk);
    mlx_array_free(l->wv); mlx_array_free(l->wo);
    mlx_array_free(l->q_norm); mlx_array_free(l->k_norm);
    mlx_array_free(l->input_norm); mlx_array_free(l->post_attn_norm);
    mlx_array_free(l->gate_proj); mlx_array_free(l->up_proj);
    mlx_array_free(l->down_proj);
}

void pplx_mlx_free(pplx_mlx_ctx_t *ctx)
{
    if (!ctx) return;
    mlx_array_free(ctx->embed_tokens);
    if (ctx->layers) {
        for (int i = 0; i < ctx->config.n_layers; i++)
            free_mlx_layer(&ctx->layers[i]);
        free(ctx->layers);
    }
    mlx_array_free(ctx->norm);
    mlx_stream_free(ctx->stream);
    if (ctx->ms) multi_safetensors_close(ctx->ms);
    free(ctx);
}

/* ========================================================================
 * Forward pass
 * ======================================================================== */

static mlx_array mlx_forward_layers(pplx_mlx_ctx_t *ctx, mlx_array x,
                                    int batch, int max_seq, int has_padding,
                                    mlx_array attn_mask,
                                    int layer_start, int layer_end)
{
    const pplx_config_t *c = &ctx->config;
    mlx_stream S = ctx->stream;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int q_dim = c->q_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    mlx_array null_arr = (mlx_array){0};

    for (int layer = layer_start; layer < layer_end; layer++) {
        mlx_layer_t *l = &ctx->layers[layer];

        mlx_array xn = mlx_array_new();
        mlx_fast_rms_norm(&xn, x, l->input_norm, c->rms_norm_eps, S);

        mlx_array q_flat = linear(xn, l->wq, S);
        mlx_array k_flat = linear(xn, l->wk, S);
        mlx_array v_flat = linear(xn, l->wv, S);
        mlx_array_free(xn);

        int q_shape[] = {batch, max_seq, n_heads,    head_dim};
        int k_shape[] = {batch, max_seq, n_kv_heads, head_dim};
        mlx_array q = mlx_array_new();
        mlx_array k = mlx_array_new();
        mlx_array v = mlx_array_new();
        mlx_reshape(&q, q_flat, q_shape, 4, S);
        mlx_reshape(&k, k_flat, k_shape, 4, S);
        mlx_reshape(&v, v_flat, k_shape, 4, S);
        mlx_array_free(q_flat); mlx_array_free(k_flat); mlx_array_free(v_flat);

        mlx_array qn = mlx_array_new();
        mlx_array kn = mlx_array_new();
        mlx_fast_rms_norm(&qn, q, l->q_norm, c->rms_norm_eps, S);
        mlx_fast_rms_norm(&kn, k, l->k_norm, c->rms_norm_eps, S);
        mlx_array_free(q); mlx_array_free(k);

        /* [B, T, heads, D] -> [B, heads, T, D] for RoPE + SDPA. */
        int perm[] = {0, 2, 1, 3};
        mlx_array qt = mlx_array_new(), kt = mlx_array_new(), vt = mlx_array_new();
        mlx_transpose_axes(&qt, qn, perm, 4, S);
        mlx_transpose_axes(&kt, kn, perm, 4, S);
        mlx_transpose_axes(&vt, v,  perm, 4, S);
        mlx_array_free(qn); mlx_array_free(kn); mlx_array_free(v);

        mlx_optional_float base = {.value = c->rope_theta, .has_value = true};
        mlx_array qr = mlx_array_new(), kr = mlx_array_new();
        mlx_fast_rope(&qr, qt, head_dim, false, base, 1.0f, 0, null_arr, S);
        mlx_fast_rope(&kr, kt, head_dim, false, base, 1.0f, 0, null_arr, S);
        mlx_array_free(qt); mlx_array_free(kt);

        mlx_array attn = mlx_array_new();
        mlx_fast_scaled_dot_product_attention(
            &attn, qr, kr, vt, scale, "",
            has_padding ? attn_mask : null_arr, null_arr, S);
        mlx_array_free(qr); mlx_array_free(kr); mlx_array_free(vt);

        mlx_array attn_t = mlx_array_new();
        mlx_transpose_axes(&attn_t, attn, perm, 4, S);
        mlx_array_free(attn);

        int flat_shape[] = {batch, max_seq, q_dim};
        mlx_array attn_flat = mlx_array_new();
        mlx_reshape(&attn_flat, attn_t, flat_shape, 3, S);
        mlx_array_free(attn_t);

        mlx_array proj = linear(attn_flat, l->wo, S);
        mlx_array_free(attn_flat);
        mlx_array x2 = mlx_array_new();
        mlx_add(&x2, x, proj, S);
        mlx_array_free(x); mlx_array_free(proj);
        x = x2;

        mlx_array xn2 = mlx_array_new();
        mlx_fast_rms_norm(&xn2, x, l->post_attn_norm, c->rms_norm_eps, S);

        mlx_array gate = linear(xn2, l->gate_proj, S);
        mlx_array up   = linear(xn2, l->up_proj, S);
        mlx_array_free(xn2);

        mlx_array gate_sig = mlx_array_new();
        mlx_sigmoid(&gate_sig, gate, S);
        mlx_array silu_gate = mlx_array_new();
        mlx_multiply(&silu_gate, gate, gate_sig, S);
        mlx_array_free(gate); mlx_array_free(gate_sig);

        mlx_array mid = mlx_array_new();
        mlx_multiply(&mid, silu_gate, up, S);
        mlx_array_free(silu_gate); mlx_array_free(up);

        mlx_array ffn = linear(mid, l->down_proj, S);
        mlx_array_free(mid);

        mlx_array x3 = mlx_array_new();
        mlx_add(&x3, x, ffn, S);
        mlx_array_free(x); mlx_array_free(ffn);
        x = x3;
    }
    return x;
}

int pplx_mlx_embed_batch(pplx_mlx_ctx_t *ctx, const pplx_input_t *inputs,
                         int batch, float *out_embeddings)
{
    if (!ctx || !inputs || batch <= 0 || !out_embeddings) return -1;

    const pplx_config_t *c = &ctx->config;
    mlx_stream S = ctx->stream;

    int hidden     = c->hidden_size;
    int max_seq    = 0;
    int has_padding = 0;

    for (int b = 0; b < batch; b++) {
        if (!inputs[b].ids || inputs[b].n_tokens <= 0) return -1;
        if (inputs[b].n_tokens > max_seq) max_seq = inputs[b].n_tokens;
    }
    if (max_seq <= 0) return -1;

    for (int b = 0; b < batch; b++) {
        if (inputs[b].n_tokens != max_seq) has_padding = 1;
    }

    size_t elems = (size_t)batch * (size_t)max_seq;
    int *padded_ids = (int *)malloc(elems * sizeof(int));
    float *pool_mask_data = (float *)malloc(elems * sizeof(float));
    float *attn_mask_data = has_padding ? (float *)malloc(elems * sizeof(float)) : NULL;
    float *len_data = (float *)malloc((size_t)batch * sizeof(float));
    if (!padded_ids || !pool_mask_data || (has_padding && !attn_mask_data) || !len_data) {
        free(padded_ids); free(pool_mask_data); free(attn_mask_data); free(len_data);
        return -1;
    }

    for (int b = 0; b < batch; b++) {
        len_data[b] = (float)inputs[b].n_tokens;
        for (int t = 0; t < max_seq; t++) {
            size_t idx = (size_t)b * max_seq + t;
            if (t < inputs[b].n_tokens) {
                int id = inputs[b].ids[t];
                if (id < 0 || id >= c->vocab_size) {
                    fprintf(stderr, "mlx: invalid token id %d at input %d position %d\n",
                            id, b, t);
                    free(padded_ids); free(pool_mask_data);
                    free(attn_mask_data); free(len_data);
                    return -1;
                }
                padded_ids[idx] = id;
                pool_mask_data[idx] = 1.0f;
                if (has_padding) attn_mask_data[idx] = 0.0f;
            } else {
                padded_ids[idx] = 0;
                pool_mask_data[idx] = 0.0f;
                if (has_padding) attn_mask_data[idx] = -1.0e9f;
            }
        }
    }

    int ids_shape[] = {batch, max_seq};
    int pool_mask_shape[] = {batch, max_seq, 1};
    int attn_mask_shape[] = {batch, 1, 1, max_seq};
    int len_shape[] = {batch, 1};

    mlx_array ids = mlx_array_new_data(padded_ids, ids_shape, 2, MLX_INT32);
    mlx_array pool_mask = mlx_array_new_data(pool_mask_data, pool_mask_shape, 3, MLX_FLOAT32);
    mlx_array attn_mask = has_padding
        ? mlx_array_new_data(attn_mask_data, attn_mask_shape, 4, MLX_FLOAT32)
        : (mlx_array){0};
    mlx_array lengths = mlx_array_new_data(len_data, len_shape, 2, MLX_FLOAT32);
    free(padded_ids); free(pool_mask_data); free(attn_mask_data); free(len_data);

    if (!arr_ok(ids) || !arr_ok(pool_mask) || (has_padding && !arr_ok(attn_mask)) ||
        !arr_ok(lengths)) {
        mlx_array_free(ids); mlx_array_free(pool_mask);
        if (has_padding) mlx_array_free(attn_mask);
        mlx_array_free(lengths);
        return -1;
    }
    if (has_padding && mlx_array_dtype(ctx->embed_tokens) == MLX_BFLOAT16) {
        mlx_array attn_mask_bf16 = mlx_array_new();
        mlx_astype(&attn_mask_bf16, attn_mask, MLX_BFLOAT16, S);
        mlx_array_free(attn_mask);
        attn_mask = attn_mask_bf16;
        if (!arr_ok(attn_mask)) {
            mlx_array_free(ids); mlx_array_free(pool_mask); mlx_array_free(lengths);
            return -1;
        }
    }

    /* 1. Embedding lookup: [B, T] -> [B, T, hidden]. */
    mlx_array x = mlx_array_new();
    mlx_take_axis(&x, ctx->embed_tokens, ids, 0, S);
    mlx_array_free(ids);

    /* 2. Transformer layers. */
    x = mlx_forward_layers(ctx, x, batch, max_seq, has_padding, attn_mask,
                           0, c->n_layers);

    /* 3. Final RMSNorm. */
    mlx_array x_normed = mlx_array_new();
    mlx_fast_rms_norm(&x_normed, x, ctx->norm, c->rms_norm_eps, S);
    mlx_array_free(x);

    /* 4. Masked mean pool over T. */
    mlx_array masked = mlx_array_new();
    mlx_multiply(&masked, x_normed, pool_mask, S);
    mlx_array_free(x_normed);
    mlx_array_free(pool_mask);

    mlx_array emb_sum = mlx_array_new();
    mlx_sum_axis(&emb_sum, masked, 1, false, S);
    mlx_array_free(masked);

    mlx_array emb = mlx_array_new();
    mlx_divide(&emb, emb_sum, lengths, S);
    mlx_array_free(emb_sum); mlx_array_free(lengths);

    if (has_padding) mlx_array_free(attn_mask);

    /* 5. Eval and copy [B, hidden] to CPU. Perplexity embeddings are
     * intentionally unnormalized; callers should use cosine similarity. */
    mlx_array emb_f32 = mlx_array_new();
    mlx_astype(&emb_f32, emb, MLX_FLOAT32, S);
    mlx_array_free(emb);

    mlx_array_eval(emb_f32);
    const float *data = mlx_array_data_float32(emb_f32);
    int rc = -1;
    if (data) {
        memcpy(out_embeddings, data, (size_t)batch * hidden * sizeof(float));
        rc = 0;
    }
    mlx_array_free(emb_f32);
    return rc;
}

int pplx_mlx_embed_into(pplx_mlx_ctx_t *ctx, const int *token_ids,
                        int n_tokens, float *out_embedding)
{
    pplx_input_t input = { token_ids, n_tokens };
    return pplx_mlx_embed_batch(ctx, &input, 1, out_embedding);
}

int pplx_mlx_forward_slice_batch(pplx_mlx_ctx_t *ctx,
                                 const pplx_input_t *inputs, int batch,
                                 const float *input_states,
                                 int layer_start, int layer_end,
                                 int apply_final_norm,
                                 float *out_states)
{
    if (!ctx || !inputs || batch <= 0 || !out_states ||
        layer_start < 0 || layer_start > layer_end ||
        layer_end > ctx->config.n_layers ||
        (layer_start == 0 && input_states) ||
        (layer_start != 0 && !input_states) ||
        (apply_final_norm && layer_end != ctx->config.n_layers))
        return -1;

    const pplx_config_t *c = &ctx->config;
    mlx_stream S = ctx->stream;
    int hidden = c->hidden_size;
    int max_seq = 0;
    int total_seq = 0;
    int has_padding = 0;
    for (int b = 0; b < batch; b++) {
        if ((layer_start == 0 && !inputs[b].ids) || inputs[b].n_tokens <= 0 ||
            total_seq > INT_MAX - inputs[b].n_tokens)
            return -1;
        total_seq += inputs[b].n_tokens;
        if (inputs[b].n_tokens > max_seq) max_seq = inputs[b].n_tokens;
    }
    for (int b = 0; b < batch; b++) {
        if (inputs[b].n_tokens != max_seq) has_padding = 1;
    }

    if ((size_t)batch > SIZE_MAX / (size_t)max_seq) return -1;
    size_t rows = (size_t)batch * (size_t)max_seq;
    if (rows > SIZE_MAX / (size_t)hidden ||
        rows * (size_t)hidden > SIZE_MAX / sizeof(float))
        return -1;
    size_t state_values = rows * (size_t)hidden;

    int *padded_ids = layer_start == 0
        ? (int *)malloc(rows * sizeof(int)) : NULL;
    float *padded_states = layer_start != 0
        ? (float *)calloc(state_values, sizeof(float)) : NULL;
    float *attn_mask_data = has_padding
        ? (float *)malloc(rows * sizeof(float)) : NULL;
    if ((layer_start == 0 && !padded_ids) ||
        (layer_start != 0 && !padded_states) ||
        (has_padding && !attn_mask_data)) {
        free(padded_ids); free(padded_states); free(attn_mask_data);
        return -1;
    }

    int packed_offset = 0;
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < max_seq; t++) {
            size_t dense_row = (size_t)b * max_seq + t;
            if (t < inputs[b].n_tokens) {
                if (layer_start == 0) {
                    int id = inputs[b].ids[t];
                    if (id < 0 || id >= c->vocab_size) {
                        fprintf(stderr, "mlx: invalid token id %d at input %d position %d\n",
                                id, b, t);
                        free(padded_ids); free(padded_states);
                        free(attn_mask_data);
                        return -1;
                    }
                    padded_ids[dense_row] = id;
                } else {
                    memcpy(padded_states + dense_row * hidden,
                           input_states + (size_t)(packed_offset + t) * hidden,
                           (size_t)hidden * sizeof(float));
                }
                if (has_padding) attn_mask_data[dense_row] = 0.0f;
            } else {
                if (layer_start == 0) padded_ids[dense_row] = 0;
                if (has_padding) attn_mask_data[dense_row] = -1.0e9f;
            }
        }
        packed_offset += inputs[b].n_tokens;
    }

    int attn_mask_shape[] = {batch, 1, 1, max_seq};
    mlx_array attn_mask = has_padding
        ? mlx_array_new_data(attn_mask_data, attn_mask_shape, 4, MLX_FLOAT32)
        : (mlx_array){0};
    free(attn_mask_data);
    if (has_padding && !arr_ok(attn_mask)) {
        free(padded_ids); free(padded_states);
        return -1;
    }
    if (has_padding && mlx_array_dtype(ctx->embed_tokens) == MLX_BFLOAT16) {
        mlx_array attn_mask_bf16 = mlx_array_new();
        mlx_astype(&attn_mask_bf16, attn_mask, MLX_BFLOAT16, S);
        mlx_array_free(attn_mask);
        attn_mask = attn_mask_bf16;
        if (!arr_ok(attn_mask)) {
            free(padded_ids); free(padded_states);
            return -1;
        }
    }

    mlx_array x;
    if (layer_start == 0) {
        int ids_shape[] = {batch, max_seq};
        mlx_array ids = mlx_array_new_data(padded_ids, ids_shape, 2, MLX_INT32);
        free(padded_ids);
        if (!arr_ok(ids)) {
            if (has_padding) mlx_array_free(attn_mask);
            return -1;
        }
        x = mlx_array_new();
        mlx_take_axis(&x, ctx->embed_tokens, ids, 0, S);
        mlx_array_free(ids);
    } else {
        int states_shape[] = {batch, max_seq, hidden};
        mlx_array states = mlx_array_new_data(padded_states, states_shape, 3,
                                              MLX_FLOAT32);
        free(padded_states);
        if (!arr_ok(states)) {
            if (has_padding) mlx_array_free(attn_mask);
            return -1;
        }
        if (mlx_array_dtype(ctx->embed_tokens) == MLX_BFLOAT16) {
            x = mlx_array_new();
            mlx_astype(&x, states, MLX_BFLOAT16, S);
            mlx_array_free(states);
        } else {
            x = states;
        }
    }

    x = mlx_forward_layers(ctx, x, batch, max_seq, has_padding, attn_mask,
                           layer_start, layer_end);
    if (has_padding) mlx_array_free(attn_mask);

    if (apply_final_norm) {
        mlx_array x_normed = mlx_array_new();
        mlx_fast_rms_norm(&x_normed, x, ctx->norm, c->rms_norm_eps, S);
        mlx_array_free(x);
        x = x_normed;
    }

    mlx_array x_f32 = mlx_array_new();
    mlx_astype(&x_f32, x, MLX_FLOAT32, S);
    mlx_array_free(x);
    mlx_array_eval(x_f32);

    const float *data = mlx_array_data_float32(x_f32);
    if (!data) {
        mlx_array_free(x_f32);
        return -1;
    }
    packed_offset = 0;
    for (int b = 0; b < batch; b++) {
        memcpy(out_states + (size_t)packed_offset * hidden,
               data + (size_t)b * max_seq * hidden,
               (size_t)inputs[b].n_tokens * hidden * sizeof(float));
        packed_offset += inputs[b].n_tokens;
    }
    mlx_array_free(x_f32);
    return 0;
}

static int pplx_mlx_forward_into(pplx_mlx_ctx_t *ctx, const int *token_ids,
                                 int n_tokens, float *out_states)
{
    pplx_input_t input = { token_ids, n_tokens };
    return pplx_mlx_forward_slice_batch(ctx, &input, 1, NULL, 0,
                                        ctx->config.n_layers, 1, out_states);
}

int pplx_mlx_embed_spans(pplx_mlx_ctx_t *ctx, const int *token_ids,
                         int n_tokens, const pplx_span_t *spans,
                         int n_spans, float *out_embeddings)
{
    if (!ctx || !token_ids || n_tokens <= 0 || !spans || n_spans <= 0 ||
        !out_embeddings)
        return -1;

    int hidden = ctx->config.hidden_size;
    float *states = (float *)malloc((size_t)n_tokens * hidden * sizeof(float));
    if (!states) return -1;

    int rc = pplx_mlx_forward_into(ctx, token_ids, n_tokens, states);
    if (rc == 0)
        rc = pplx_pool_spans(&ctx->config, states, n_tokens, spans, n_spans,
                             out_embeddings);
    free(states);
    return rc;
}

float *pplx_mlx_embed(pplx_mlx_ctx_t *ctx, const int *token_ids, int n_tokens)
{
    if (!ctx || !token_ids || n_tokens <= 0) return NULL;

    int hidden = ctx->config.hidden_size;
    float *out = (float *)malloc((size_t)hidden * sizeof(float));
    if (!out) return NULL;

    if (pplx_mlx_embed_into(ctx, token_ids, n_tokens, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}
