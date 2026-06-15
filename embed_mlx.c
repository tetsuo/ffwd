#include "embed_mlx.h"
#include "embed_config.h"
#include "embed.h"
#include "qwen_safetensors.h"

#include <mlx/c/mlx.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h>

/* ========================================================================
 * Per-layer MLX weight arrays
 * ======================================================================== */

typedef struct {
    mlx_array w;
    mlx_array wt;
    mlx_array scales;
    mlx_array biases;
    int quantized;
    int bits;
    int group_size;
} mlx_linear_t;

typedef struct {
    mlx_linear_t wq, wk, wv, wo;
    mlx_array q_bias, k_bias, v_bias;
    mlx_array q_norm, k_norm;
    mlx_array input_norm, post_attn_norm;
    mlx_linear_t gate_proj, up_proj, down_proj;
} mlx_layer_t;

struct embed_mlx_ctx {
    embed_config_t config;
    mlx_array embed_tokens;
    mlx_layer_t *layers; /* heap-allocated [n_layers] */
    mlx_array norm;
    mlx_stream stream;
    multi_safetensors_t *ms;
    int layer_start; /* loaded transformer range, inclusive */
    int layer_end;   /* loaded transformer range, exclusive */
    int quantize_bits;
    int quantize_group_size;
    mlx_dtype weight_dtype;
};

struct embed_mlx_late_ctx {
    embed_mlx_ctx_t *base;
    multi_safetensors_t *dense_ms;
    mlx_linear_t projection;
    int token_dim;
};

struct embed_mlx_late_vectors {
    embed_mlx_late_ctx_t *owner;
    mlx_array vectors; /* [tokens, dim] */
    int tokens;
    int dim;
};

/* ========================================================================
 * Helpers
 * ======================================================================== */

static int arr_ok(mlx_array a) { return a.ctx != NULL; }

static int mlx_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a)
        return -1;
    *out = a * b;
    return 0;
}

static mlx_dtype mlx_weight_dtype(const embed_mlx_ctx_t *ctx) { return ctx->weight_dtype; }

static size_t safetensor_dtype_size(safetensor_dtype_t dtype) {
    switch (dtype) {
    case DTYPE_F32:
        return sizeof(float);
    case DTYPE_BF16:
        return sizeof(uint16_t);
    default:
        return 0;
    }
}

static int mlx_tensor_has_supported_shape(const safetensors_file_t *sf,
                                          const safetensor_t *t,
                                          const char *name,
                                          const int *shape,
                                          int ndim) {
    size_t elem_size = safetensor_dtype_size(t->dtype);
    if (elem_size == 0) {
        fprintf(stderr, "mlx: expected F32 or BF16 for %s\n", name);
        return 0;
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "mlx: bad rank for %s: got %d, expected %d\n", name, t->ndim, ndim);
        return 0;
    }

    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0 || t->shape[i] != shape[i]) {
            fprintf(stderr,
                    "mlx: bad shape for %s at dim %d: "
                    "got %lld, expected %d\n",
                    name, i, (long long)t->shape[i], shape[i]);
            return 0;
        }
        if (mlx_mul_size(numel, (size_t)shape[i], &numel) != 0) {
            fprintf(stderr, "mlx: shape too large for %s\n", name);
            return 0;
        }
    }

    size_t bytes;
    if (mlx_mul_size(numel, elem_size, &bytes) != 0) {
        fprintf(stderr, "mlx: tensor data too large for %s\n", name);
        return 0;
    }
    if (t->data_size != bytes) {
        fprintf(stderr, "mlx: bad data size for %s: got %zu, expected %zu\n", name, t->data_size,
                bytes);
        return 0;
    }

    if (sf->header_size > SIZE_MAX - 8)
        return 0;
    size_t data_start = 8 + sf->header_size;
    if (data_start > sf->file_size || t->data_offset > sf->file_size - data_start ||
        t->data_size > sf->file_size - data_start - t->data_offset) {
        fprintf(stderr, "mlx: tensor data out of bounds for %s\n", name);
        return 0;
    }
    return 1;
}

static mlx_array
load_tensor(multi_safetensors_t *ms, const char *name, const int *shape, int ndim) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "mlx: tensor not found: %s\n", name);
        return (mlx_array){0};
    }
    if (!mlx_tensor_has_supported_shape(sf, t, name, shape, ndim))
        return (mlx_array){0};
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

static mlx_linear_t
load_linear_tensor(multi_safetensors_t *ms, const char *name, const int *shape, int ndim) {
    mlx_linear_t l = {0};
    l.w = load_tensor(ms, name, shape, ndim);
    return l;
}

static int linear_ok(mlx_linear_t l) { return arr_ok(l.w) || arr_ok(l.wt); }

static void free_linear(mlx_linear_t *l) {
    mlx_array_free(l->w);
    mlx_array_free(l->wt);
    mlx_array_free(l->scales);
    mlx_array_free(l->biases);
    memset(l, 0, sizeof(*l));
}

static int prepare_linear_transpose(mlx_linear_t *l, mlx_stream s) {
    if (!l || l->quantized || !arr_ok(l->w) || arr_ok(l->wt))
        return 0;

    mlx_array transposed = mlx_array_new();
    mlx_transpose(&transposed, l->w, s);
    if (!arr_ok(transposed)) {
        mlx_array_free(transposed);
        fprintf(stderr, "mlx: failed to prepare transposed linear weight\n");
        return -1;
    }

    l->wt = mlx_array_new();
    mlx_contiguous(&l->wt, transposed, false, s);
    mlx_array_free(transposed);
    if (!arr_ok(l->wt)) {
        mlx_array_free(l->wt);
        l->wt = (mlx_array){0};
        fprintf(stderr, "mlx: failed to materialize transposed linear weight\n");
        return -1;
    }
    mlx_array_eval(l->wt);
    mlx_synchronize(s);
    mlx_array_free(l->w);
    l->w = (mlx_array){0};
    return 0;
}

static int quantize_linear(mlx_linear_t *l, int bits, int group_size, mlx_stream s) {
    if (!l || !arr_ok(l->w) || bits == 0)
        return 0;
    if (bits != 8) {
        fprintf(stderr, "mlx: quantized weights support only 8 bits\n");
        return -1;
    }
    if (group_size <= 0)
        group_size = 64;

    mlx_optional_int gs = {.value = group_size, .has_value = true};
    mlx_optional_int qb = {.value = bits, .has_value = true};
    mlx_array null_arr = (mlx_array){0};
    mlx_vector_array q = mlx_vector_array_new();
    mlx_quantize(&q, l->w, gs, qb, "affine", null_arr, s);
    if (!q.ctx || mlx_vector_array_size(q) < 2) {
        mlx_vector_array_free(q);
        fprintf(stderr, "mlx: quantize failed\n");
        return -1;
    }

    mlx_array qw = (mlx_array){0};
    mlx_array scales = (mlx_array){0};
    mlx_array biases = (mlx_array){0};
    mlx_vector_array_get(&qw, q, 0);
    mlx_vector_array_get(&scales, q, 1);
    if (mlx_vector_array_size(q) >= 3)
        mlx_vector_array_get(&biases, q, 2);
    mlx_vector_array_free(q);

    if (!arr_ok(qw) || !arr_ok(scales)) {
        mlx_array_free(qw);
        mlx_array_free(scales);
        mlx_array_free(biases);
        fprintf(stderr, "mlx: quantize returned invalid arrays\n");
        return -1;
    }
    mlx_array_eval(qw);
    mlx_array_eval(scales);
    if (arr_ok(biases))
        mlx_array_eval(biases);
    mlx_synchronize(s);

    mlx_array_free(l->w);
    mlx_array_free(l->wt);
    l->w = qw;
    l->wt = (mlx_array){0};
    l->scales = scales;
    l->biases = biases;
    l->quantized = 1;
    l->bits = bits;
    l->group_size = group_size;
    return 0;
}

/* y = x @ W^T */
static mlx_array linear(mlx_array x, const mlx_linear_t *W, mlx_stream s) {
    if (W->quantized) {
        if (!arr_ok(W->w))
            return (mlx_array){0};
        mlx_array res = mlx_array_new();
        mlx_optional_int gs = {.value = W->group_size, .has_value = true};
        mlx_optional_int bits = {.value = W->bits, .has_value = true};
        mlx_array null_arr = (mlx_array){0};
        mlx_quantized_matmul(&res, x, W->w, W->scales, arr_ok(W->biases) ? W->biases : null_arr,
                             true, gs, bits, "affine", s);
        return res;
    }
    mlx_array Wt;
    int free_wt = 0;
    if (arr_ok(W->wt)) {
        Wt = W->wt;
    } else if (arr_ok(W->w)) {
        Wt = mlx_array_new();
        mlx_transpose(&Wt, W->w, s);
        free_wt = 1;
    } else {
        return (mlx_array){0};
    }
    mlx_array res = mlx_array_new();
    if (mlx_array_dtype(Wt) == MLX_BFLOAT16 && mlx_array_dtype(x) == MLX_FLOAT32) {
        /* F32 activations with BF16 weights: upcast the weight operand and run
         * the matmul in F32 (so accumulation is F32). Only the weights are
         * BF16-rounded; the residual stream stays F32 (see mlx_forward_layers),
         * matching the CPU/CUDA accuracy while keeping BF16 weight storage. */
        mlx_array Wf = mlx_array_new();
        mlx_astype(&Wf, Wt, MLX_FLOAT32, s);
        mlx_matmul(&res, x, Wf, s);
        mlx_array_free(Wf);
    } else {
        mlx_matmul(&res, x, Wt, s);
    }
    if (free_wt)
        mlx_array_free(Wt);
    return res;
}

const embed_config_t *embed_mlx_config(const embed_mlx_ctx_t *ctx) {
    return ctx ? &ctx->config : NULL;
}

/* Upcast a tensor to F32 in place (no-op if already F32 or empty). Used for the
 * tiny RMSNorm vectors so non-quantized BF16 models can run norms and the
 * residual stream in F32 while keeping the large matmul weights in BF16. */
static void upcast_f32_inplace(mlx_array *a, mlx_stream s) {
    if (!a || !arr_ok(*a) || mlx_array_dtype(*a) == MLX_FLOAT32)
        return;
    mlx_array f = mlx_array_new();
    mlx_astype(&f, *a, MLX_FLOAT32, s);
    mlx_array_free(*a);
    *a = f;
}

/* ========================================================================
 * Load
 * ======================================================================== */

static int mlx_model_dir_has_file(const char *model_dir, const char *rel) {
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

static int mlx_model_dir_has_late_projection(const char *model_dir) {
    return mlx_model_dir_has_file(model_dir, "1_Dense/config.json") &&
           mlx_model_dir_has_file(model_dir, "1_Dense/model.safetensors");
}

static embed_mlx_options_t normalize_mlx_options(const embed_mlx_options_t *opts) {
    embed_mlx_options_t out = {0};
    if (opts)
        out = *opts;
    if (out.quantize_bits != 0 && out.quantize_bits != 8) {
        fprintf(stderr, "mlx: --mlx-quantize-bits supports only 8\n");
        out.quantize_bits = -1;
    }
    if (out.quantize_bits && out.quantize_group_size <= 0)
        out.quantize_group_size = 64;
    return out;
}

static int quantize_layer(mlx_layer_t *l, int bits, int group_size, mlx_stream s) {
    if (bits == 0)
        return 0;
    return quantize_linear(&l->wq, bits, group_size, s) ||
           quantize_linear(&l->wk, bits, group_size, s) ||
           quantize_linear(&l->wv, bits, group_size, s) ||
           quantize_linear(&l->wo, bits, group_size, s) ||
           quantize_linear(&l->gate_proj, bits, group_size, s) ||
           quantize_linear(&l->up_proj, bits, group_size, s) ||
           quantize_linear(&l->down_proj, bits, group_size, s);
}

static int prepare_layer_transposes(mlx_layer_t *l, mlx_stream s) {
    return prepare_linear_transpose(&l->wq, s) || prepare_linear_transpose(&l->wk, s) ||
           prepare_linear_transpose(&l->wv, s) || prepare_linear_transpose(&l->wo, s) ||
           prepare_linear_transpose(&l->gate_proj, s) || prepare_linear_transpose(&l->up_proj, s) ||
           prepare_linear_transpose(&l->down_proj, s);
}

static embed_mlx_ctx_t *mlx_load_range_ex(const char *model_dir,
                                          int layer_start,
                                          int layer_end,
                                          const embed_mlx_options_t *opts,
                                          int allow_late) {
    embed_mlx_options_t options = normalize_mlx_options(opts);
    if (options.quantize_bits < 0)
        return NULL;
    if (!allow_late && mlx_model_dir_has_late_projection(model_dir)) {
        fprintf(stderr, "mlx: late-interaction models require token-level "
                        "MaxSim support and are not valid pooled embedding models "
                        "yet\n");
        return NULL;
    }

    embed_mlx_ctx_t *ctx = calloc(1, sizeof(embed_mlx_ctx_t));
    if (!ctx)
        return NULL;
    ctx->quantize_bits = options.quantize_bits;
    ctx->quantize_group_size = options.quantize_group_size;

    if (embed_config_parse(&ctx->config, model_dir) != 0) {
        free(ctx);
        return NULL;
    }

    const embed_config_t *c = &ctx->config;
    if (layer_end == -1)
        layer_end = c->n_layers;
    if (layer_start < 0 || layer_start >= layer_end || layer_end > c->n_layers) {
        fprintf(stderr, "mlx: invalid layer range [%d, %d) for %d-layer model\n", layer_start,
                layer_end, c->n_layers);
        free(ctx);
        return NULL;
    }
    ctx->layer_start = layer_start;
    ctx->layer_end = layer_end;

    ctx->ms = multi_safetensors_open(model_dir);
    if (!ctx->ms) {
        fprintf(stderr, "mlx: failed to open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }

    const char *weight_prefix = multi_safetensors_weight_prefix(ctx->ms, "embed_tokens.weight");
    if (!weight_prefix) {
        fprintf(stderr, "mlx: neither embed_tokens.weight nor "
                        "model.embed_tokens.weight was found\n");
        goto fail;
    }

    int h = c->hidden_size, qd = c->q_dim, kvd = c->kv_dim;
    int inter = c->intermediate_size, hd = c->head_dim;
    char name[256];

    /* Only the first stage needs token embeddings. */
    if (layer_start == 0) {
        int emb_shape[] = {c->vocab_size, h};
        snprintf(name, sizeof(name), "%sembed_tokens.weight", weight_prefix);
        ctx->embed_tokens = load_tensor(ctx->ms, name, emb_shape, 2);
        if (!arr_ok(ctx->embed_tokens))
            goto fail;
    }

    /* Layers */
    ctx->layers = calloc(c->n_layers, sizeof(mlx_layer_t));
    if (!ctx->layers)
        goto fail;

    for (int i = layer_start; i < layer_end; i++) {
        mlx_layer_t *l = &ctx->layers[i];

#define LD(fld, fmt, ...)                                                           \
    do {                                                                            \
        snprintf(name, sizeof(name), fmt, weight_prefix, i);                        \
        int sh[] = {__VA_ARGS__};                                                   \
        l->fld = load_linear_tensor(ctx->ms, name, sh, sizeof(sh) / sizeof(sh[0])); \
        if (!linear_ok(l->fld))                                                     \
            goto fail;                                                              \
    } while (0)

        LD(wq, "%slayers.%d.self_attn.q_proj.weight", qd, h);
        LD(wk, "%slayers.%d.self_attn.k_proj.weight", kvd, h);
        LD(wv, "%slayers.%d.self_attn.v_proj.weight", kvd, h);
        LD(wo, "%slayers.%d.self_attn.o_proj.weight", h, qd);
#undef LD

#define LD(fld, fmt, ...)                                                    \
    do {                                                                     \
        snprintf(name, sizeof(name), fmt, weight_prefix, i);                 \
        int sh[] = {__VA_ARGS__};                                            \
        l->fld = load_tensor(ctx->ms, name, sh, sizeof(sh) / sizeof(sh[0])); \
        if (!arr_ok(l->fld))                                                 \
            goto fail;                                                       \
    } while (0)

        if (c->qkv_bias) {
            LD(q_bias, "%slayers.%d.self_attn.q_proj.bias", qd);
            LD(k_bias, "%slayers.%d.self_attn.k_proj.bias", kvd);
            LD(v_bias, "%slayers.%d.self_attn.v_proj.bias", kvd);
        }
        if (c->qk_norm) {
            LD(q_norm, "%slayers.%d.self_attn.q_norm.weight", hd);
            LD(k_norm, "%slayers.%d.self_attn.k_norm.weight", hd);
        }
        LD(input_norm, "%slayers.%d.input_layernorm.weight", h);
        LD(post_attn_norm, "%slayers.%d.post_attention_layernorm.weight", h);
#undef LD

#define LD(fld, fmt, ...)                                                           \
    do {                                                                            \
        snprintf(name, sizeof(name), fmt, weight_prefix, i);                        \
        int sh[] = {__VA_ARGS__};                                                   \
        l->fld = load_linear_tensor(ctx->ms, name, sh, sizeof(sh) / sizeof(sh[0])); \
        if (!linear_ok(l->fld))                                                     \
            goto fail;                                                              \
    } while (0)

        LD(gate_proj, "%slayers.%d.mlp.gate_proj.weight", inter, h);
        LD(up_proj, "%slayers.%d.mlp.up_proj.weight", inter, h);
        LD(down_proj, "%slayers.%d.mlp.down_proj.weight", h, inter);
#undef LD

        if (embed_verbose >= 2)
            fprintf(stderr, "  mlx layer %d loaded\n", i);
    }

    if (layer_end == c->n_layers) {
        int ns[] = {h};
        snprintf(name, sizeof(name), "%snorm.weight", weight_prefix);
        ctx->norm = load_tensor(ctx->ms, name, ns, 1);
        if (!arr_ok(ctx->norm))
            goto fail;
    }

    ctx->stream = mlx_default_gpu_stream_new();
    ctx->weight_dtype = mlx_array_dtype(ctx->layers[layer_start].wq.w);
    /* Non-quantized BF16: run RMSNorm and the residual stream in F32. Upcast the
     * tiny norm vectors so they match the F32 activations in mlx_forward_layers;
     * the large matmul weights stay BF16. Q8 keeps its existing BF16 norms. */
    if (ctx->weight_dtype == MLX_BFLOAT16 && !ctx->quantize_bits) {
        for (int i = layer_start; i < layer_end; i++) {
            mlx_layer_t *l = &ctx->layers[i];
            upcast_f32_inplace(&l->q_bias, ctx->stream);
            upcast_f32_inplace(&l->k_bias, ctx->stream);
            upcast_f32_inplace(&l->v_bias, ctx->stream);
            upcast_f32_inplace(&l->q_norm, ctx->stream);
            upcast_f32_inplace(&l->k_norm, ctx->stream);
            upcast_f32_inplace(&l->input_norm, ctx->stream);
            upcast_f32_inplace(&l->post_attn_norm, ctx->stream);
        }
        upcast_f32_inplace(&ctx->norm, ctx->stream);
    }
    if (ctx->quantize_bits) {
        for (int i = layer_start; i < layer_end; i++) {
            if (quantize_layer(&ctx->layers[i], ctx->quantize_bits, ctx->quantize_group_size,
                               ctx->stream) != 0)
                goto fail;
        }
        mlx_clear_cache();
        if (embed_verbose >= 1)
            fprintf(stderr,
                    "mlx: quantized linear weights to %d-bit "
                    "(group_size=%d)\n",
                    ctx->quantize_bits, ctx->quantize_group_size);
    } else {
        for (int i = layer_start; i < layer_end; i++) {
            if (prepare_layer_transposes(&ctx->layers[i], ctx->stream) != 0)
                goto fail;
        }
    }

    if (embed_verbose >= 2) {
        size_t active = 0, cache = 0, peak = 0;
        mlx_get_active_memory(&active);
        mlx_get_cache_memory(&cache);
        mlx_get_peak_memory(&peak);
        fprintf(stderr,
                "mlx: memory active=%.2f GiB cache=%.2f GiB "
                "peak=%.2f GiB\n",
                (double)active / (1024.0 * 1024.0 * 1024.0),
                (double)cache / (1024.0 * 1024.0 * 1024.0),
                (double)peak / (1024.0 * 1024.0 * 1024.0));
    }

    if (embed_verbose >= 1)
        fprintf(stderr, "mlx: layers [%d, %d) loaded (%d-dim)\n", layer_start, layer_end, h);

    return ctx;

fail:
    embed_mlx_free(ctx);
    return NULL;
}

embed_mlx_ctx_t *embed_mlx_load(const char *model_dir) {
    return mlx_load_range_ex(model_dir, 0, -1, NULL, 0);
}

embed_mlx_ctx_t *embed_mlx_load_with_options(const char *model_dir,
                                             const embed_mlx_options_t *opts) {
    return mlx_load_range_ex(model_dir, 0, -1, opts, 0);
}

/* ========================================================================
 * Free
 * ======================================================================== */

static void free_mlx_layer(mlx_layer_t *l) {
    free_linear(&l->wq);
    free_linear(&l->wk);
    free_linear(&l->wv);
    free_linear(&l->wo);
    mlx_array_free(l->q_bias);
    mlx_array_free(l->k_bias);
    mlx_array_free(l->v_bias);
    mlx_array_free(l->q_norm);
    mlx_array_free(l->k_norm);
    mlx_array_free(l->input_norm);
    mlx_array_free(l->post_attn_norm);
    free_linear(&l->gate_proj);
    free_linear(&l->up_proj);
    free_linear(&l->down_proj);
}

void embed_mlx_free(embed_mlx_ctx_t *ctx) {
    if (!ctx)
        return;
    mlx_array_free(ctx->embed_tokens);
    if (ctx->layers) {
        for (int i = 0; i < ctx->config.n_layers; i++)
            free_mlx_layer(&ctx->layers[i]);
        free(ctx->layers);
    }
    mlx_array_free(ctx->norm);
    if (ctx->stream.ctx)
        mlx_stream_free(ctx->stream);
    if (ctx->ms)
        multi_safetensors_close(ctx->ms);
    free(ctx);
}

embed_mlx_late_ctx_t *embed_mlx_late_load_with_options(const char *model_dir,
                                                       const embed_mlx_options_t *opts) {
    if (!mlx_model_dir_has_late_projection(model_dir)) {
        fprintf(stderr, "mlx late: missing 1_Dense projection\n");
        return NULL;
    }

    embed_mlx_late_ctx_t *late = calloc(1, sizeof(*late));
    if (!late)
        return NULL;

    late->base = mlx_load_range_ex(model_dir, 0, -1, opts, 1);
    if (!late->base)
        goto fail;

    char dense_dir[1024];
    int n = snprintf(dense_dir, sizeof(dense_dir), "%s/1_Dense", model_dir);
    if (n < 0 || (size_t)n >= sizeof(dense_dir)) {
        fprintf(stderr, "mlx late: model path too long\n");
        goto fail;
    }

    late->dense_ms = multi_safetensors_open(dense_dir);
    if (!late->dense_ms) {
        fprintf(stderr, "mlx late: failed to open safetensors in %s\n", dense_dir);
        goto fail;
    }

    const safetensor_t *t = multi_safetensors_find(late->dense_ms, "linear.weight", NULL);
    if (!t) {
        fprintf(stderr, "mlx late: missing linear.weight\n");
        goto fail;
    }
    if (t->ndim != 2 || t->shape[0] <= 0 || t->shape[0] > INT_MAX ||
        t->shape[1] != late->base->config.hidden_size) {
        fprintf(stderr, "mlx late: bad projection shape\n");
        goto fail;
    }

    int shape[] = {(int)t->shape[0], late->base->config.hidden_size};
    late->projection = load_linear_tensor(late->dense_ms, "linear.weight", shape, 2);
    if (!linear_ok(late->projection))
        goto fail;
    if (prepare_linear_transpose(&late->projection, late->base->stream) != 0)
        goto fail;
    late->token_dim = shape[0];

    if (embed_verbose >= 1)
        fprintf(stderr, "mlx late: token projection loaded (%d-dim)\n", late->token_dim);
    return late;

fail:
    embed_mlx_late_free(late);
    return NULL;
}

embed_mlx_late_ctx_t *embed_mlx_late_load(const char *model_dir) {
    return embed_mlx_late_load_with_options(model_dir, NULL);
}

void embed_mlx_late_free(embed_mlx_late_ctx_t *ctx) {
    if (!ctx)
        return;
    free_linear(&ctx->projection);
    if (ctx->dense_ms)
        multi_safetensors_close(ctx->dense_ms);
    embed_mlx_free(ctx->base);
    free(ctx);
}

const embed_config_t *embed_mlx_late_config(const embed_mlx_late_ctx_t *ctx) {
    return ctx && ctx->base ? &ctx->base->config : NULL;
}

int embed_mlx_late_token_dim(const embed_mlx_late_ctx_t *ctx) { return ctx ? ctx->token_dim : 0; }

/* ========================================================================
 * Forward pass
 * ======================================================================== */

static mlx_array mlx_forward_layers(embed_mlx_ctx_t *ctx,
                                    mlx_array x,
                                    int batch,
                                    int max_seq,
                                    int has_padding,
                                    mlx_array attn_mask,
                                    int layer_start,
                                    int layer_end) {
    const embed_config_t *c = &ctx->config;
    mlx_stream S = ctx->stream;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int q_dim = c->q_dim;
    float scale = 1.0f / sqrtf((float)head_dim);
    mlx_array null_arr = (mlx_array){0};
    const char *mask_mode = c->attention_mode == EMBED_ATTENTION_CAUSAL ? "causal" : "";

    /* For non-quantized BF16 weights, run the residual stream, norms, and
     * attention in F32; matmuls downcast their operands to BF16 internally
     * (see linear). This keeps weight storage at BF16 while avoiding the
     * cross-layer accumulation error of a fully BF16 forward. F32 and quantized
     * models are unaffected. */
    mlx_array local_mask = attn_mask;
    int free_mask = 0;
    int mixed_bf16 =
        mlx_weight_dtype(ctx) == MLX_BFLOAT16 && !ctx->layers[layer_start].wq.quantized;
    if (mixed_bf16) {
        if (mlx_array_dtype(x) != MLX_FLOAT32) {
            mlx_array xf = mlx_array_new();
            mlx_astype(&xf, x, MLX_FLOAT32, S);
            mlx_array_free(x);
            x = xf;
        }
        if (has_padding && arr_ok(attn_mask) && mlx_array_dtype(attn_mask) != MLX_FLOAT32) {
            local_mask = mlx_array_new();
            mlx_astype(&local_mask, attn_mask, MLX_FLOAT32, S);
            free_mask = 1;
        }
    }

    for (int layer = layer_start; layer < layer_end; layer++) {
        mlx_layer_t *l = &ctx->layers[layer];

        mlx_array xn = mlx_array_new();
        mlx_fast_rms_norm(&xn, x, l->input_norm, c->rms_norm_eps, S);

        mlx_array q_flat = linear(xn, &l->wq, S);
        mlx_array k_flat = linear(xn, &l->wk, S);
        mlx_array v_flat = linear(xn, &l->wv, S);
        mlx_array_free(xn);
        if (c->qkv_bias) {
            mlx_array biased = mlx_array_new();
            mlx_add(&biased, q_flat, l->q_bias, S);
            mlx_array_free(q_flat);
            q_flat = biased;
            biased = mlx_array_new();
            mlx_add(&biased, k_flat, l->k_bias, S);
            mlx_array_free(k_flat);
            k_flat = biased;
            biased = mlx_array_new();
            mlx_add(&biased, v_flat, l->v_bias, S);
            mlx_array_free(v_flat);
            v_flat = biased;
        }

        int q_shape[] = {batch, max_seq, n_heads, head_dim};
        int k_shape[] = {batch, max_seq, n_kv_heads, head_dim};
        mlx_array q = mlx_array_new();
        mlx_array k = mlx_array_new();
        mlx_array v = mlx_array_new();
        mlx_reshape(&q, q_flat, q_shape, 4, S);
        mlx_reshape(&k, k_flat, k_shape, 4, S);
        mlx_reshape(&v, v_flat, k_shape, 4, S);
        mlx_array_free(q_flat);
        mlx_array_free(k_flat);
        mlx_array_free(v_flat);

        mlx_array qn = q;
        mlx_array kn = k;
        if (c->qk_norm) {
            qn = mlx_array_new();
            kn = mlx_array_new();
            mlx_fast_rms_norm(&qn, q, l->q_norm, c->rms_norm_eps, S);
            mlx_fast_rms_norm(&kn, k, l->k_norm, c->rms_norm_eps, S);
            mlx_array_free(q);
            mlx_array_free(k);
        }

        /* [B, T, heads, D] -> [B, heads, T, D] for RoPE + SDPA. */
        int perm[] = {0, 2, 1, 3};
        mlx_array qt = mlx_array_new(), kt = mlx_array_new(), vt = mlx_array_new();
        mlx_transpose_axes(&qt, qn, perm, 4, S);
        mlx_transpose_axes(&kt, kn, perm, 4, S);
        mlx_transpose_axes(&vt, v, perm, 4, S);
        mlx_array_free(qn);
        mlx_array_free(kn);
        mlx_array_free(v);

        mlx_optional_float base = {.value = c->rope_theta, .has_value = true};
        mlx_array qr = mlx_array_new(), kr = mlx_array_new();
        mlx_fast_rope(&qr, qt, head_dim, false, base, 1.0f, 0, null_arr, S);
        mlx_fast_rope(&kr, kt, head_dim, false, base, 1.0f, 0, null_arr, S);
        mlx_array_free(qt);
        mlx_array_free(kt);

        mlx_array attn = mlx_array_new();
        mlx_fast_scaled_dot_product_attention(&attn, qr, kr, vt, scale, mask_mode,
                                              has_padding ? local_mask : null_arr, null_arr, S);
        mlx_array_free(qr);
        mlx_array_free(kr);
        mlx_array_free(vt);

        mlx_array attn_t = mlx_array_new();
        mlx_transpose_axes(&attn_t, attn, perm, 4, S);
        mlx_array_free(attn);

        int flat_shape[] = {batch, max_seq, q_dim};
        mlx_array attn_flat = mlx_array_new();
        mlx_reshape(&attn_flat, attn_t, flat_shape, 3, S);
        mlx_array_free(attn_t);

        mlx_array proj = linear(attn_flat, &l->wo, S);
        mlx_array_free(attn_flat);
        mlx_array x2 = mlx_array_new();
        mlx_add(&x2, x, proj, S);
        mlx_array_free(x);
        mlx_array_free(proj);
        x = x2;

        mlx_array xn2 = mlx_array_new();
        mlx_fast_rms_norm(&xn2, x, l->post_attn_norm, c->rms_norm_eps, S);

        mlx_array gate = linear(xn2, &l->gate_proj, S);
        mlx_array up = linear(xn2, &l->up_proj, S);
        mlx_array_free(xn2);

        mlx_array gate_sig = mlx_array_new();
        mlx_sigmoid(&gate_sig, gate, S);
        mlx_array silu_gate = mlx_array_new();
        mlx_multiply(&silu_gate, gate, gate_sig, S);
        mlx_array_free(gate);
        mlx_array_free(gate_sig);

        mlx_array mid = mlx_array_new();
        mlx_multiply(&mid, silu_gate, up, S);
        mlx_array_free(silu_gate);
        mlx_array_free(up);

        mlx_array ffn = linear(mid, &l->down_proj, S);
        mlx_array_free(mid);

        mlx_array x3 = mlx_array_new();
        mlx_add(&x3, x, ffn, S);
        mlx_array_free(x);
        mlx_array_free(ffn);
        x = x3;
    }
    if (free_mask)
        mlx_array_free(local_mask);
    return x;
}

embed_mlx_late_vectors_t *embed_mlx_late_encode_tokens_device(embed_mlx_late_ctx_t *ctx,
                                                              const int *token_ids,
                                                              int n_tokens,
                                                              int normalize) {
    if (!ctx || !ctx->base || !token_ids || n_tokens <= 0 || ctx->token_dim <= 0 ||
        !linear_ok(ctx->projection))
        return NULL;

    embed_mlx_ctx_t *base = ctx->base;
    if (base->layer_start != 0 || base->layer_end != base->config.n_layers ||
        !arr_ok(base->embed_tokens) || !arr_ok(base->norm))
        return NULL;

    const embed_config_t *c = &base->config;
    mlx_stream S = base->stream;
    for (int i = 0; i < n_tokens; i++) {
        if (token_ids[i] < 0 || token_ids[i] >= c->vocab_size) {
            fprintf(stderr, "mlx late: invalid token id %d at position %d\n", token_ids[i], i);
            return NULL;
        }
    }

    size_t id_bytes;
    if (mlx_mul_size((size_t)n_tokens, sizeof(int), &id_bytes) != 0)
        return NULL;

    int *ids_data = (int *)malloc(id_bytes);
    if (!ids_data)
        return NULL;
    memcpy(ids_data, token_ids, id_bytes);

    int ids_shape[] = {1, n_tokens};
    mlx_array ids = mlx_array_new_data(ids_data, ids_shape, 2, MLX_INT32);
    free(ids_data);
    if (!arr_ok(ids))
        return NULL;

    embed_mlx_late_vectors_t *result = NULL;
    mlx_array x = mlx_array_new();
    mlx_array x_normed = (mlx_array){0};
    mlx_array projected = (mlx_array){0};
    mlx_array reshaped = (mlx_array){0};

    mlx_take_axis(&x, base->embed_tokens, ids, 0, S);
    mlx_array_free(ids);
    if (!arr_ok(x))
        goto cleanup;

    x = mlx_forward_layers(base, x, 1, n_tokens, 0, (mlx_array){0}, 0, c->n_layers);
    if (!arr_ok(x))
        goto cleanup;

    x_normed = mlx_array_new();
    mlx_fast_rms_norm(&x_normed, x, base->norm, c->rms_norm_eps, S);
    mlx_array_free(x);
    x = (mlx_array){0};
    if (!arr_ok(x_normed))
        goto cleanup;

    projected = linear(x_normed, &ctx->projection, S);
    mlx_array_free(x_normed);
    x_normed = (mlx_array){0};
    if (!arr_ok(projected))
        goto cleanup;

    if (normalize) {
        mlx_array sq = mlx_array_new();
        mlx_square(&sq, projected, S);
        if (!arr_ok(sq)) {
            mlx_array_free(sq);
            goto cleanup;
        }

        mlx_array norm2 = mlx_array_new();
        mlx_sum_axis(&norm2, sq, 2, true, S);
        mlx_array_free(sq);
        if (!arr_ok(norm2)) {
            mlx_array_free(norm2);
            goto cleanup;
        }

        mlx_array inv_norm = mlx_array_new();
        mlx_rsqrt(&inv_norm, norm2, S);
        mlx_array_free(norm2);
        if (!arr_ok(inv_norm)) {
            mlx_array_free(inv_norm);
            goto cleanup;
        }

        mlx_array normalized = mlx_array_new();
        mlx_multiply(&normalized, projected, inv_norm, S);
        mlx_array_free(inv_norm);
        mlx_array_free(projected);
        projected = normalized;
        if (!arr_ok(projected))
            goto cleanup;
    }

    int vec_shape[] = {n_tokens, ctx->token_dim};
    reshaped = mlx_array_new();
    mlx_reshape(&reshaped, projected, vec_shape, 2, S);
    if (!arr_ok(reshaped))
        goto cleanup;

    result = (embed_mlx_late_vectors_t *)calloc(1, sizeof(*result));
    if (!result)
        goto cleanup;
    result->owner = ctx;
    result->vectors = reshaped;
    result->tokens = n_tokens;
    result->dim = ctx->token_dim;
    reshaped = (mlx_array){0};

cleanup:
    mlx_array_free(reshaped);
    mlx_array_free(projected);
    mlx_array_free(x_normed);
    mlx_array_free(x);
    return result;
}

void embed_mlx_late_vectors_free(embed_mlx_late_vectors_t *vecs) {
    if (!vecs)
        return;
    mlx_array_free(vecs->vectors);
    free(vecs);
}

int embed_mlx_late_vectors_token_count(const embed_mlx_late_vectors_t *vecs) {
    return vecs ? vecs->tokens : 0;
}

int embed_mlx_late_vectors_dim(const embed_mlx_late_vectors_t *vecs) {
    return vecs ? vecs->dim : 0;
}

int embed_mlx_late_vectors_copy(const embed_mlx_late_vectors_t *vecs, float *out_vectors) {
    if (!vecs || !out_vectors || !arr_ok(vecs->vectors) || vecs->tokens <= 0 || vecs->dim <= 0)
        return -1;

    size_t values, bytes;
    if (mlx_mul_size((size_t)vecs->tokens, (size_t)vecs->dim, &values) != 0 ||
        mlx_mul_size(values, sizeof(float), &bytes) != 0)
        return -1;

    mlx_stream S = vecs->owner->base->stream;
    mlx_array out_f32 = mlx_array_new();
    mlx_astype(&out_f32, vecs->vectors, MLX_FLOAT32, S);
    if (!arr_ok(out_f32)) {
        mlx_array_free(out_f32);
        return -1;
    }

    int rc = -1;
    mlx_array_eval(out_f32);
    const float *data = mlx_array_data_float32(out_f32);
    if (data) {
        memcpy(out_vectors, data, bytes);
        rc = 0;
    }
    mlx_array_free(out_f32);
    return rc;
}

/*
 * Batched late-interaction document encoder.
 *
 * Encodes every candidate in one padded transformer pass, then gathers each
 * document's kept token vectors into a single packed [total_keep, token_dim]
 * array in document order. out_offsets receives the prefix sum of kept counts
 * (out_offsets[i + 1] - out_offsets[i] == n_keep[i]), the exact layout
 * embed_mlx_late_maxsim_batch_device consumes. This collapses N per-candidate
 * transformer calls into one forward.
 *
 * Right-padding keeps real tokens at positions 0..n_tokens[d]-1 so RoPE
 * positions are unchanged; a bidirectional model gets a padding attention mask
 * so real tokens never attend to pad slots (the same masking the dense pooled
 * batch path uses). Kept-token vectors therefore match per-document encoding to
 * within MLX's batched-GEMM rounding.
 */
embed_mlx_late_vectors_t *embed_mlx_late_encode_docs_device(embed_mlx_late_ctx_t *ctx,
                                                            const int *const *doc_ids,
                                                            const int *n_tokens,
                                                            const int *const *keep,
                                                            const int *n_keep,
                                                            int n_docs,
                                                            int normalize,
                                                            int *out_offsets) {
    if (!ctx || !ctx->base || !doc_ids || !n_tokens || !keep || !n_keep || !out_offsets ||
        n_docs <= 0 || ctx->token_dim <= 0 || !linear_ok(ctx->projection))
        return NULL;

    embed_mlx_ctx_t *base = ctx->base;
    if (base->layer_start != 0 || base->layer_end != base->config.n_layers ||
        !arr_ok(base->embed_tokens) || !arr_ok(base->norm))
        return NULL;

    const embed_config_t *c = &base->config;
    mlx_stream S = base->stream;
    int causal = c->attention_mode == EMBED_ATTENTION_CAUSAL;

    /* Validate inputs and size the batch: longest sequence and total kept
     * tokens. out_offsets is the running prefix sum of kept counts. */
    int max_seq = 0;
    int total_keep = 0;
    out_offsets[0] = 0;
    for (int d = 0; d < n_docs; d++) {
        if (!doc_ids[d] || n_tokens[d] <= 0 || !keep[d] || n_keep[d] <= 0 ||
            n_keep[d] > n_tokens[d] || total_keep > INT_MAX - n_keep[d])
            return NULL;
        if (n_tokens[d] > max_seq)
            max_seq = n_tokens[d];
        for (int k = 0; k < n_keep[d]; k++) {
            if (keep[d][k] < 0 || keep[d][k] >= n_tokens[d])
                return NULL;
        }
        total_keep += n_keep[d];
        out_offsets[d + 1] = total_keep;
    }

    /* The gather index lives in an MLX int32 array, so the flattened
     * [n_docs * max_seq, dim] row count must stay within int range. */
    size_t elems, ids_bytes, mask_bytes, gather_bytes;
    if (mlx_mul_size((size_t)n_docs, (size_t)max_seq, &elems) != 0 || elems > INT_MAX ||
        mlx_mul_size(elems, sizeof(int), &ids_bytes) != 0 ||
        mlx_mul_size(elems, sizeof(float), &mask_bytes) != 0 ||
        mlx_mul_size((size_t)total_keep, sizeof(int), &gather_bytes) != 0)
        return NULL;

    int has_padding = 0;
    for (int d = 0; d < n_docs; d++)
        if (n_tokens[d] != max_seq)
            has_padding = 1;
    int needs_attn_mask = has_padding && !causal;
    if (ids_bytes == 0 || gather_bytes == 0 || (needs_attn_mask && mask_bytes == 0))
        return NULL;

    int *padded_ids = (int *)malloc(ids_bytes);
    float *attn_mask_data = needs_attn_mask ? (float *)malloc(mask_bytes) : NULL;
    int *gather_idx = (int *)malloc(gather_bytes);
    if (!padded_ids || (needs_attn_mask && !attn_mask_data) || !gather_idx) {
        free(padded_ids);
        free(attn_mask_data);
        free(gather_idx);
        return NULL;
    }

    int gpos = 0;
    for (int d = 0; d < n_docs; d++) {
        for (int t = 0; t < max_seq; t++) {
            size_t idx = (size_t)d * max_seq + t;
            if (t < n_tokens[d]) {
                int id = doc_ids[d][t];
                if (id < 0 || id >= c->vocab_size) {
                    fprintf(stderr, "mlx late: invalid token id %d at doc %d position %d\n", id, d,
                            t);
                    free(padded_ids);
                    free(attn_mask_data);
                    free(gather_idx);
                    return NULL;
                }
                padded_ids[idx] = id;
                if (needs_attn_mask)
                    attn_mask_data[idx] = 0.0f;
            } else {
                padded_ids[idx] = 0;
                if (needs_attn_mask)
                    attn_mask_data[idx] = -1.0e9f;
            }
        }
        /* Map each kept token to its row in the flattened projection. */
        for (int k = 0; k < n_keep[d]; k++)
            gather_idx[gpos++] = d * max_seq + keep[d][k];
    }

    int ids_shape[] = {n_docs, max_seq};
    int attn_mask_shape[] = {n_docs, 1, 1, max_seq};
    int gather_shape[] = {total_keep};
    mlx_array ids = mlx_array_new_data(padded_ids, ids_shape, 2, MLX_INT32);
    mlx_array attn_mask = needs_attn_mask
                              ? mlx_array_new_data(attn_mask_data, attn_mask_shape, 4, MLX_FLOAT32)
                              : (mlx_array){0};
    mlx_array gather = mlx_array_new_data(gather_idx, gather_shape, 1, MLX_INT32);
    free(padded_ids);
    free(attn_mask_data);
    free(gather_idx);
    if (!arr_ok(ids) || (needs_attn_mask && !arr_ok(attn_mask)) || !arr_ok(gather)) {
        mlx_array_free(ids);
        mlx_array_free(attn_mask);
        mlx_array_free(gather);
        return NULL;
    }
    if (needs_attn_mask && mlx_weight_dtype(base) == MLX_BFLOAT16) {
        mlx_array attn_mask_bf16 = mlx_array_new();
        mlx_astype(&attn_mask_bf16, attn_mask, MLX_BFLOAT16, S);
        mlx_array_free(attn_mask);
        attn_mask = attn_mask_bf16;
        if (!arr_ok(attn_mask)) {
            mlx_array_free(ids);
            mlx_array_free(gather);
            return NULL;
        }
    }

    embed_mlx_late_vectors_t *result = NULL;
    mlx_array x = mlx_array_new();
    mlx_array x_normed = (mlx_array){0};
    mlx_array projected = (mlx_array){0};
    mlx_array flat = (mlx_array){0};
    mlx_array gathered = (mlx_array){0};

    mlx_take_axis(&x, base->embed_tokens, ids, 0, S);
    mlx_array_free(ids);
    if (!arr_ok(x))
        goto cleanup;

    x = mlx_forward_layers(base, x, n_docs, max_seq, needs_attn_mask, attn_mask, 0, c->n_layers);
    if (!arr_ok(x))
        goto cleanup;

    x_normed = mlx_array_new();
    mlx_fast_rms_norm(&x_normed, x, base->norm, c->rms_norm_eps, S);
    mlx_array_free(x);
    x = (mlx_array){0};
    if (!arr_ok(x_normed))
        goto cleanup;

    projected = linear(x_normed, &ctx->projection, S);
    mlx_array_free(x_normed);
    x_normed = (mlx_array){0};
    if (!arr_ok(projected))
        goto cleanup;

    /* Flatten [n_docs, max_seq, dim] -> [n_docs * max_seq, dim] and gather the
     * kept tokens in document order into one packed [total_keep, dim] array. */
    int flat_shape[] = {(int)elems, ctx->token_dim};
    flat = mlx_array_new();
    mlx_reshape(&flat, projected, flat_shape, 2, S);
    mlx_array_free(projected);
    projected = (mlx_array){0};
    if (!arr_ok(flat))
        goto cleanup;

    gathered = mlx_array_new();
    mlx_take_axis(&gathered, flat, gather, 0, S);
    mlx_array_free(flat);
    flat = (mlx_array){0};
    if (!arr_ok(gathered))
        goto cleanup;

    /* Normalize after gathering: per-token L2 is row-independent, so this is
     * identical to normalizing the full batch but skips the dropped and padded
     * rows entirely. */
    if (normalize) {
        mlx_array sq = mlx_array_new();
        mlx_square(&sq, gathered, S);
        if (!arr_ok(sq)) {
            mlx_array_free(sq);
            goto cleanup;
        }
        mlx_array norm2 = mlx_array_new();
        mlx_sum_axis(&norm2, sq, 1, true, S);
        mlx_array_free(sq);
        if (!arr_ok(norm2)) {
            mlx_array_free(norm2);
            goto cleanup;
        }
        mlx_array inv_norm = mlx_array_new();
        mlx_rsqrt(&inv_norm, norm2, S);
        mlx_array_free(norm2);
        if (!arr_ok(inv_norm)) {
            mlx_array_free(inv_norm);
            goto cleanup;
        }
        mlx_array normalized = mlx_array_new();
        mlx_multiply(&normalized, gathered, inv_norm, S);
        mlx_array_free(inv_norm);
        mlx_array_free(gathered);
        gathered = normalized;
        if (!arr_ok(gathered))
            goto cleanup;
    }

    result = (embed_mlx_late_vectors_t *)calloc(1, sizeof(*result));
    if (!result)
        goto cleanup;
    result->owner = ctx;
    result->vectors = gathered;
    result->tokens = total_keep;
    result->dim = ctx->token_dim;
    gathered = (mlx_array){0};

cleanup:
    mlx_array_free(gathered);
    mlx_array_free(flat);
    mlx_array_free(projected);
    mlx_array_free(x_normed);
    mlx_array_free(x);
    mlx_array_free(gather);
    mlx_array_free(attn_mask);
    return result;
}

/* Device-resident select/concat primitives. The server rerank path batches
 * candidates through embed_mlx_late_encode_docs_device, but the late-interaction
 * verification harness (scripts/check_late_interaction.py) drives these directly
 * to validate per-document gather and concat against the CPU MaxSim reference. */
embed_mlx_late_vectors_t *embed_mlx_late_vectors_concat(
    embed_mlx_late_ctx_t *ctx, const embed_mlx_late_vectors_t *const *items, int count) {
    if (!ctx || !ctx->base || !items || count <= 0)
        return NULL;
    int dim = 0;
    int total = 0;
    for (int i = 0; i < count; i++) {
        const embed_mlx_late_vectors_t *v = items[i];
        if (!v || v->owner != ctx || !arr_ok(v->vectors) || v->tokens <= 0 || v->dim <= 0)
            return NULL;
        if (i == 0)
            dim = v->dim;
        if (v->dim != dim || total > INT_MAX - v->tokens)
            return NULL;
        total += v->tokens;
    }

    mlx_array *parts = (mlx_array *)malloc((size_t)count * sizeof(*parts));
    if (!parts)
        return NULL;
    for (int i = 0; i < count; i++)
        parts[i] = items[i]->vectors;

    mlx_stream S = ctx->base->stream;
    mlx_vector_array vec = mlx_vector_array_new_data(parts, (size_t)count);
    free(parts);
    if (!vec.ctx)
        return NULL;

    mlx_array joined = mlx_array_new();
    mlx_concatenate_axis(&joined, vec, 0, S);
    mlx_vector_array_free(vec);
    if (!arr_ok(joined)) {
        mlx_array_free(joined);
        return NULL;
    }

    embed_mlx_late_vectors_t *out = (embed_mlx_late_vectors_t *)calloc(1, sizeof(*out));
    if (!out) {
        mlx_array_free(joined);
        return NULL;
    }
    out->owner = ctx;
    out->vectors = joined;
    out->tokens = total;
    out->dim = dim;
    return out;
}

embed_mlx_late_vectors_t *embed_mlx_late_vectors_select(embed_mlx_late_ctx_t *ctx,
                                                        const embed_mlx_late_vectors_t *vecs,
                                                        const int *token_indices,
                                                        int count) {
    if (!ctx || !ctx->base || !vecs || vecs->owner != ctx || !arr_ok(vecs->vectors) ||
        !token_indices || count <= 0 || vecs->tokens <= 0 || vecs->dim <= 0)
        return NULL;

    for (int i = 0; i < count; i++) {
        if (token_indices[i] < 0 || token_indices[i] >= vecs->tokens)
            return NULL;
    }

    size_t idx_bytes;
    if (mlx_mul_size((size_t)count, sizeof(int), &idx_bytes) != 0)
        return NULL;
    int *idx_data = (int *)malloc(idx_bytes);
    if (!idx_data)
        return NULL;
    memcpy(idx_data, token_indices, idx_bytes);

    int idx_shape[] = {count};
    mlx_array idx = mlx_array_new_data(idx_data, idx_shape, 1, MLX_INT32);
    free(idx_data);
    if (!arr_ok(idx))
        return NULL;

    mlx_stream S = ctx->base->stream;
    mlx_array selected = mlx_array_new();
    mlx_take_axis(&selected, vecs->vectors, idx, 0, S);
    mlx_array_free(idx);
    if (!arr_ok(selected)) {
        mlx_array_free(selected);
        return NULL;
    }

    embed_mlx_late_vectors_t *out = (embed_mlx_late_vectors_t *)calloc(1, sizeof(*out));
    if (!out) {
        mlx_array_free(selected);
        return NULL;
    }
    out->owner = ctx;
    out->vectors = selected;
    out->tokens = count;
    out->dim = vecs->dim;
    return out;
}

int embed_mlx_late_maxsim_batch_device(embed_mlx_late_ctx_t *ctx,
                                       const embed_mlx_late_vectors_t *query,
                                       const embed_mlx_late_vectors_t *docs,
                                       const int *doc_offsets,
                                       int docs_count,
                                       float *scores) {
    if (!ctx || !query || !docs || !doc_offsets || docs_count <= 0 || !scores ||
        query->owner != ctx || docs->owner != ctx || query->dim <= 0 || query->dim != docs->dim ||
        query->tokens <= 0 || docs->tokens <= 0 || !arr_ok(query->vectors) ||
        !arr_ok(docs->vectors) || doc_offsets[0] != 0)
        return -1;

    int *lengths = (int *)malloc((size_t)docs_count * sizeof(*lengths));
    if (!lengths)
        return -1;
    int max_doc_tokens = 0;
    for (int i = 0; i < docs_count; i++) {
        if (doc_offsets[i] < 0 || doc_offsets[i] >= doc_offsets[i + 1] ||
            doc_offsets[i + 1] > docs->tokens) {
            free(lengths);
            return -1;
        }
        int n = doc_offsets[i + 1] - doc_offsets[i];
        lengths[i] = n;
        if (n > max_doc_tokens)
            max_doc_tokens = n;
    }
    if (max_doc_tokens <= 0) {
        free(lengths);
        return -1;
    }

    mlx_stream S = ctx->base->stream;
    int dim = query->dim;
    int rc = -1;
    mlx_array *doc_rows = (mlx_array *)calloc((size_t)docs_count, sizeof(*doc_rows));
    if (!doc_rows) {
        free(lengths);
        return -1;
    }

    mlx_vector_array doc_vec = (mlx_vector_array){0};
    mlx_array pad_value = mlx_array_new_float32(0.0f);
    mlx_array doc_batch = (mlx_array){0};
    mlx_array query_3d = (mlx_array){0};
    mlx_array query_batched = (mlx_array){0};
    mlx_array docs_t = (mlx_array){0};
    mlx_array sims = (mlx_array){0};
    mlx_array lengths_arr = (mlx_array){0};
    mlx_array positions = (mlx_array){0};
    mlx_array lengths_2d = (mlx_array){0};
    mlx_array positions_2d = (mlx_array){0};
    mlx_array valid = (mlx_array){0};
    mlx_array valid_3d = (mlx_array){0};
    mlx_array invalid_value = mlx_array_new_float32(-FLT_MAX);
    mlx_array masked_sims = (mlx_array){0};
    mlx_array maxes = (mlx_array){0};
    mlx_array sums = (mlx_array){0};
    mlx_array sums_f32 = (mlx_array){0};

    for (int i = 0; i < docs_count; i++) {
        int n = lengths[i];
        int start[] = {doc_offsets[i], 0};
        int stop[] = {doc_offsets[i + 1], dim};
        int strides[] = {1, 1};

        mlx_array doc = mlx_array_new();
        mlx_array padded = (mlx_array){0};

        mlx_slice(&doc, docs->vectors, start, 2, stop, 2, strides, 2, S);
        if (!arr_ok(doc))
            goto cleanup;

        if (n < max_doc_tokens) {
            int axes[] = {0};
            int low[] = {0};
            int high[] = {max_doc_tokens - n};
            padded = mlx_array_new();
            mlx_pad(&padded, doc, axes, 1, low, 1, high, 1, pad_value, "constant", S);
            mlx_array_free(doc);
            doc = padded;
            padded = (mlx_array){0};
            if (!arr_ok(doc))
                goto cleanup;
        }

        doc_rows[i] = mlx_array_new();
        mlx_expand_dims(&doc_rows[i], doc, 0, S);
        mlx_array_free(doc);
        if (!arr_ok(doc_rows[i]))
            goto cleanup;
    }

    doc_vec = mlx_vector_array_new_data(doc_rows, (size_t)docs_count);
    if (!doc_vec.ctx)
        goto cleanup;

    doc_batch = mlx_array_new();
    mlx_concatenate_axis(&doc_batch, doc_vec, 0, S);
    if (!arr_ok(doc_batch))
        goto cleanup;

    query_3d = mlx_array_new();
    mlx_expand_dims(&query_3d, query->vectors, 0, S);
    if (!arr_ok(query_3d))
        goto cleanup;

    int q_shape[] = {docs_count, query->tokens, dim};
    query_batched = mlx_array_new();
    mlx_broadcast_to(&query_batched, query_3d, q_shape, 3, S);
    if (!arr_ok(query_batched))
        goto cleanup;

    int perm[] = {0, 2, 1};
    docs_t = mlx_array_new();
    mlx_transpose_axes(&docs_t, doc_batch, perm, 3, S);
    if (!arr_ok(docs_t))
        goto cleanup;

    sims = mlx_array_new();
    mlx_matmul(&sims, query_batched, docs_t, S);
    if (!arr_ok(sims))
        goto cleanup;

    int len_shape[] = {docs_count};
    lengths_arr = mlx_array_new_data(lengths, len_shape, 1, MLX_INT32);
    if (!arr_ok(lengths_arr))
        goto cleanup;

    positions = mlx_array_new();
    mlx_arange(&positions, 0.0, (double)max_doc_tokens, 1.0, MLX_INT32, S);
    if (!arr_ok(positions))
        goto cleanup;

    lengths_2d = mlx_array_new();
    mlx_expand_dims(&lengths_2d, lengths_arr, 1, S);
    if (!arr_ok(lengths_2d))
        goto cleanup;

    positions_2d = mlx_array_new();
    mlx_expand_dims(&positions_2d, positions, 0, S);
    if (!arr_ok(positions_2d))
        goto cleanup;

    valid = mlx_array_new();
    mlx_less(&valid, positions_2d, lengths_2d, S);
    if (!arr_ok(valid))
        goto cleanup;

    valid_3d = mlx_array_new();
    mlx_expand_dims(&valid_3d, valid, 1, S);
    if (!arr_ok(valid_3d))
        goto cleanup;

    masked_sims = mlx_array_new();
    mlx_where(&masked_sims, valid_3d, sims, invalid_value, S);
    if (!arr_ok(masked_sims))
        goto cleanup;

    maxes = mlx_array_new();
    mlx_max_axis(&maxes, masked_sims, 2, false, S);
    if (!arr_ok(maxes))
        goto cleanup;

    sums = mlx_array_new();
    mlx_sum_axis(&sums, maxes, 1, false, S);
    if (!arr_ok(sums))
        goto cleanup;

    sums_f32 = mlx_array_new();
    mlx_astype(&sums_f32, sums, MLX_FLOAT32, S);
    if (!arr_ok(sums_f32))
        goto cleanup;

    mlx_array_eval(sums_f32);
    const float *data = mlx_array_data_float32(sums_f32);
    if (!data)
        goto cleanup;
    memcpy(scores, data, (size_t)docs_count * sizeof(float));
    rc = 0;

cleanup:
    mlx_array_free(sums_f32);
    mlx_array_free(sums);
    mlx_array_free(maxes);
    mlx_array_free(masked_sims);
    mlx_array_free(invalid_value);
    mlx_array_free(valid_3d);
    mlx_array_free(valid);
    mlx_array_free(positions_2d);
    mlx_array_free(lengths_2d);
    mlx_array_free(positions);
    mlx_array_free(lengths_arr);
    mlx_array_free(sims);
    mlx_array_free(docs_t);
    mlx_array_free(query_batched);
    mlx_array_free(query_3d);
    mlx_array_free(doc_batch);
    if (doc_vec.ctx)
        mlx_vector_array_free(doc_vec);
    mlx_array_free(pad_value);
    for (int i = 0; i < docs_count; i++)
        mlx_array_free(doc_rows[i]);
    free(doc_rows);
    free(lengths);
    return rc;
}

int embed_mlx_late_encode_tokens(embed_mlx_late_ctx_t *ctx,
                                 const int *token_ids,
                                 int n_tokens,
                                 int normalize,
                                 float *out_vectors) {
    embed_mlx_late_vectors_t *vecs =
        embed_mlx_late_encode_tokens_device(ctx, token_ids, n_tokens, normalize);
    if (!vecs)
        return -1;
    int rc = embed_mlx_late_vectors_copy(vecs, out_vectors);
    embed_mlx_late_vectors_free(vecs);
    return rc;
}

static int embed_mlx_encode_batch_dense(embed_mlx_ctx_t *ctx,
                                        const embed_input_t *inputs,
                                        int batch,
                                        float *out_embeddings) {
    if (!ctx || !inputs || batch <= 0 || !out_embeddings || ctx->layer_start != 0 ||
        ctx->layer_end != ctx->config.n_layers)
        return -1;

    const embed_config_t *c = &ctx->config;
    mlx_stream S = ctx->stream;

    int hidden = c->hidden_size;
    int max_seq = 0;
    int has_padding = 0;
    int causal = c->attention_mode == EMBED_ATTENTION_CAUSAL;
    int cls_pool = c->pooling_mode == EMBED_POOL_CLS;
    /* CLS and last-token are both single-token pooling: gather one row per
     * sequence (the first for CLS, the last otherwise) rather than mean-pool. */
    int single_token_pool = c->pooling_mode == EMBED_POOL_LAST_TOKEN || cls_pool;
    if (hidden <= 0 || c->vocab_size <= 0)
        return -1;

    for (int b = 0; b < batch; b++) {
        if (!inputs[b].ids || inputs[b].n_tokens <= 0)
            return -1;
        if (inputs[b].n_tokens > max_seq)
            max_seq = inputs[b].n_tokens;
    }
    if (max_seq <= 0)
        return -1;

    for (int b = 0; b < batch; b++) {
        if (inputs[b].n_tokens != max_seq)
            has_padding = 1;
    }
    int needs_pool_mask = has_padding && !single_token_pool;
    int needs_attn_mask = has_padding && !causal;

    size_t elems, ids_bytes, mask_bytes, len_bytes, out_values, out_bytes;
    size_t last_index_bytes = 0;
    if (mlx_mul_size((size_t)batch, (size_t)max_seq, &elems) != 0 ||
        mlx_mul_size(elems, sizeof(int), &ids_bytes) != 0 ||
        mlx_mul_size(elems, sizeof(float), &mask_bytes) != 0 ||
        mlx_mul_size((size_t)batch, sizeof(float), &len_bytes) != 0 ||
        mlx_mul_size((size_t)batch, (size_t)hidden, &out_values) != 0 ||
        mlx_mul_size(out_values, sizeof(float), &out_bytes) != 0 ||
        (single_token_pool && mlx_mul_size(out_values, sizeof(int), &last_index_bytes) != 0))
        return -1;
    if (ids_bytes == 0 || out_bytes == 0 ||
        (needs_pool_mask && (mask_bytes == 0 || len_bytes == 0)) ||
        (needs_attn_mask && mask_bytes == 0) || (single_token_pool && last_index_bytes == 0))
        return -1;
    int *padded_ids = (int *)malloc(ids_bytes);
    float *pool_mask_data = needs_pool_mask ? (float *)malloc(mask_bytes) : NULL;
    float *attn_mask_data = needs_attn_mask ? (float *)malloc(mask_bytes) : NULL;
    float *len_data = needs_pool_mask ? (float *)malloc(len_bytes) : NULL;
    int *last_index_data = single_token_pool ? (int *)malloc(last_index_bytes) : NULL;
    if (!padded_ids || (needs_pool_mask && (!pool_mask_data || !len_data)) ||
        (needs_attn_mask && !attn_mask_data)) {
        free(padded_ids);
        free(pool_mask_data);
        free(attn_mask_data);
        free(len_data);
        free(last_index_data);
        return -1;
    }
    if (single_token_pool && !last_index_data) {
        free(padded_ids);
        free(pool_mask_data);
        free(attn_mask_data);
        free(len_data);
        return -1;
    }

    for (int b = 0; b < batch; b++) {
        if (needs_pool_mask)
            len_data[b] = (float)inputs[b].n_tokens;
        if (single_token_pool) {
            int index = cls_pool ? 0 : inputs[b].n_tokens - 1;
            for (int d = 0; d < hidden; d++)
                last_index_data[(size_t)b * hidden + d] = index;
        }
        for (int t = 0; t < max_seq; t++) {
            size_t idx = (size_t)b * max_seq + t;
            if (t < inputs[b].n_tokens) {
                int id = inputs[b].ids[t];
                if (id < 0 || id >= c->vocab_size) {
                    fprintf(stderr, "mlx: invalid token id %d at input %d position %d\n", id, b, t);
                    free(padded_ids);
                    free(pool_mask_data);
                    free(attn_mask_data);
                    free(len_data);
                    free(last_index_data);
                    return -1;
                }
                padded_ids[idx] = id;
                if (needs_pool_mask)
                    pool_mask_data[idx] = 1.0f;
                if (needs_attn_mask)
                    attn_mask_data[idx] = 0.0f;
            } else {
                padded_ids[idx] = 0;
                if (needs_pool_mask)
                    pool_mask_data[idx] = 0.0f;
                if (needs_attn_mask)
                    attn_mask_data[idx] = -1.0e9f;
            }
        }
    }

    int ids_shape[] = {batch, max_seq};
    int pool_mask_shape[] = {batch, max_seq, 1};
    int attn_mask_shape[] = {batch, 1, 1, max_seq};
    int len_shape[] = {batch, 1};
    int last_index_shape[] = {batch, 1, hidden};

    mlx_array ids = mlx_array_new_data(padded_ids, ids_shape, 2, MLX_INT32);
    mlx_array pool_mask = needs_pool_mask
                              ? mlx_array_new_data(pool_mask_data, pool_mask_shape, 3, MLX_FLOAT32)
                              : (mlx_array){0};
    mlx_array attn_mask = needs_attn_mask
                              ? mlx_array_new_data(attn_mask_data, attn_mask_shape, 4, MLX_FLOAT32)
                              : (mlx_array){0};
    mlx_array lengths =
        needs_pool_mask ? mlx_array_new_data(len_data, len_shape, 2, MLX_FLOAT32) : (mlx_array){0};
    mlx_array last_indices =
        single_token_pool ? mlx_array_new_data(last_index_data, last_index_shape, 3, MLX_INT32)
                          : (mlx_array){0};
    free(padded_ids);
    free(pool_mask_data);
    free(attn_mask_data);
    free(len_data);
    free(last_index_data);

    if (!arr_ok(ids) || (needs_pool_mask && (!arr_ok(pool_mask) || !arr_ok(lengths))) ||
        (needs_attn_mask && !arr_ok(attn_mask)) || (single_token_pool && !arr_ok(last_indices))) {
        mlx_array_free(ids);
        mlx_array_free(pool_mask);
        mlx_array_free(attn_mask);
        mlx_array_free(lengths);
        mlx_array_free(last_indices);
        return -1;
    }
    if (needs_attn_mask && mlx_weight_dtype(ctx) == MLX_BFLOAT16) {
        mlx_array attn_mask_bf16 = mlx_array_new();
        mlx_astype(&attn_mask_bf16, attn_mask, MLX_BFLOAT16, S);
        mlx_array_free(attn_mask);
        attn_mask = attn_mask_bf16;
        if (!arr_ok(attn_mask)) {
            mlx_array_free(ids);
            mlx_array_free(pool_mask);
            mlx_array_free(lengths);
            mlx_array_free(last_indices);
            return -1;
        }
    }

    /* 1. Embedding lookup: [B, T] -> [B, T, hidden]. */
    mlx_array x = mlx_array_new();
    mlx_take_axis(&x, ctx->embed_tokens, ids, 0, S);
    mlx_array_free(ids);

    /* 2. Transformer layers. */
    x = mlx_forward_layers(ctx, x, batch, max_seq, needs_attn_mask, attn_mask, 0, c->n_layers);

    /* 3. Final RMSNorm. */
    mlx_array x_normed = mlx_array_new();
    mlx_fast_rms_norm(&x_normed, x, ctx->norm, c->rms_norm_eps, S);
    mlx_array_free(x);

    mlx_array emb = mlx_array_new();
    if (single_token_pool) {
        mlx_array selected = mlx_array_new();
        mlx_take_along_axis(&selected, x_normed, last_indices, 1, S);
        mlx_array_free(x_normed);
        mlx_array_free(last_indices);
        int emb_shape[] = {batch, hidden};
        mlx_reshape(&emb, selected, emb_shape, 2, S);
        mlx_array_free(selected);
        mlx_array_free(pool_mask);
        mlx_array_free(attn_mask);
        mlx_array_free(lengths);
    } else if (has_padding) {
        /* 4. Masked mean pool over T. */
        mlx_array masked = mlx_array_new();
        mlx_multiply(&masked, x_normed, pool_mask, S);
        mlx_array_free(x_normed);
        mlx_array_free(pool_mask);

        mlx_array emb_sum = mlx_array_new();
        mlx_sum_axis(&emb_sum, masked, 1, false, S);
        mlx_array_free(masked);

        mlx_divide(&emb, emb_sum, lengths, S);
        mlx_array_free(emb_sum);
        mlx_array_free(lengths);
        mlx_array_free(attn_mask);
    } else {
        /* Equal-length batches do not need masks or a separate divide. */
        mlx_mean_axis(&emb, x_normed, 1, false, S);
        mlx_array_free(x_normed);
    }

    /* 5. Eval and copy [B, hidden] to CPU. Perplexity embeddings are
     * intentionally unnormalized; callers should use cosine similarity. */
    mlx_array emb_f32 = mlx_array_new();
    mlx_astype(&emb_f32, emb, MLX_FLOAT32, S);
    mlx_array_free(emb);

    mlx_array_eval(emb_f32);
    const float *data = mlx_array_data_float32(emb_f32);
    int rc = -1;
    if (data) {
        memcpy(out_embeddings, data, out_bytes);
        if (c->normalize_embeddings) {
            rc = 0;
            for (int b = 0; b < batch; b++) {
                if (embed_l2_normalize(out_embeddings + (size_t)b * hidden, hidden) != 0) {
                    rc = -1;
                    break;
                }
            }
        } else {
            rc = 0;
        }
    }
    mlx_array_free(emb_f32);
    return rc;
}

typedef struct {
    int index;
    const embed_input_t *input;
} mlx_batch_item_t;

static int mlx_batch_item_cmp(const void *a, const void *b) {
    const mlx_batch_item_t *ia = (const mlx_batch_item_t *)a;
    const mlx_batch_item_t *ib = (const mlx_batch_item_t *)b;
    int da = ia->input->n_tokens;
    int db = ib->input->n_tokens;
    return (da > db) - (da < db);
}

static int mlx_padding_exceeds_limit(size_t padded, size_t tokens) {
    if (tokens > SIZE_MAX / 11 || padded > SIZE_MAX / 5)
        return 1;
    return padded * 5 > tokens * 11;
}

static int mlx_batch_should_split(const embed_input_t *inputs,
                                  int batch,
                                  int *out_total_tokens,
                                  int *out_max_seq) {
    int max_seq = 0;
    int total_tokens = 0;
    for (int b = 0; b < batch; b++) {
        if (!inputs[b].ids || inputs[b].n_tokens <= 0)
            return 0;
        if (inputs[b].n_tokens > max_seq)
            max_seq = inputs[b].n_tokens;
        if (total_tokens > INT_MAX - inputs[b].n_tokens)
            return 0;
        total_tokens += inputs[b].n_tokens;
    }
    if (out_total_tokens)
        *out_total_tokens = total_tokens;
    if (out_max_seq)
        *out_max_seq = max_seq;
    if (batch < 4 || max_seq < 64 || total_tokens <= 0)
        return 0;

    size_t padded;
    if (mlx_mul_size((size_t)batch, (size_t)max_seq, &padded) != 0)
        return 0;

    return mlx_padding_exceeds_limit(padded, (size_t)total_tokens);
}

static int mlx_should_add_to_group(const embed_mlx_ctx_t *ctx,
                                   int group_count,
                                   int group_max,
                                   int next_tokens) {
    if (!ctx || ctx->config.hidden_size < 2048 || group_count <= 0)
        return 1;

    int next_count = group_count + 1;
    int next_max = next_tokens > group_max ? next_tokens : group_max;
    size_t merged, split_left, split_total;
    if (mlx_mul_size((size_t)next_count, (size_t)next_max, &merged) != 0 ||
        mlx_mul_size((size_t)group_count, (size_t)group_max, &split_left) != 0)
        return 0;
    /* Treat one extra MLX launch as roughly 256 padded token rows for 4B.
     * This preserves small mixed groups while splitting larger padded tails. */
    if (split_left > SIZE_MAX - (size_t)next_tokens ||
        split_left + (size_t)next_tokens > SIZE_MAX - 256)
        return 0;
    split_total = split_left + (size_t)next_tokens + 256;
    return merged <= split_total;
}

int embed_mlx_encode_batch(embed_mlx_ctx_t *ctx,
                           const embed_input_t *inputs,
                           int batch,
                           float *out_embeddings) {
    if (!ctx || !inputs || batch <= 0 || !out_embeddings)
        return -1;

    if (!mlx_batch_should_split(inputs, batch, NULL, NULL))
        return embed_mlx_encode_batch_dense(ctx, inputs, batch, out_embeddings);

    const int hidden = ctx->config.hidden_size;
    mlx_batch_item_t *items = (mlx_batch_item_t *)malloc((size_t)batch * sizeof(*items));
    embed_input_t *group_inputs = (embed_input_t *)malloc((size_t)batch * sizeof(*group_inputs));
    size_t group_values, group_bytes;
    if (mlx_mul_size((size_t)batch, (size_t)hidden, &group_values) != 0 ||
        mlx_mul_size(group_values, sizeof(float), &group_bytes) != 0) {
        free(items);
        free(group_inputs);
        return -1;
    }
    float *group_out = (float *)malloc(group_bytes);
    if (!items || !group_inputs || !group_out) {
        free(items);
        free(group_inputs);
        free(group_out);
        return -1;
    }
    for (int i = 0; i < batch; i++) {
        items[i].index = i;
        items[i].input = &inputs[i];
    }
    qsort(items, (size_t)batch, sizeof(*items), mlx_batch_item_cmp);

    int rc = 0;
    for (int start = 0; start < batch && rc == 0;) {
        int group_max = 0;
        int group_tokens = 0;
        int end = start;
        while (end < batch) {
            int next_tokens = items[end].input->n_tokens;
            int next_count = end - start + 1;
            int next_max = next_tokens > group_max ? next_tokens : group_max;
            int next_total = group_tokens + next_tokens;
            if (end > start) {
                size_t next_padded;
                if (mlx_mul_size((size_t)next_count, (size_t)next_max, &next_padded) != 0 ||
                    mlx_padding_exceeds_limit(next_padded, (size_t)next_total) ||
                    !mlx_should_add_to_group(ctx, end - start, group_max, next_tokens))
                    break;
            }
            group_max = next_max;
            group_tokens = next_total;
            end++;
        }

        int group_count = end - start;
        for (int i = 0; i < group_count; i++)
            group_inputs[i] = *items[start + i].input;
        rc = embed_mlx_encode_batch_dense(ctx, group_inputs, group_count, group_out);
        if (rc != 0)
            break;
        for (int i = 0; i < group_count; i++) {
            int dst = items[start + i].index;
            memcpy(out_embeddings + (size_t)dst * hidden, group_out + (size_t)i * hidden,
                   (size_t)hidden * sizeof(float));
        }
        start = end;
    }

    free(items);
    free(group_inputs);
    free(group_out);
    return rc;
}

int embed_mlx_encode_into(embed_mlx_ctx_t *ctx,
                          const int *token_ids,
                          int n_tokens,
                          float *out_embedding) {
    embed_input_t input = {token_ids, n_tokens};
    return embed_mlx_encode_batch(ctx, &input, 1, out_embedding);
}

typedef struct {
    int index;
    int span_offset;
    const embed_context_input_t *input;
} mlx_context_batch_item_t;

static int mlx_context_batch_item_cmp(const void *a, const void *b) {
    const mlx_context_batch_item_t *ia = (const mlx_context_batch_item_t *)a;
    const mlx_context_batch_item_t *ib = (const mlx_context_batch_item_t *)b;
    int da = ia->input->input.n_tokens;
    int db = ib->input->input.n_tokens;
    return (da > db) - (da < db);
}

static int mlx_context_padding_exceeds_limit(size_t padded, size_t tokens) {
    if (tokens > SIZE_MAX / 5 || padded > SIZE_MAX / 4)
        return 1;
    return padded * 4 > tokens * 5;
}

static int mlx_context_batch_should_split(const embed_context_input_t *inputs,
                                          int batch,
                                          int *out_total_spans) {
    int max_seq = 0;
    int total_tokens = 0;
    int total_spans = 0;
    for (int b = 0; b < batch; b++) {
        int n_tokens = inputs[b].input.n_tokens;
        if (!inputs[b].input.ids || n_tokens <= 0 || !inputs[b].spans || inputs[b].n_spans <= 0)
            return 0;
        if (n_tokens > max_seq)
            max_seq = n_tokens;
        if (total_tokens > INT_MAX - n_tokens || total_spans > INT_MAX - inputs[b].n_spans)
            return 0;
        total_tokens += n_tokens;
        total_spans += inputs[b].n_spans;
    }
    if (out_total_spans)
        *out_total_spans = total_spans;
    if (batch < 2 || max_seq < 64 || total_tokens <= 0)
        return 0;

    size_t padded;
    if (mlx_mul_size((size_t)batch, (size_t)max_seq, &padded) != 0)
        return 0;
    return mlx_context_padding_exceeds_limit(padded, (size_t)total_tokens);
}

static int embed_mlx_encode_spans_batch_dense(embed_mlx_ctx_t *ctx,
                                              const embed_context_input_t *inputs,
                                              int batch,
                                              float *out_embeddings) {
    if (!ctx || !inputs || batch <= 0 || !out_embeddings || ctx->layer_start != 0 ||
        ctx->layer_end != ctx->config.n_layers || !arr_ok(ctx->embed_tokens) || !arr_ok(ctx->norm))
        return -1;

    const embed_config_t *c = &ctx->config;
    mlx_stream S = ctx->stream;
    int hidden = c->hidden_size;
    int max_seq = 0;
    int total_spans = 0;
    int has_padding = 0;
    for (int b = 0; b < batch; b++) {
        const embed_context_input_t *input = &inputs[b];
        int n_tokens = input->input.n_tokens;
        if (!input->input.ids || n_tokens <= 0 || !input->spans || input->n_spans <= 0 ||
            total_spans > INT_MAX - input->n_spans)
            return -1;
        total_spans += input->n_spans;
        if (n_tokens > max_seq)
            max_seq = n_tokens;
        for (int t = 0; t < n_tokens; t++) {
            if (input->input.ids[t] < 0 || input->input.ids[t] >= c->vocab_size) {
                fprintf(stderr, "mlx: invalid token id %d at input %d position %d\n",
                        input->input.ids[t], b, t);
                return -1;
            }
        }
        for (int s = 0; s < input->n_spans; s++) {
            if (input->spans[s].start < 0 || input->spans[s].start > n_tokens ||
                input->spans[s].n_tokens <= 0 ||
                input->spans[s].n_tokens > n_tokens - input->spans[s].start)
                return -1;
        }
    }
    for (int b = 0; b < batch; b++)
        if (inputs[b].input.n_tokens != max_seq)
            has_padding = 1;

    size_t rows, ids_bytes, mask_bytes, index_bytes;
    size_t span_float_bytes, out_values, out_bytes;
    if (mlx_mul_size((size_t)batch, (size_t)max_seq, &rows) != 0 ||
        mlx_mul_size(rows, sizeof(int), &ids_bytes) != 0 ||
        mlx_mul_size(rows, sizeof(float), &mask_bytes) != 0 ||
        mlx_mul_size((size_t)total_spans, sizeof(int), &index_bytes) != 0 ||
        mlx_mul_size((size_t)total_spans, sizeof(float), &span_float_bytes) != 0 ||
        mlx_mul_size((size_t)total_spans, (size_t)hidden, &out_values) != 0 ||
        mlx_mul_size(out_values, sizeof(float), &out_bytes) != 0)
        return -1;
    if (rows > INT_MAX)
        return -1;
    int *padded_ids = (int *)malloc(ids_bytes);
    float *attn_mask_data = has_padding ? (float *)malloc(mask_bytes) : NULL;
    if (!padded_ids || (has_padding && !attn_mask_data)) {
        free(padded_ids);
        free(attn_mask_data);
        return -1;
    }
    for (int b = 0; b < batch; b++) {
        const embed_input_t *input = &inputs[b].input;
        for (int t = 0; t < max_seq; t++) {
            size_t row = (size_t)b * max_seq + t;
            if (t < input->n_tokens) {
                padded_ids[row] = input->ids[t];
                if (has_padding)
                    attn_mask_data[row] = 0.0f;
            } else {
                padded_ids[row] = 0;
                if (has_padding)
                    attn_mask_data[row] = -1.0e9f;
            }
        }
    }

    mlx_array ids = (mlx_array){0};
    mlx_array attn_mask = (mlx_array){0};
    mlx_array x = (mlx_array){0};
    mlx_array x_normed = (mlx_array){0};
    mlx_array prefix = (mlx_array){0};
    mlx_array flat_prefix = (mlx_array){0};
    mlx_array end_idx = (mlx_array){0};
    mlx_array start_idx = (mlx_array){0};
    mlx_array lengths = (mlx_array){0};
    mlx_array start_mask = (mlx_array){0};
    mlx_array end_values = (mlx_array){0};
    mlx_array start_values = (mlx_array){0};
    mlx_array masked_start = (mlx_array){0};
    mlx_array span_sums = (mlx_array){0};
    mlx_array embeddings = (mlx_array){0};
    mlx_array embeddings_f32 = (mlx_array){0};
    mlx_array *pooled = NULL;
    mlx_vector_array pooled_vec = (mlx_vector_array){0};
    int *end_idx_data = NULL;
    int *start_idx_data = NULL;
    float *lengths_data = NULL;
    float *start_mask_data = NULL;
    int rc = -1;

    int ids_shape[] = {batch, max_seq};
    int attn_mask_shape[] = {batch, 1, 1, max_seq};
    ids = mlx_array_new_data(padded_ids, ids_shape, 2, MLX_INT32);
    attn_mask = has_padding ? mlx_array_new_data(attn_mask_data, attn_mask_shape, 4, MLX_FLOAT32)
                            : (mlx_array){0};
    free(padded_ids);
    free(attn_mask_data);
    if (!arr_ok(ids))
        goto cleanup;
    if (has_padding && !arr_ok(attn_mask))
        goto cleanup;
    if (has_padding && mlx_weight_dtype(ctx) == MLX_BFLOAT16) {
        mlx_array attn_mask_bf16 = mlx_array_new();
        mlx_astype(&attn_mask_bf16, attn_mask, MLX_BFLOAT16, S);
        mlx_array_free(attn_mask);
        attn_mask = attn_mask_bf16;
        if (!arr_ok(attn_mask))
            goto cleanup;
    }

    x = mlx_array_new();
    mlx_take_axis(&x, ctx->embed_tokens, ids, 0, S);
    mlx_array_free(ids);
    ids = (mlx_array){0};
    if (!arr_ok(x))
        goto cleanup;

    x = mlx_forward_layers(ctx, x, batch, max_seq, has_padding, attn_mask, 0, c->n_layers);
    if (has_padding) {
        mlx_array_free(attn_mask);
        attn_mask = (mlx_array){0};
    }
    if (!arr_ok(x))
        goto cleanup;

    x_normed = mlx_array_new();
    mlx_fast_rms_norm(&x_normed, x, ctx->norm, c->rms_norm_eps, S);
    mlx_array_free(x);
    x = (mlx_array){0};
    if (!arr_ok(x_normed))
        goto cleanup;

    if (mlx_weight_dtype(ctx) == MLX_BFLOAT16) {
        pooled = (mlx_array *)calloc((size_t)total_spans, sizeof(*pooled));
        if (!pooled)
            goto cleanup;

        int pooled_count = 0;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < inputs[b].n_spans; s++) {
                const embed_span_t *span = &inputs[b].spans[s];
                int start[] = {b, span->start, 0};
                int stop[] = {b + 1, span->start + span->n_tokens, hidden};
                int strides[] = {1, 1, 1};
                mlx_array slice = mlx_array_new();
                mlx_slice(&slice, x_normed, start, 3, stop, 3, strides, 3, S);
                if (!arr_ok(slice)) {
                    mlx_array_free(slice);
                    goto cleanup;
                }

                pooled[pooled_count] = mlx_array_new();
                mlx_mean_axis(&pooled[pooled_count], slice, 1, false, S);
                mlx_array_free(slice);
                if (!arr_ok(pooled[pooled_count]))
                    goto cleanup;
                pooled_count++;
            }
        }

        pooled_vec = mlx_vector_array_new_data(pooled, (size_t)total_spans);
        if (!pooled_vec.ctx)
            goto cleanup;
        embeddings = mlx_array_new();
        mlx_concatenate_axis(&embeddings, pooled_vec, 0, S);
        if (!arr_ok(embeddings))
            goto cleanup;
    } else {
        end_idx_data = (int *)malloc(index_bytes);
        start_idx_data = (int *)malloc(index_bytes);
        lengths_data = (float *)malloc(span_float_bytes);
        start_mask_data = (float *)malloc(span_float_bytes);
        if (!end_idx_data || !start_idx_data || !lengths_data || !start_mask_data)
            goto cleanup;

        int span_count = 0;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < inputs[b].n_spans; s++) {
                const embed_span_t *span = &inputs[b].spans[s];
                int base = b * max_seq;
                int start = span->start;
                int stop = span->start + span->n_tokens;
                end_idx_data[span_count] = base + stop - 1;
                start_idx_data[span_count] = start > 0 ? base + start - 1 : base;
                lengths_data[span_count] = (float)span->n_tokens;
                start_mask_data[span_count] = start > 0 ? 1.0f : 0.0f;
                span_count++;
            }
        }

        int index_shape[] = {total_spans};
        int scale_shape[] = {total_spans, 1};
        end_idx = mlx_array_new_data(end_idx_data, index_shape, 1, MLX_INT32);
        start_idx = mlx_array_new_data(start_idx_data, index_shape, 1, MLX_INT32);
        lengths = mlx_array_new_data(lengths_data, scale_shape, 2, MLX_FLOAT32);
        start_mask = mlx_array_new_data(start_mask_data, scale_shape, 2, MLX_FLOAT32);
        free(end_idx_data);
        end_idx_data = NULL;
        free(start_idx_data);
        start_idx_data = NULL;
        free(lengths_data);
        lengths_data = NULL;
        free(start_mask_data);
        start_mask_data = NULL;
        if (!arr_ok(end_idx) || !arr_ok(start_idx) || !arr_ok(lengths) || !arr_ok(start_mask))
            goto cleanup;

        prefix = mlx_array_new();
        mlx_cumsum(&prefix, x_normed, 1, false, true, S);
        mlx_array_free(x_normed);
        x_normed = (mlx_array){0};
        if (!arr_ok(prefix))
            goto cleanup;

        int flat_shape[] = {(int)rows, hidden};
        flat_prefix = mlx_array_new();
        mlx_reshape(&flat_prefix, prefix, flat_shape, 2, S);
        mlx_array_free(prefix);
        prefix = (mlx_array){0};
        if (!arr_ok(flat_prefix))
            goto cleanup;

        end_values = mlx_array_new();
        mlx_take_axis(&end_values, flat_prefix, end_idx, 0, S);
        if (!arr_ok(end_values))
            goto cleanup;

        start_values = mlx_array_new();
        mlx_take_axis(&start_values, flat_prefix, start_idx, 0, S);
        if (!arr_ok(start_values))
            goto cleanup;

        masked_start = mlx_array_new();
        mlx_multiply(&masked_start, start_values, start_mask, S);
        if (!arr_ok(masked_start))
            goto cleanup;

        span_sums = mlx_array_new();
        mlx_subtract(&span_sums, end_values, masked_start, S);
        if (!arr_ok(span_sums))
            goto cleanup;

        embeddings = mlx_array_new();
        mlx_divide(&embeddings, span_sums, lengths, S);
        if (!arr_ok(embeddings))
            goto cleanup;
    }

    embeddings_f32 = mlx_array_new();
    mlx_astype(&embeddings_f32, embeddings, MLX_FLOAT32, S);
    if (!arr_ok(embeddings_f32))
        goto cleanup;

    mlx_array_eval(embeddings_f32);
    const float *data = mlx_array_data_float32(embeddings_f32);
    if (data) {
        memcpy(out_embeddings, data, out_bytes);
        rc = 0;
    }

cleanup:
    free(start_mask_data);
    free(lengths_data);
    free(start_idx_data);
    free(end_idx_data);
    mlx_array_free(embeddings_f32);
    mlx_array_free(embeddings);
    if (pooled_vec.ctx)
        mlx_vector_array_free(pooled_vec);
    if (pooled) {
        for (int s = 0; s < total_spans; s++)
            mlx_array_free(pooled[s]);
        free(pooled);
    }
    mlx_array_free(span_sums);
    mlx_array_free(masked_start);
    mlx_array_free(start_values);
    mlx_array_free(end_values);
    mlx_array_free(start_mask);
    mlx_array_free(lengths);
    mlx_array_free(start_idx);
    mlx_array_free(end_idx);
    mlx_array_free(flat_prefix);
    mlx_array_free(prefix);
    mlx_array_free(x_normed);
    mlx_array_free(x);
    mlx_array_free(attn_mask);
    mlx_array_free(ids);
    return rc;
}

int embed_mlx_encode_spans_batch(embed_mlx_ctx_t *ctx,
                                 const embed_context_input_t *inputs,
                                 int batch,
                                 float *out_embeddings) {
    if (!ctx || !inputs || batch <= 0 || !out_embeddings)
        return -1;

    int total_spans = 0;
    if (!mlx_context_batch_should_split(inputs, batch, &total_spans))
        return embed_mlx_encode_spans_batch_dense(ctx, inputs, batch, out_embeddings);

    const int hidden = ctx->config.hidden_size;
    mlx_context_batch_item_t *items =
        (mlx_context_batch_item_t *)malloc((size_t)batch * sizeof(*items));
    embed_context_input_t *group_inputs =
        (embed_context_input_t *)malloc((size_t)batch * sizeof(*group_inputs));
    size_t group_values, group_bytes;
    if (mlx_mul_size((size_t)total_spans, (size_t)hidden, &group_values) != 0 ||
        mlx_mul_size(group_values, sizeof(float), &group_bytes) != 0) {
        free(items);
        free(group_inputs);
        return -1;
    }
    float *group_out = (float *)malloc(group_bytes);
    if (!items || !group_inputs || !group_out) {
        free(items);
        free(group_inputs);
        free(group_out);
        return -1;
    }

    int span_offset = 0;
    for (int i = 0; i < batch; i++) {
        items[i].index = i;
        items[i].span_offset = span_offset;
        items[i].input = &inputs[i];
        span_offset += inputs[i].n_spans;
    }
    qsort(items, (size_t)batch, sizeof(*items), mlx_context_batch_item_cmp);

    int rc = 0;
    for (int start = 0; start < batch && rc == 0;) {
        int group_max = 0;
        int group_tokens = 0;
        int end = start;
        while (end < batch) {
            int next_tokens = items[end].input->input.n_tokens;
            int next_count = end - start + 1;
            int next_max = next_tokens > group_max ? next_tokens : group_max;
            int next_total = group_tokens + next_tokens;
            if (end > start) {
                size_t next_padded;
                if (mlx_mul_size((size_t)next_count, (size_t)next_max, &next_padded) != 0 ||
                    mlx_context_padding_exceeds_limit(next_padded, (size_t)next_total) ||
                    !mlx_should_add_to_group(ctx, end - start, group_max, next_tokens))
                    break;
            }
            group_max = next_max;
            group_tokens = next_total;
            end++;
        }

        int group_count = end - start;
        for (int i = 0; i < group_count; i++)
            group_inputs[i] = *items[start + i].input;
        rc = embed_mlx_encode_spans_batch_dense(ctx, group_inputs, group_count, group_out);
        if (rc != 0)
            break;
        int src_span = 0;
        for (int i = 0; i < group_count; i++) {
            int n_spans = group_inputs[i].n_spans;
            int dst_span = items[start + i].span_offset;
            memcpy(out_embeddings + (size_t)dst_span * hidden,
                   group_out + (size_t)src_span * hidden, (size_t)n_spans * hidden * sizeof(float));
            src_span += n_spans;
        }
        start = end;
    }

    free(items);
    free(group_inputs);
    free(group_out);
    return rc;
}

int embed_mlx_encode_spans(embed_mlx_ctx_t *ctx,
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
    return embed_mlx_encode_spans_batch(ctx, &input, 1, out_embeddings);
}

float *embed_mlx_encode(embed_mlx_ctx_t *ctx, const int *token_ids, int n_tokens) {
    if (!ctx || !token_ids || n_tokens <= 0)
        return NULL;

    int hidden = ctx->config.hidden_size;
    size_t out_bytes;
    if (mlx_mul_size((size_t)hidden, sizeof(float), &out_bytes) != 0)
        return NULL;
    float *out = (float *)malloc(out_bytes);
    if (!out)
        return NULL;

    if (embed_mlx_encode_into(ctx, token_ids, n_tokens, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}
