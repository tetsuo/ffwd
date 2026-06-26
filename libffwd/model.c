#include "ffwd.h"
#include "model_internal.h"
#include "config.h"
#include "dtype.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Model-directory probes */

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

int model_dir_has_late_projection(const char *model_dir) {
    return model_dir_has_file(model_dir, "1_Dense/config.json") &&
           model_dir_has_file(model_dir, "1_Dense/model.safetensors");
}

/* Weight tensor loading and validation (direct mmap pointers into safetensors) */

static size_t dtype_size(safetensor_dtype_t dtype) {
    switch (dtype) {
    case DTYPE_F32:
        return sizeof(float);
    case DTYPE_BF16:
    case DTYPE_F16:
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

int tensor_has_supported_shape(const safetensors_file_t *sf,
                               const safetensor_t *t,
                               const char *name,
                               const int64_t *shape,
                               int ndim,
                               int bf16_ok) {
    /* bf16_ok admits either 16-bit float format (BF16 or F16); both are widened
     * to F32 by the caller (BF16 lazily per matmul, F16 once at load). */
    int sixteen_ok = bf16_ok && (t->dtype == DTYPE_BF16 || t->dtype == DTYPE_F16);
    if (t->dtype != DTYPE_F32 && !sixteen_ok) {
        fprintf(stderr, "ffwd: unsupported dtype for %s: got %s, expected %s\n", name,
                dtype_name(t->dtype), bf16_ok ? "F32, BF16, or F16" : "F32");
        return 0;
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "ffwd: bad rank for %s: got %d, expected %d\n", name, t->ndim, ndim);
        return 0;
    }

    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (t->shape[i] != shape[i]) {
            fprintf(stderr,
                    "ffwd: bad shape for %s at dim %d: "
                    "got %lld, expected %lld\n",
                    name, i, (long long)t->shape[i], (long long)shape[i]);
            return 0;
        }
        if (shape[i] < 0 || (uint64_t)shape[i] > SIZE_MAX / numel) {
            fprintf(stderr, "ffwd: shape too large for %s\n", name);
            return 0;
        }
        numel *= (size_t)shape[i];
    }

    size_t elem_size = dtype_size(t->dtype);
    if (elem_size == 0) {
        fprintf(stderr, "ffwd: unsupported dtype for %s: got %s\n", name, dtype_name(t->dtype));
        return 0;
    }

    size_t bytes;
    if (mul_size(numel, elem_size, &bytes) != 0) {
        fprintf(stderr, "ffwd: tensor data too large for %s\n", name);
        return 0;
    }
    if (t->data_size != bytes) {
        fprintf(stderr, "ffwd: bad data size for %s: got %zu, expected %zu\n", name, t->data_size,
                bytes);
        return 0;
    }

    size_t data_start = 8 + sf->header_size;
    if (data_start > sf->file_size || t->data_offset > sf->file_size - data_start ||
        t->data_size > sf->file_size - data_start - t->data_offset) {
        fprintf(stderr, "ffwd: tensor data out of bounds for %s\n", name);
        return 0;
    }
    return 1;
}

/* Widen a safetensors tensor to f32. Caller frees. */
static float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t) {
    int64_t n = safetensor_numel(t);
    if (n <= 0)
        return NULL;
    float *out = malloc((size_t)n * sizeof(float));
    if (!out)
        return NULL;
    const void *data = safetensors_data(sf, t);
    switch (t->dtype) {
    case DTYPE_F32:
        memcpy(out, data, (size_t)n * sizeof(float));
        break;
    case DTYPE_BF16: {
        const uint16_t *s = (const uint16_t *)data;
        for (int64_t i = 0; i < n; i++)
            out[i] = ffwd_bf16_to_f32(s[i]);
        break;
    }
    case DTYPE_F16: {
        const uint16_t *s = (const uint16_t *)data;
        for (int64_t i = 0; i < n; i++)
            out[i] = ffwd_f16_to_f32(s[i]);
        break;
    }
    default:
        free(out);
        return NULL;
    }
    return out;
}

ffwd_weight_ref_t weight_ref_from_tensor(const safetensors_file_t *sf, const safetensor_t *t) {
    ffwd_weight_ref_t ref = {DTYPE_UNKNOWN, NULL, 0};
    /* F32/BF16 are consumed directly by the forward pass, so the ref borrows the
     * mmap (zero copy; BF16 is widened lazily per matmul). F16 has neither a
     * fused kernel nor a zero-copy F32 view, so widen it once here into an owned
     * buffer the model frees on unload. DTYPE_F16 never reaches the forward. */
    if (t->dtype == DTYPE_F16) {
        float *buf = safetensors_get_f32(sf, t);
        if (!buf)
            return ref;
        ref.dtype = DTYPE_F32;
        ref.data = buf;
        ref.owned = 1;
    } else {
        ref.dtype = t->dtype;
        ref.data = safetensors_data(sf, t);
    }
    return ref;
}

static ffwd_weight_ref_t
load_weight_direct(multi_safetensors_t *ms, const char *name, const int64_t *shape, int ndim) {
    ffwd_weight_ref_t ref = {DTYPE_UNKNOWN, NULL, 0};
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "ffwd: tensor not found: %s\n", name);
        return ref;
    }
    if (!tensor_has_supported_shape(sf, t, name, shape, ndim, 1))
        return ref;

    return weight_ref_from_tensor(sf, t);
}

static float *
load_norm_f32(multi_safetensors_t *ms, const char *name, const int64_t *shape, int ndim) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) {
        fprintf(stderr, "ffwd: tensor not found: %s\n", name);
        return NULL;
    }
    if (!tensor_has_supported_shape(sf, t, name, shape, ndim, 1))
        return NULL;
    return safetensors_get_f32(sf, t);
}

/* Model / workspace load and free */

/* BERT-family weight loader.
 * Tensor names differ completely from Qwen
 * (encoder.layer.N.*, embeddings.*), so this family has its own entry point.
 *
 * Reused ffwd_layer_t slots keep their BERT meanings:
 * wo is the attention output dense;
 * input_norm/post_attn_norm are the two block LayerNorm weights;
 * up_proj/down_proj are the two FFN matrices.
 *
 * See internal.h.
 */
static int
load_bert_weights(ffwd_model_t *model, multi_safetensors_t *ms, int layer_start, int layer_end) {
    const ffwd_config_t *cfg = &model->config;
    ffwd_weights_t *w = &model->weights;
    char name[256];

    /* sentence-transformers checkpoints store encoder tensors unprefixed; raw
     * HF BERT and RoBERTa/XLM-R checkpoints use "bert." or "roberta.". */
    const char *p;
    if (multi_safetensors_find(ms, "embeddings.word_embeddings.weight", NULL))
        p = "";
    else if (multi_safetensors_find(ms, "bert.embeddings.word_embeddings.weight", NULL))
        p = "bert.";
    else if (multi_safetensors_find(ms, "roberta.embeddings.word_embeddings.weight", NULL))
        p = "roberta.";
    else {
        fprintf(stderr, "ffwd_model_load: BERT-family word embeddings not found\n");
        return -1;
    }

    int hidden = cfg->hidden_size, inter = cfg->intermediate_size;

    if (layer_start == 0) {
        const int64_t emb_shape[2] = {cfg->vocab_size, hidden};
        snprintf(name, sizeof(name), "%sembeddings.word_embeddings.weight", p);
        w->embed_tokens = load_weight_direct(ms, name, emb_shape, 2);
        const int64_t pos_shape[2] = {cfg->max_position_embeddings, hidden};
        snprintf(name, sizeof(name), "%sembeddings.position_embeddings.weight", p);
        w->position_embeddings = load_norm_f32(ms, name, pos_shape, 2);
        const int64_t type_shape[2] = {cfg->type_vocab_size, hidden};
        snprintf(name, sizeof(name), "%sembeddings.token_type_embeddings.weight", p);
        w->token_type_embeddings = load_norm_f32(ms, name, type_shape, 2);
        const int64_t ln_shape[1] = {hidden};
        snprintf(name, sizeof(name), "%sembeddings.LayerNorm.weight", p);
        w->ffwd_ln_w = load_norm_f32(ms, name, ln_shape, 1);
        snprintf(name, sizeof(name), "%sembeddings.LayerNorm.bias", p);
        w->ffwd_ln_b = load_norm_f32(ms, name, ln_shape, 1);
        if (!w->embed_tokens.data || !w->position_embeddings || !w->token_type_embeddings ||
            !w->ffwd_ln_w || !w->ffwd_ln_b)
            return -1;
    }

    w->layers = (ffwd_layer_t *)calloc(cfg->n_layers, sizeof(ffwd_layer_t));
    if (!w->layers)
        return -1;

    for (int i = layer_start; i < layer_end; i++) {
        ffwd_layer_t *l = &w->layers[i];

#define BERT_W(field, suffix, d0, d1)                                     \
    do {                                                                  \
        const int64_t e[2] = {(d0), (d1)};                                \
        snprintf(name, sizeof(name), "%sencoder.layer.%d." suffix, p, i); \
        l->field = load_weight_direct(ms, name, e, 2);                    \
        if (!l->field.data)                                               \
            return -1;                                                    \
    } while (0)
#define BERT_V(field, suffix, d0)                                         \
    do {                                                                  \
        const int64_t e[1] = {(d0)};                                      \
        snprintf(name, sizeof(name), "%sencoder.layer.%d." suffix, p, i); \
        l->field = load_norm_f32(ms, name, e, 1);                         \
        if (!l->field)                                                    \
            return -1;                                                    \
    } while (0)

        BERT_W(wq, "attention.self.query.weight", hidden, hidden);
        BERT_V(q_bias, "attention.self.query.bias", hidden);
        BERT_W(wk, "attention.self.key.weight", hidden, hidden);
        BERT_V(k_bias, "attention.self.key.bias", hidden);
        BERT_W(wv, "attention.self.value.weight", hidden, hidden);
        BERT_V(v_bias, "attention.self.value.bias", hidden);
        BERT_W(wo, "attention.output.dense.weight", hidden, hidden);
        BERT_V(o_bias, "attention.output.dense.bias", hidden);
        BERT_V(input_norm, "attention.output.LayerNorm.weight", hidden);
        BERT_V(attn_ln_bias, "attention.output.LayerNorm.bias", hidden);
        BERT_W(up_proj, "intermediate.dense.weight", inter, hidden);
        BERT_V(ffn_inter_bias, "intermediate.dense.bias", inter);
        BERT_W(down_proj, "output.dense.weight", hidden, inter);
        BERT_V(ffn_out_bias, "output.dense.bias", hidden);
        BERT_V(post_attn_norm, "output.LayerNorm.weight", hidden);
        BERT_V(ffn_ln_bias, "output.LayerNorm.bias", hidden);

#undef BERT_W
#undef BERT_V
    }
    /* BERT has no final norm: w->norm stays NULL, and pooling skips it. */
    return 0;
}

ffwd_model_t *
model_load_range_ex(const char *model_dir, int layer_start, int layer_end, int allow_late) {
    if (!allow_late && model_dir_has_late_projection(model_dir)) {
        fprintf(stderr, "ffwd_model_load: late-interaction models require "
                        "token-level MaxSim support and are not valid pooled "
                        "embedding models yet\n");
        return NULL;
    }

    ffwd_model_t *model = (ffwd_model_t *)calloc(1, sizeof(ffwd_model_t));
    if (!model)
        return NULL;

    /* Parse config.json */
    if (ffwd_config_parse(&model->config, model_dir) != 0) {
        free(model);
        return NULL;
    }

    const ffwd_config_t *cfg = &model->config;
    if (layer_end == -1)
        layer_end = cfg->n_layers;
    if (layer_start < 0 || layer_start >= layer_end || layer_end > cfg->n_layers) {
        fprintf(stderr,
                "ffwd_model_load: invalid layer range [%d, %d) "
                "for %d-layer model\n",
                layer_start, layer_end, cfg->n_layers);
        free(model);
        return NULL;
    }
    model->layer_start = layer_start;
    model->layer_end = layer_end;

    /* Open safetensors (handles single file or multi-shard) */
    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "ffwd_model_load: failed to open safetensors in %s\n", model_dir);
        free(model);
        return NULL;
    }
    model->safetensors = ms;

    if (cfg->family == FFWD_FAMILY_BERT) {
        if (load_bert_weights(model, ms, layer_start, layer_end) != 0)
            goto fail;
        if (ffwd_verbose >= 1)
            fprintf(stderr, "ffwd_model_load: BERT layers [%d, %d) loaded (%d-dim)\n", layer_start,
                    layer_end, cfg->hidden_size);
        return model;
    }

    const char *weight_prefix = multi_safetensors_weight_prefix(ms, "embed_tokens.weight");
    if (!weight_prefix) {
        fprintf(stderr, "ffwd_model_load: neither embed_tokens.weight nor "
                        "model.embed_tokens.weight was found\n");
        goto fail;
    }

    if (ffwd_verbose >= 1)
        fprintf(stderr, "ffwd_model_load: loading weights from %s\n", model_dir);

    char name[256];

    /* Only the first stage needs token embeddings. */
    ffwd_weights_t *w = &model->weights;
    if (layer_start == 0) {
        const int64_t embed_shape[2] = {cfg->vocab_size, cfg->hidden_size};
        snprintf(name, sizeof(name), "%sembed_tokens.weight", weight_prefix);
        w->embed_tokens = load_weight_direct(ms, name, embed_shape, 2);
        if (!w->embed_tokens.data)
            goto fail;
    }

    /* Allocate layer array */
    w->layers = (ffwd_layer_t *)calloc(cfg->n_layers, sizeof(ffwd_layer_t));
    if (!w->layers)
        goto fail;

    /* Per-layer weights */
    for (int i = layer_start; i < layer_end; i++) {
        ffwd_layer_t *l = &w->layers[i];

#define LOAD_NORM(field, fmt, d0)                            \
    do {                                                     \
        const int64_t expect[1] = {(d0)};                    \
        snprintf(name, sizeof(name), fmt, weight_prefix, i); \
        l->field = load_norm_f32(ms, name, expect, 1);       \
        if (!l->field)                                       \
            goto fail;                                       \
    } while (0)

#define LOAD2(field, fmt, d0, d1)                            \
    do {                                                     \
        const int64_t expect[2] = {(d0), (d1)};              \
        snprintf(name, sizeof(name), fmt, weight_prefix, i); \
        l->field = load_weight_direct(ms, name, expect, 2);  \
        if (!l->field.data)                                  \
            goto fail;                                       \
    } while (0)

        LOAD2(wq, "%slayers.%d.self_attn.q_proj.weight", cfg->q_dim, cfg->hidden_size);
        LOAD2(wk, "%slayers.%d.self_attn.k_proj.weight", cfg->kv_dim, cfg->hidden_size);
        LOAD2(wv, "%slayers.%d.self_attn.v_proj.weight", cfg->kv_dim, cfg->hidden_size);
        LOAD2(wo, "%slayers.%d.self_attn.o_proj.weight", cfg->hidden_size, cfg->q_dim);
        if (cfg->qkv_bias) {
            LOAD_NORM(q_bias, "%slayers.%d.self_attn.q_proj.bias", cfg->q_dim);
            LOAD_NORM(k_bias, "%slayers.%d.self_attn.k_proj.bias", cfg->kv_dim);
            LOAD_NORM(v_bias, "%slayers.%d.self_attn.v_proj.bias", cfg->kv_dim);
        }
        if (cfg->qk_norm) {
            LOAD_NORM(q_norm, "%slayers.%d.self_attn.q_norm.weight", cfg->head_dim);
            LOAD_NORM(k_norm, "%slayers.%d.self_attn.k_norm.weight", cfg->head_dim);
        }
        LOAD_NORM(input_norm, "%slayers.%d.input_layernorm.weight", cfg->hidden_size);
        LOAD_NORM(post_attn_norm, "%slayers.%d.post_attention_layernorm.weight", cfg->hidden_size);
        LOAD2(gate_proj, "%slayers.%d.mlp.gate_proj.weight", cfg->intermediate_size,
              cfg->hidden_size);
        LOAD2(up_proj, "%slayers.%d.mlp.up_proj.weight", cfg->intermediate_size, cfg->hidden_size);
        LOAD2(down_proj, "%slayers.%d.mlp.down_proj.weight", cfg->hidden_size,
              cfg->intermediate_size);

#undef LOAD2
#undef LOAD_NORM

        if (ffwd_verbose >= 2)
            fprintf(stderr, "  layer %d loaded\n", i);
    }

    /* Only the final stage applies the output RMSNorm. */
    if (layer_end == cfg->n_layers) {
        const int64_t norm_shape[1] = {cfg->hidden_size};
        snprintf(name, sizeof(name), "%snorm.weight", weight_prefix);
        w->norm = load_norm_f32(ms, name, norm_shape, 1);
        if (!w->norm)
            goto fail;
    }

    if (ffwd_verbose >= 1)
        fprintf(stderr,
                "ffwd_model_load: layers [%d, %d) loaded "
                "(%d-dim embeddings)\n",
                layer_start, layer_end, cfg->hidden_size);

    return model;

fail:
    ffwd_model_free(model);
    return NULL;
}

ffwd_model_t *ffwd_model_load(const char *model_dir) {
    return model_load_range_ex(model_dir, 0, -1, 0);
}

/* Release a weight ref's backing store. Only F16-widened refs own their buffer;
 * borrowed F32/BF16 views into the mmap are a no-op (see weight_ref_from_tensor). */
static void free_weight_ref(ffwd_weight_ref_t *w) {
    if (w->owned)
        free((void *)w->data);
    w->data = NULL;
    w->owned = 0;
}

void ffwd_model_free(ffwd_model_t *model) {
    if (!model)
        return;
    if (model->weights.layers) {
        for (int i = 0; i < model->config.n_layers; i++) {
            ffwd_layer_t *l = &model->weights.layers[i];
            free_weight_ref(&l->wq);
            free_weight_ref(&l->wk);
            free_weight_ref(&l->wv);
            free_weight_ref(&l->wo);
            free_weight_ref(&l->gate_proj);
            free_weight_ref(&l->up_proj);
            free_weight_ref(&l->down_proj);
            free((void *)l->q_bias);
            free((void *)l->k_bias);
            free((void *)l->v_bias);
            free((void *)l->q_norm);
            free((void *)l->k_norm);
            free((void *)l->input_norm);
            free((void *)l->post_attn_norm);
            free((void *)l->o_bias);
            free((void *)l->attn_ln_bias);
            free((void *)l->ffn_inter_bias);
            free((void *)l->ffn_out_bias);
            free((void *)l->ffn_ln_bias);
        }
    }
    free_weight_ref(&model->weights.embed_tokens);
    free((void *)model->weights.norm);
    free((void *)model->weights.position_embeddings);
    free((void *)model->weights.token_type_embeddings);
    free((void *)model->weights.ffwd_ln_w);
    free((void *)model->weights.ffwd_ln_b);
    if (model->safetensors)
        multi_safetensors_close((multi_safetensors_t *)model->safetensors);
    free(model->weights.layers);
    free(model);
}

const ffwd_config_t *ffwd_model_config(const ffwd_model_t *model) {
    return model ? &model->config : NULL;
}
