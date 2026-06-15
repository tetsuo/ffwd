/* tests/tiny_model.h - synthesize a tiny 1-layer model directory (config.json
 * + model.safetensors) for hermetic tests. Values are BF16-representable so an
 * F32 and a BF16 copy of the same model produce identical embeddings. */

#ifndef TINY_MODEL_H
#define TINY_MODEL_H

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static uint16_t tm_f32_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (uint16_t)(u >> 16);
}

static float tm_bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Deterministic, BF16-exact value for element i of tensor `name`. */
static float tm_value(const char *name, size_t i) {
    float v;
    size_t len = strlen(name);
    if ((len >= 11 && strcmp(name + len - 11, "norm.weight") == 0) ||
        strstr(name, "layernorm.weight"))
        v = 0.75f + 0.03f * (float)((i % 5) + 1);
    else if (strcmp(name, "embed_tokens.weight") == 0)
        v = sinf((float)i * 0.17f) * 0.2f;
    else
        v = sinf((float)i * 0.11f + (float)len) * 0.08f;
    return tm_bf16_to_f32(tm_f32_to_bf16(v));
}

typedef struct {
    const char *name;
    int rows, cols; /* cols == 0 means a 1-D tensor of `rows` values */
} tm_spec_t;

/* Always one layer; everything else is free so a test can match a real
 * model family's dimensions (e.g. hidden 1024 for the 0.6B server slot). */
typedef struct {
    int hidden, heads, kv_heads, head_dim, intermediate, vocab;
} tm_dims_t;

enum { TM_N_SPECS = 13 };

static int
tm_write_config(const char *dir, const tm_dims_t *d, const char *model_type, int eos_token_id) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f,
            "{\"hidden_size\":%d,\"num_hidden_layers\":1,"
            "\"num_attention_heads\":%d,\"num_key_value_heads\":%d,"
            "\"head_dim\":%d,\"intermediate_size\":%d,\"vocab_size\":%d,"
            "\"rms_norm_eps\":1e-6,\"rope_theta\":10000.0",
            d->hidden, d->heads, d->kv_heads, d->head_dim, d->intermediate, d->vocab);
    if (model_type)
        fprintf(f, ",\"model_type\":\"%s\",\"eos_token_id\":%d", model_type, eos_token_id);
    fputs("}", f);
    fclose(f);
    return 0;
}

/* Write a safetensors file holding `specs` with tm_value() contents.
 * dtype: "F32" or "BF16". Returns 0 on success. */
static int tm_write_safetensors_with_prefix(
    const char *path, const char *dtype, const tm_spec_t *specs, int n_specs, const char *prefix) {
    int bf16 = strcmp(dtype, "BF16") == 0;
    size_t esize = bf16 ? 2 : 4;

    char header[4096];
    size_t hoff = 0, doff = 0;
    hoff += (size_t)snprintf(header + hoff, sizeof(header) - hoff, "{");
    for (int t = 0; t < n_specs; t++) {
        const tm_spec_t *s = &specs[t];
        size_t n = (size_t)s->rows * (s->cols ? (size_t)s->cols : 1);
        size_t end = doff + n * esize;
        if (s->cols)
            hoff +=
                (size_t)snprintf(header + hoff, sizeof(header) - hoff,
                                 "%s\"%s%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],"
                                 "\"data_offsets\":[%zu,%zu]}",
                                 t ? "," : "", prefix, s->name, dtype, s->rows, s->cols, doff, end);
        else
            hoff += (size_t)snprintf(header + hoff, sizeof(header) - hoff,
                                     "%s\"%s%s\":{\"dtype\":\"%s\",\"shape\":[%d],"
                                     "\"data_offsets\":[%zu,%zu]}",
                                     t ? "," : "", prefix, s->name, dtype, s->rows, doff, end);
        doff = end;
    }
    hoff += (size_t)snprintf(header + hoff, sizeof(header) - hoff, "}");
    if (hoff >= sizeof(header) - 1)
        return -1;

    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    unsigned char len8[8];
    for (int i = 0; i < 8; i++)
        len8[i] = (unsigned char)(hoff >> (8 * i));
    fwrite(len8, 1, 8, f);
    fwrite(header, 1, hoff, f);
    for (int t = 0; t < n_specs; t++) {
        const tm_spec_t *s = &specs[t];
        size_t n = (size_t)s->rows * (s->cols ? (size_t)s->cols : 1);
        for (size_t i = 0; i < n; i++) {
            float v = tm_value(s->name, i);
            if (bf16) {
                uint16_t h = tm_f32_to_bf16(v);
                fwrite(&h, sizeof(h), 1, f);
            } else {
                fwrite(&v, sizeof(v), 1, f);
            }
        }
    }
    fclose(f);
    return 0;
}

static int
tm_write_safetensors(const char *path, const char *dtype, const tm_spec_t *specs, int n_specs) {
    return tm_write_safetensors_with_prefix(path, dtype, specs, n_specs, "");
}

/* Write config.json + model.safetensors with the given dimensions into dir.
 * dtype: "F32" or "BF16". Returns 0 on success. */
static int tm_write_model_dims_with_prefix(const char *dir,
                                           const char *dtype,
                                           const tm_dims_t *d,
                                           const char *prefix) {
    int q = d->heads * d->head_dim, kv = d->kv_heads * d->head_dim;
    const tm_spec_t specs[TM_N_SPECS] = {
        {"embed_tokens.weight", d->vocab, d->hidden},
        {"layers.0.self_attn.q_proj.weight", q, d->hidden},
        {"layers.0.self_attn.k_proj.weight", kv, d->hidden},
        {"layers.0.self_attn.v_proj.weight", kv, d->hidden},
        {"layers.0.self_attn.o_proj.weight", d->hidden, q},
        {"layers.0.self_attn.q_norm.weight", d->head_dim, 0},
        {"layers.0.self_attn.k_norm.weight", d->head_dim, 0},
        {"layers.0.input_layernorm.weight", d->hidden, 0},
        {"layers.0.post_attention_layernorm.weight", d->hidden, 0},
        {"layers.0.mlp.gate_proj.weight", d->intermediate, d->hidden},
        {"layers.0.mlp.up_proj.weight", d->intermediate, d->hidden},
        {"layers.0.mlp.down_proj.weight", d->hidden, d->intermediate},
        {"norm.weight", d->hidden, 0},
    };

    char path[2048];
    if (tm_write_config(dir, d, NULL, -1) != 0)
        return -1;

    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    return tm_write_safetensors_with_prefix(path, dtype, specs, TM_N_SPECS, prefix);
}

static int tm_write_model_dims(const char *dir, const char *dtype, const tm_dims_t *d) {
    return tm_write_model_dims_with_prefix(dir, dtype, d, "");
}

static inline int
tm_write_prefixed_model_dims(const char *dir, const char *dtype, const tm_dims_t *d) {
    return tm_write_model_dims_with_prefix(dir, dtype, d, "model.");
}

static inline int tm_write_qwen3_model_dims(const char *dir,
                                            const char *dtype,
                                            const tm_dims_t *d,
                                            int eos_token_id) {
    if (tm_write_model_dims(dir, dtype, d) != 0)
        return -1;
    return tm_write_config(dir, d, "qwen3", eos_token_id);
}

/* Add the 1_Dense per-token projection head that turns a base model dir into
 * a late-interaction snapshot. weight_name is "linear.weight" for a valid
 * head; pass another name to synthesize a malformed head. */
static inline int tm_write_late_projection_named(
    const char *dir, const char *dtype, int token_dim, int hidden, const char *weight_name) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/1_Dense", dir);
    if (mkdir(path, 0755) != 0 && errno != EEXIST)
        return -1;

    snprintf(path, sizeof(path), "%s/1_Dense/config.json", dir);
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f, "{\"in_features\":%d,\"out_features\":%d,\"bias\":false}", hidden, token_dim);
    fclose(f);

    const tm_spec_t spec = {weight_name, token_dim, hidden};
    snprintf(path, sizeof(path), "%s/1_Dense/model.safetensors", dir);
    return tm_write_safetensors(path, dtype, &spec, 1);
}

static inline int
tm_write_late_projection(const char *dir, const char *dtype, int token_dim, int hidden) {
    return tm_write_late_projection_named(dir, dtype, token_dim, hidden, "linear.weight");
}

/* The historical default fixture: 4-dim hidden, 16-token vocab. */
static inline int tm_write_model(const char *dir, const char *dtype) {
    static const tm_dims_t d = {4, 2, 1, 2, 8, 16};
    return tm_write_model_dims(dir, dtype, &d);
}

#endif /* TINY_MODEL_H */
