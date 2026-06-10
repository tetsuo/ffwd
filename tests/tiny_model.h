/* tests/tiny_model.h - synthesize a tiny 1-layer model directory (config.json
 * + model.safetensors) for hermetic tests. Values are BF16-representable so an
 * F32 and a BF16 copy of the same model produce identical embeddings. */

#ifndef TINY_MODEL_H
#define TINY_MODEL_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t tm_f32_to_bf16(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (uint16_t)(u >> 16);
}

static float tm_bf16_to_f32(uint16_t b)
{
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Deterministic, BF16-exact value for element i of tensor `name`. */
static float tm_value(const char *name, size_t i)
{
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
    int rows, cols;     /* cols == 0 means a 1-D tensor of `rows` values */
} tm_spec_t;

static const tm_spec_t TM_SPECS[] = {
    {"embed_tokens.weight", 16, 4},
    {"layers.0.self_attn.q_proj.weight", 4, 4},
    {"layers.0.self_attn.k_proj.weight", 2, 4},
    {"layers.0.self_attn.v_proj.weight", 2, 4},
    {"layers.0.self_attn.o_proj.weight", 4, 4},
    {"layers.0.self_attn.q_norm.weight", 2, 0},
    {"layers.0.self_attn.k_norm.weight", 2, 0},
    {"layers.0.input_layernorm.weight", 4, 0},
    {"layers.0.post_attention_layernorm.weight", 4, 0},
    {"layers.0.mlp.gate_proj.weight", 8, 4},
    {"layers.0.mlp.up_proj.weight", 8, 4},
    {"layers.0.mlp.down_proj.weight", 4, 8},
    {"norm.weight", 4, 0},
};
enum { TM_N_SPECS = sizeof(TM_SPECS) / sizeof(TM_SPECS[0]) };

/* Write config.json + model.safetensors into dir. dtype: "F32" or "BF16".
 * Returns 0 on success. */
static int tm_write_model(const char *dir, const char *dtype)
{
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fputs("{\"hidden_size\":4,\"num_hidden_layers\":1,"
          "\"num_attention_heads\":2,\"num_key_value_heads\":1,"
          "\"head_dim\":2,\"intermediate_size\":8,\"vocab_size\":16,"
          "\"rms_norm_eps\":1e-6,\"rope_theta\":10000.0}", f);
    fclose(f);

    int bf16 = strcmp(dtype, "BF16") == 0;
    size_t esize = bf16 ? 2 : 4;

    char header[4096];
    size_t hoff = 0, doff = 0;
    hoff += (size_t)snprintf(header + hoff, sizeof(header) - hoff, "{");
    for (int t = 0; t < TM_N_SPECS; t++) {
        const tm_spec_t *s = &TM_SPECS[t];
        size_t n = (size_t)s->rows * (s->cols ? (size_t)s->cols : 1);
        size_t end = doff + n * esize;
        if (s->cols)
            hoff += (size_t)snprintf(header + hoff, sizeof(header) - hoff,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d],"
                "\"data_offsets\":[%zu,%zu]}",
                t ? "," : "", s->name, dtype, s->rows, s->cols, doff, end);
        else
            hoff += (size_t)snprintf(header + hoff, sizeof(header) - hoff,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%d],"
                "\"data_offsets\":[%zu,%zu]}",
                t ? "," : "", s->name, dtype, s->rows, doff, end);
        doff = end;
    }
    hoff += (size_t)snprintf(header + hoff, sizeof(header) - hoff, "}");
    if (hoff >= sizeof(header) - 1) return -1;

    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char len8[8];
    for (int i = 0; i < 8; i++) len8[i] = (unsigned char)(hoff >> (8 * i));
    fwrite(len8, 1, 8, f);
    fwrite(header, 1, hoff, f);
    for (int t = 0; t < TM_N_SPECS; t++) {
        const tm_spec_t *s = &TM_SPECS[t];
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

#endif /* TINY_MODEL_H */
