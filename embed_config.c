#include "embed_config.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
        p++;
    return p;
}

static const char *json_find_key(const char *json, const char *key) {
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strstr(p, key)) != NULL) {
        if (p > json && p[-1] == '"') {
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
    return v ? atoi(v) : fallback;
}

static double json_get_double(const char *json, const char *key, double fallback) {
    const char *v = json_find_key(json, key);
    return v ? atof(v) : fallback;
}

static int json_string_equals(const char *json, const char *key, const char *expected) {
    const char *v = json_find_key(json, key);
    if (!v || *v != '"')
        return 0;
    v++;
    size_t len = strlen(expected);
    return strncmp(v, expected, len) == 0 && v[len] == '"';
}

int embed_config_parse(embed_config_t *cfg, const char *model_dir) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/config.json", model_dir);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "embed_config: model path too long\n");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "embed_config: cannot open %s\n", path);
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

    if (cfg->hidden_size <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0 || cfg->n_kv_heads <= 0 ||
        cfg->head_dim <= 0 || cfg->intermediate_size <= 0 || cfg->vocab_size <= 0 ||
        cfg->q_dim <= 0 || cfg->kv_dim <= 0 || (cfg->head_dim & 1) ||
        cfg->n_heads % cfg->n_kv_heads != 0 || !isfinite(cfg->rms_norm_eps) ||
        cfg->rms_norm_eps <= 0.0f || !isfinite(cfg->rope_theta) || cfg->rope_theta <= 0.0f) {
        fprintf(stderr,
                "embed_config: invalid config in %s "
                "(hidden=%d, layers=%d, heads=%d/%d, head_dim=%d, inter=%d)\n",
                path, cfg->hidden_size, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim,
                cfg->intermediate_size);
        return -1;
    }
    if (cfg->n_layers > EMBED_MAX_LAYERS) {
        fprintf(stderr, "embed_config: too many layers (%d > %d)\n", cfg->n_layers,
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
