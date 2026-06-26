#include "config.h"

#include <yyjson.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Verbosity flag for optional stderr diagnostics across the codebase.
 * Defined in config.c because every backend links it. */
int ffwd_verbose = 0;

typedef enum {
    JSON_BOOL_MISSING = 0,
    JSON_BOOL_FALSE = 1,
    JSON_BOOL_TRUE = 2,
    JSON_BOOL_INVALID = -1,
} json_bool_t;

/* Read fields from a parsed JSON object.
 * yyjson getters tolerate NULL, so missing keys or wrong types cleanly return
 * the fallback.
 *
 * This replaces substring scanning, which could match a key anywhere in the
 * document, including inside nested sub-configs. A real parser avoids that
 * failure mode. */
static int json_get_int(yyjson_val *obj, const char *key, int fallback) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (yyjson_is_int(v))
        return (int)yyjson_get_int(v);
    if (yyjson_is_real(v))
        return (int)yyjson_get_real(v);
    return fallback;
}

static double json_get_double(yyjson_val *obj, const char *key, double fallback) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    return yyjson_is_num(v) ? yyjson_get_num(v) : fallback;
}

static int json_string_equals(yyjson_val *obj, const char *key, const char *expected) {
    const char *s = yyjson_get_str(yyjson_obj_get(obj, key));
    return s && strcmp(s, expected) == 0;
}

static json_bool_t json_get_bool(yyjson_val *obj, const char *key) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v)
        return JSON_BOOL_MISSING;
    if (yyjson_is_bool(v))
        return yyjson_get_bool(v) ? JSON_BOOL_TRUE : JSON_BOOL_FALSE;
    return JSON_BOOL_INVALID;
}

/* Return 1 with a NUL-terminated buffer, 0 when an optional file is absent,
 * and -1 for an existing file that could not be read. */
static int read_optional_file(const char *model_dir, const char *relative, char **out) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", model_dir, relative);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ffwd_config: model path too long\n");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f && errno == ENOENT)
        return 0;
    if (!f) {
        fprintf(stderr, "ffwd_config: cannot open %s\n", path);
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
    *out = buf;
    return 1;
}

static int parse_sentence_transformers_metadata(ffwd_config_t *cfg, const char *model_dir) {
    char *buf = NULL;
    int rc = read_optional_file(model_dir, "1_Pooling/config.json", &buf);
    if (rc < 0)
        return -1;
    if (rc > 0) {
        yyjson_doc *doc = yyjson_read(buf, strlen(buf), 0);
        yyjson_val *root = yyjson_doc_get_root(doc);
        json_bool_t mean = json_get_bool(root, "pooling_mode_mean_tokens");
        json_bool_t last = json_get_bool(root, "pooling_mode_lasttoken");
        json_bool_t cls = json_get_bool(root, "pooling_mode_cls_token");
        json_bool_t max = json_get_bool(root, "pooling_mode_max_tokens");
        json_bool_t sqrt_len = json_get_bool(root, "pooling_mode_mean_sqrt_len_tokens");
        json_bool_t weighted = json_get_bool(root, "pooling_mode_weightedmean_tokens");
        if (mean < 0 || last < 0 || cls < 0 || max < 0 || sqrt_len < 0 || weighted < 0) {
            fprintf(stderr, "ffwd_config: invalid pooling metadata\n");
            yyjson_doc_free(doc);
            free(buf);
            return -1;
        }
        int supported = (mean == JSON_BOOL_TRUE) + (last == JSON_BOOL_TRUE) + (cls == JSON_BOOL_TRUE);
        int unsupported =
            max == JSON_BOOL_TRUE || sqrt_len == JSON_BOOL_TRUE || weighted == JSON_BOOL_TRUE;
        if (supported != 1 || unsupported) {
            fprintf(stderr, "ffwd_config: unsupported Sentence Transformers pooling mode\n");
            yyjson_doc_free(doc);
            free(buf);
            return -1;
        }
        cfg->pooling_mode = cls == JSON_BOOL_TRUE    ? FFWD_POOL_CLS
                            : last == JSON_BOOL_TRUE ? FFWD_POOL_LAST_TOKEN
                                                     : FFWD_POOL_MEAN;
        yyjson_doc_free(doc);
        free(buf);
        buf = NULL;
    }

    rc = read_optional_file(model_dir, "modules.json", &buf);
    if (rc < 0)
        return -1;
    if (rc > 0) {
        /* modules.json is an array of module objects; the embeddings are
         * normalized iff one module's "type" is the Normalize class. Walking the
         * array is exact, unlike a substring match that any field could trip. */
        yyjson_doc *doc = yyjson_read(buf, strlen(buf), 0);
        yyjson_val *root = yyjson_doc_get_root(doc);
        int normalize = 0;
        size_t idx, n;
        yyjson_val *mod;
        yyjson_arr_foreach(root, idx, n, mod) {
            if (json_string_equals(mod, "type", "sentence_transformers.models.Normalize")) {
                normalize = 1;
                break;
            }
        }
        cfg->normalize_embeddings = normalize;
        yyjson_doc_free(doc);
        free(buf);
        buf = NULL;
    }

    rc = read_optional_file(model_dir, "tokenizer_config.json", &buf);
    if (rc < 0)
        return -1;
    if (rc > 0) {
        yyjson_doc *doc = yyjson_read(buf, strlen(buf), 0);
        yyjson_val *root = yyjson_doc_get_root(doc);
        json_bool_t add_eos = json_get_bool(root, "add_eos_token");
        if (add_eos < 0) {
            fprintf(stderr, "ffwd_config: invalid add_eos_token metadata\n");
            yyjson_doc_free(doc);
            free(buf);
            return -1;
        }
        if (add_eos != JSON_BOOL_MISSING)
            cfg->append_terminal_token = add_eos == JSON_BOOL_TRUE;
        yyjson_doc_free(doc);
        free(buf);
    }
    return 0;
}

int ffwd_config_parse(ffwd_config_t *cfg, const char *model_dir) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/config.json", model_dir);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ffwd_config: model path too long\n");
        return -1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ffwd_config: cannot open %s\n", path);
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

    yyjson_doc *doc = yyjson_read(buf, (size_t)sz, 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        fprintf(stderr, "ffwd_config: invalid JSON in %s\n", path);
        yyjson_doc_free(doc);
        free(buf);
        return -1;
    }

    cfg->hidden_size = json_get_int(root, "hidden_size", 0);
    cfg->n_layers = json_get_int(root, "num_hidden_layers", 0);
    cfg->n_heads = json_get_int(root, "num_attention_heads", 0);
    cfg->n_kv_heads = json_get_int(root, "num_key_value_heads", 0);
    cfg->head_dim = json_get_int(root, "head_dim", FFWD_HEAD_DIM);
    cfg->intermediate_size = json_get_int(root, "intermediate_size", 0);
    cfg->vocab_size = json_get_int(root, "vocab_size", FFWD_VOCAB_SIZE);
    cfg->rms_norm_eps = (float)json_get_double(root, "rms_norm_eps", 1e-6);
    cfg->layer_norm_eps = (float)json_get_double(root, "layer_norm_eps", 1e-12);
    cfg->rope_theta = (float)json_get_double(root, "rope_theta", 1000000.0);
    cfg->max_position_embeddings = json_get_int(root, "max_position_embeddings", 0);
    cfg->position_id_offset = 0;
    cfg->type_vocab_size = json_get_int(root, "type_vocab_size", 2);
    cfg->family = FFWD_FAMILY_QWEN3;
    cfg->qk_norm = 1;
    cfg->qkv_bias = 0;
    cfg->attention_mode = FFWD_ATTENTION_BIDIRECTIONAL;
    cfg->pooling_mode = FFWD_POOL_MEAN;
    cfg->normalize_embeddings = 0;
    cfg->append_terminal_token = 0;

    if (json_string_equals(root, "model_type", "qwen3")) {
        cfg->attention_mode = FFWD_ATTENTION_CAUSAL;
        cfg->pooling_mode = FFWD_POOL_LAST_TOKEN;
        cfg->normalize_embeddings = 1;
        cfg->append_terminal_token = 1;
    } else if (json_string_equals(root, "model_type", "qwen2")) {
        cfg->qk_norm = 0;
        cfg->qkv_bias = 1;
        cfg->attention_mode = FFWD_ATTENTION_CAUSAL;
    } else if (json_string_equals(root, "model_type", "bert") ||
               json_string_equals(root, "model_type", "roberta") ||
               json_string_equals(root, "model_type", "xlm-roberta")) {
        cfg->family = FFWD_FAMILY_BERT;
        cfg->qk_norm = 0;
        cfg->qkv_bias = 1; /* q/k/v/o and both dense layers carry bias */
        cfg->attention_mode = FFWD_ATTENTION_BIDIRECTIONAL;
        cfg->n_kv_heads = cfg->n_heads; /* full multi-head attention, no GQA */
        cfg->head_dim = cfg->n_heads > 0 ? cfg->hidden_size / cfg->n_heads : 0;
        if (json_string_equals(root, "model_type", "roberta") ||
            json_string_equals(root, "model_type", "xlm-roberta")) {
            int pad_id = json_get_int(root, "pad_token_id", 1);
            if (pad_id < 0 || pad_id == INT_MAX) {
                fprintf(stderr, "ffwd_config: invalid RoBERTa pad_token_id\n");
                yyjson_doc_free(doc);
                free(buf);
                return -1;
            }
            cfg->position_id_offset = pad_id + 1;
        }
        /* hidden_act selects the feed-forward GeLU curve. gelu_new and
         * gelu_pytorch_tanh are the same tanh approximation; everything else
         * (gelu, absent) is the exact erf GeLU the released encoders use. */
        if (json_string_equals(root, "hidden_act", "gelu_new") ||
            json_string_equals(root, "hidden_act", "gelu_pytorch_tanh")) {
            cfg->ffn_act = FFWD_ACT_GELU_TANH;
        }
        if (cfg->max_position_embeddings <= 0 ||
            cfg->position_id_offset >= cfg->max_position_embeddings || cfg->type_vocab_size <= 0 ||
            !isfinite(cfg->layer_norm_eps) || cfg->layer_norm_eps <= 0.0f) {
            fprintf(stderr, "ffwd_config: invalid BERT layer_norm_eps/max_position_embeddings\n");
            yyjson_doc_free(doc);
            free(buf);
            return -1;
        }
    }

    json_bool_t is_causal = json_get_bool(root, "is_causal");
    if (is_causal < 0) {
        fprintf(stderr, "ffwd_config: invalid is_causal in %s\n", path);
        yyjson_doc_free(doc);
        free(buf);
        return -1;
    }
    if (is_causal != JSON_BOOL_MISSING) {
        cfg->attention_mode =
            is_causal == JSON_BOOL_TRUE ? FFWD_ATTENTION_CAUSAL : FFWD_ATTENTION_BIDIRECTIONAL;
    }

    cfg->q_dim = 0;
    cfg->kv_dim = 0;
    if (cfg->n_heads > 0 && cfg->head_dim > 0 && cfg->n_heads <= INT_MAX / cfg->head_dim)
        cfg->q_dim = cfg->n_heads * cfg->head_dim;
    if (cfg->n_kv_heads > 0 && cfg->head_dim > 0 && cfg->n_kv_heads <= INT_MAX / cfg->head_dim)
        cfg->kv_dim = cfg->n_kv_heads * cfg->head_dim;

    yyjson_doc_free(doc);
    free(buf);

    if (parse_sentence_transformers_metadata(cfg, model_dir) != 0)
        return -1;

    if (cfg->hidden_size <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0 || cfg->n_kv_heads <= 0 ||
        cfg->head_dim <= 0 || cfg->intermediate_size <= 0 || cfg->vocab_size <= 0 ||
        cfg->q_dim <= 0 || cfg->kv_dim <= 0 || (cfg->head_dim & 1) ||
        cfg->n_heads % cfg->n_kv_heads != 0 || !isfinite(cfg->rms_norm_eps) ||
        cfg->rms_norm_eps <= 0.0f || !isfinite(cfg->rope_theta) || cfg->rope_theta <= 0.0f) {
        fprintf(stderr,
                "ffwd_config: invalid config in %s "
                "(hidden=%d, layers=%d, heads=%d/%d, head_dim=%d, inter=%d)\n",
                path, cfg->hidden_size, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads, cfg->head_dim,
                cfg->intermediate_size);
        return -1;
    }
    if (cfg->n_layers > FFWD_MAX_LAYERS) {
        fprintf(stderr, "ffwd_config: too many layers (%d > %d)\n", cfg->n_layers, FFWD_MAX_LAYERS);
        return -1;
    }

    if (ffwd_verbose >= 1)
        fprintf(stderr,
                "config: hidden=%d, layers=%d, heads=%d/%d, "
                "inter=%d, head_dim=%d, attention=%s, pooling=%s, qk_norm=%s, qkv_bias=%s\n",
                cfg->hidden_size, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
                cfg->intermediate_size, cfg->head_dim,
                cfg->attention_mode == FFWD_ATTENTION_CAUSAL ? "causal" : "bidirectional",
                cfg->pooling_mode == FFWD_POOL_LAST_TOKEN ? "last-token"
                : cfg->pooling_mode == FFWD_POOL_CLS      ? "cls"
                                                          : "mean",
                cfg->qk_norm ? "yes" : "no", cfg->qkv_bias ? "yes" : "no");

    return 0;
}
