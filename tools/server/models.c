/* Model load/lifecycle, plus tokenization and inference dispatch.
 *
 * Each loaded model owns resolved serving metadata for the operator-chosen
 * label.
 *
 * Transformer behavior comes from model files through ffwd_config.
 * Server-only facts missing from those files, such as output API family and
 * Matryoshka floor, come from the load spec.
 *
 * tokenize_* and model_ffwd_* helpers are the request handlers' single dispatch
 * point, hiding the compiled backend choice behind one signature.
 */

#include "server_internal.h"
#include "util.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

loaded_model *loaded_model_for_label(http_server *s, const char *label) {
    if (!label)
        return NULL;
    for (int i = 0; i < s->n_models; i++) {
        if (s->models[i].info && !strcmp(label, s->models[i].info->id))
            return &s->models[i];
    }
    return NULL;
}

static model_kind model_kind_from_public(ffwd_server_model_kind_t kind) {
    switch (kind) {
    case FFWD_SERVER_MODEL_CONTEXTUAL:
        return MODEL_KIND_CONTEXTUAL;
    case FFWD_SERVER_MODEL_LATE:
        return MODEL_KIND_LATE;
    case FFWD_SERVER_MODEL_STANDARD:
    default:
        return MODEL_KIND_STANDARD;
    }
}

static embedding_api_t embedding_api_from_public(ffwd_server_embedding_api_t api) {
    return api == FFWD_SERVER_API_PERPLEXITY ? FFWD_API_PERPLEXITY : FFWD_API_OPENAI;
}

static int read_optional_text_file(const char *model_dir, const char *relative, char **out) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", model_dir, relative);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        server_log("ffwd-server: model path too long");
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f && errno == ENOENT)
        return 0;
    if (!f) {
        server_log("ffwd-server: failed to open %s", path);
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
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    buf[sz] = '\0';
    *out = buf;
    return 1;
}

static int read_query_instruct_from_files(const char *model_dir, char **out) {
    char *buf = NULL;
    int rc = read_optional_text_file(model_dir, "config_sentence_transformers.json", &buf);
    if (rc <= 0)
        return rc;
    yyjson_doc *doc = yyjson_read(buf, strlen(buf), 0);
    free(buf);
    if (!doc) {
        server_log("ffwd-server: invalid config_sentence_transformers.json: %s", model_dir);
        return -1;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *prompts = yyjson_obj_get(root, "prompts");
    yyjson_val *query = prompts ? yyjson_obj_get(prompts, "query") : NULL;
    if (query && !yyjson_is_str(query)) {
        yyjson_doc_free(doc);
        server_log("ffwd-server: prompts.query must be a string: %s", model_dir);
        return -1;
    }
    const char *qs = yyjson_get_str(query);
    if (qs && qs[0]) {
        *out = xstrdup(qs);
        yyjson_doc_free(doc);
        return *out ? 1 : -1;
    }
    yyjson_doc_free(doc);
    return 0;
}

static int init_model_info(model_info **out, const ffwd_server_model_spec_t *spec) {
    model_info *info = xcalloc(1, sizeof(*info));
    info->id = xstrdup(spec->id);
    info->kind = model_kind_from_public(spec->kind);
    info->api = embedding_api_from_public(spec->api);
    info->min_dim = spec->min_dim;
    if (spec->query_instruct) {
        info->query_instruct = xstrdup(spec->query_instruct);
    } else if (read_query_instruct_from_files(spec->path, &info->query_instruct) < 0) {
        free(info->id);
        free(info);
        return -1;
    }
    if (!info->id || (spec->query_instruct && !info->query_instruct)) {
        free(info->query_instruct);
        free(info->id);
        free(info);
        return -1;
    }
    *out = info;
    return 0;
}

static void free_model_info(model_info *info) {
    if (!info)
        return;
    free(info->id);
    free(info->query_instruct);
    free(info);
}

void free_token_bufs(token_buf *t, int n) {
    if (!t)
        return;
    for (int i = 0; i < n; i++)
        free(t[i].ids);
}

void embedding_request_free(embedding_request *r) {
    if (!r)
        return;
    free_token_bufs(r->tokens, r->n_inputs);
    free(r->tokens);
    free(r->inputs);
    if (r->root)
        yyjson_doc_free(r->root);
    memset(r, 0, sizeof(*r));
}

void contextual_request_free(contextual_request *r) {
    if (!r)
        return;
    for (int i = 0; i < r->n_docs; i++) {
        free(r->docs[i].ids);
        free(r->docs[i].spans);
    }
    free(r->docs);
    if (r->root)
        yyjson_doc_free(r->root);
    memset(r, 0, sizeof(*r));
}

void late_tokens_free(late_tokens *t) {
    if (!t)
        return;
    free(t->ids);
    free(t->keep);
    memset(t, 0, sizeof(*t));
}

void rerank_request_free(rerank_request *r) {
    if (!r)
        return;
    late_tokens_free(&r->query);
    for (int i = 0; i < r->n_documents; i++)
        late_tokens_free(&r->documents[i]);
    free(r->documents);
    if (r->root)
        yyjson_doc_free(r->root);
    memset(r, 0, sizeof(*r));
}

/* Thin wrappers that add server per-stage timing to the tokenization job.
 * Request handlers call them unchanged. */
int tokenize_input(
    loaded_model *m, job *j, const char *text, const char *query_instruct, token_buf *out) {
    uint64_t t0 = nstime();
    out->ids = ffwd_tokenize(m->tok, text, query_instruct, &out->n_tokens);
    if (j)
        j->tokenize_ns += nstime() - t0;
    if (!out->ids) {
        memset(out, 0, sizeof(*out));
        return -1;
    }
    return 0;
}

int tokenize_one(loaded_model *m, job *j, const char *text, token_buf *out) {
    return tokenize_input(m, j, text, NULL, out);
}

int tokenize_late_text(loaded_model *m, job *j, const char *text, int is_query, late_tokens *out) {
    uint64_t t0 = nstime();
    int rc = ffwd_tokenize_late(m->tok, text, is_query, out);
    if (j)
        j->tokenize_ns += nstime() - t0;
    return rc;
}

int model_ffwd_batch(loaded_model *m, const ffwd_input_t *inputs, int batch, float *out) {
    return ffwd_encode_batch(m->backend, inputs, batch, out);
}

int model_ffwd_spans_batch(loaded_model *m,
                           const ffwd_context_input_t *inputs,
                           int batch,
                           float *out) {
    return ffwd_encode_spans_batch(m->backend, inputs, batch, out);
}

static int model_uses_dense_batches(const loaded_model *m) {
    return ffwd_uses_dense_batches(m->backend);
}

/* Inputs are sorted by length before chunking. CPU packs real tokens while MLX
 * pads each dense row to the longest input in the chunk. */
int inference_batch_accepts_input(
    const loaded_model *m, int batch, int packed_tokens, int next_tokens, int max_batch_tokens) {
    if (batch == 0)
        return 1;
    if (model_uses_dense_batches(m))
        return next_tokens <= max_batch_tokens / (batch + 1);
    return next_tokens <= max_batch_tokens - packed_tokens;
}

int configure_loaded_model(loaded_model *m, const ffwd_config_t *config, int token_dim) {
    if (!m || !m->info || !config || config->hidden_size <= 0) {
        if (m && m->info)
            server_log("ffwd-server: model %s has invalid config", m->info->id);
        return -1;
    }
    if (m->info->kind == MODEL_KIND_LATE) {
        if (token_dim <= 0) {
            server_log("ffwd-server: late model %s is missing token projection metadata",
                       m->info->id);
            return -1;
        }
    } else if (token_dim != 0) {
        server_log("ffwd-server: non-late model %s unexpectedly has token projection metadata",
                   m->info->id);
        return -1;
    }

    int min_dim = m->info->min_dim > 0 ? m->info->min_dim : config->hidden_size;
    if (min_dim <= 0 || min_dim > config->hidden_size) {
        server_log("ffwd-server: model %s has invalid min_dim=%d for dim=%d", m->info->id, min_dim,
                   config->hidden_size);
        return -1;
    }
    m->info->dim = config->hidden_size;
    m->info->min_dim = min_dim;
    m->info->token_dim = token_dim;
    m->info->attention_mode = config->attention_mode;
    m->info->pooling_mode = config->pooling_mode;
    m->info->normalize_embeddings = config->normalize_embeddings;
    m->renormalize_truncated =
        config->normalize_embeddings && config->pooling_mode == FFWD_POOL_LAST_TOKEN;
    return 0;
}

/* Apply the loaded model's config to its serving metadata and validate it. CPU
 * runs this in load_one_model; a GPU build runs it on the inference thread once
 * ffwd_activate has created the context. */
int finalize_loaded_model(loaded_model *m) {
    const ffwd_config_t *config = ffwd_config(m->backend);
    int token_dim = ffwd_token_dim(m->backend);
    ffwd_tok_set_append_terminal(m->tok, config->append_terminal_token);
    return configure_loaded_model(m, config, token_dim);
}

int load_one_model(http_server *s, const ffwd_server_model_spec_t *spec) {
    /* Grow the model list by one entry. */
    loaded_model *nm = realloc(s->models, (size_t)(s->n_models + 1) * sizeof(*s->models));
    if (!nm)
        return -1;
    s->models = nm;
    loaded_model *m = &s->models[s->n_models];
    memset(m, 0, sizeof(*m));
    s->n_models++;

    if (init_model_info(&m->info, spec) != 0)
        return -1;
    const char *path = spec->path;
    m->path = xstrdup(path);
    char err[256];
    /* The tokenizer (file probe, special-token resolution, [CLS]/[SEP] vs Qwen
     * terminal-token layout) is owned by libffwd now. */
    m->tok = ffwd_tok_open(path, m->info->kind == MODEL_KIND_LATE, err, sizeof(err));
    if (!m->tok) {
        server_log("ffwd-server: %s: %s", path, err);
        return -1;
    }

    m->backend =
        ffwd_open(path, m->info->kind == MODEL_KIND_LATE, &s->backend_opts, err, sizeof(err));
    if (!m->backend) {
        server_log("ffwd-server: %s: %s", path, err);
        return -1;
    }
    /* CPU loads the model in open, so its config is ready to finalize now. A GPU
     * build defers the load - and this finalize - to the inference thread. */
    if (ffwd_config(m->backend))
        return finalize_loaded_model(m);
    return 0;
}

void free_models(http_server *s) {
    for (int i = 0; i < s->n_models; i++) {
        loaded_model *m = &s->models[i];
        free_model_info(m->info);
        free(m->path);
        ffwd_tok_free(m->tok);
        ffwd_free(m->backend);
    }
    free(s->models);
    s->models = NULL;
    s->n_models = 0;
}
