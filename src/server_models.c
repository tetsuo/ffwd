#include "server_internal.h"
#include "server_util.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ========================================================================
 * Model load/lifecycle, and tokenize/inference dispatch
 *
 * Each loaded model owns resolved serving metadata for the operator-chosen
 * label. Transformer behavior comes from the model files through embed_config;
 * server-only facts absent from those files, such as output API family and
 * Matryoshka floor, come from the load spec. The tokenize_* and model_embed_*
 * helpers are the single dispatch point the request handlers call, hiding the
 * CPU/MLX/CUDA backend choice behind one signature.
 * ======================================================================== */

loaded_model *loaded_model_for_label(http_server *s, const char *label) {
    if (!label)
        return NULL;
    for (int i = 0; i < s->n_models; i++) {
        if (s->models[i].info && !strcmp(label, s->models[i].info->id))
            return &s->models[i];
    }
    return NULL;
}

static model_kind model_kind_from_public(embed_server_model_kind_t kind) {
    switch (kind) {
    case EMBED_SERVER_MODEL_CONTEXTUAL:
        return MODEL_KIND_CONTEXTUAL;
    case EMBED_SERVER_MODEL_LATE:
        return MODEL_KIND_LATE;
    case EMBED_SERVER_MODEL_STANDARD:
    default:
        return MODEL_KIND_STANDARD;
    }
}

static embedding_api_t embedding_api_from_public(embed_server_embedding_api_t api) {
    return api == EMBED_SERVER_API_PERPLEXITY ? EMBED_API_PERPLEXITY : EMBED_API_OPENAI;
}

static int read_optional_text_file(const char *model_dir, const char *relative, char **out) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", model_dir, relative);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        server_log("embed-server: model path too long");
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f && errno == ENOENT)
        return 0;
    if (!f) {
        server_log("embed-server: failed to open %s", path);
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
    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) {
        server_log("embed-server: invalid config_sentence_transformers.json: %s", model_dir);
        return -1;
    }
    cJSON *prompts = cJSON_GetObjectItemCaseSensitive(root, "prompts");
    cJSON *query = prompts ? cJSON_GetObjectItemCaseSensitive(prompts, "query") : NULL;
    if (query && !cJSON_IsString(query)) {
        cJSON_Delete(root);
        server_log("embed-server: prompts.query must be a string: %s", model_dir);
        return -1;
    }
    if (query && query->valuestring && query->valuestring[0]) {
        *out = xstrdup(query->valuestring);
        cJSON_Delete(root);
        return *out ? 1 : -1;
    }
    cJSON_Delete(root);
    return 0;
}

static int init_model_info(model_info **out, const embed_server_model_spec_t *spec) {
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
        cJSON_Delete(r->root);
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
        cJSON_Delete(r->root);
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
        cJSON_Delete(r->root);
    memset(r, 0, sizeof(*r));
}

int tokenize_one(loaded_model *m, job *j, const char *text, token_buf *out) {
    memset(out, 0, sizeof(*out));
    uint64_t t0 = nstime();
    if (m->wp_tok) {
        int n = 0;
        int *core = wordpiece_tokenizer_encode_with_workspace(m->wp_tok, m->wp_tok_ws, text, &n);
        if (j)
            j->tokenize_ns += nstime() - t0;
        /* BERT wraps the WordPiece ids with [CLS] ... [SEP], exactly as the BPE
         * branch below appends its terminal token. A whitespace-only input
         * cleans to zero core tokens (core == NULL, n == 0); the bare
         * [CLS][SEP] pair is what Hugging Face produces for it too. Grow the
         * returned buffer in place so only one allocation is touched. */
        if (n < 0 || n > INT_MAX - 2) {
            free(core);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        int *ids = realloc(core, (size_t)(n + 2) * sizeof(*ids));
        if (!ids) {
            free(core);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        memmove(ids + 1, ids, (size_t)n * sizeof(*ids));
        ids[0] = m->cls_id;
        ids[n + 1] = m->sep_id;
        out->ids = ids;
        out->n_tokens = n + 2;
        return 0;
    }
    if (m->sp_tok) {
        int n = 0;
        int *core =
            sentencepiece_tokenizer_encode_with_workspace(m->sp_tok, m->sp_tok_ws, text, &n);
        if (j)
            j->tokenize_ns += nstime() - t0;
        if (n < 0 || n > INT_MAX - 2) {
            free(core);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        int *ids = realloc(core, (size_t)(n + 2) * sizeof(*ids));
        if (!ids) {
            free(core);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        memmove(ids + 1, ids, (size_t)n * sizeof(*ids));
        ids[0] = m->cls_id;
        ids[n + 1] = m->sep_id;
        out->ids = ids;
        out->n_tokens = n + 2;
        return 0;
    }
    out->ids = embed_tokenizer_encode_with_workspace(m->tok, m->tok_ws, text, &out->n_tokens);
    if (j)
        j->tokenize_ns += nstime() - t0;
    if (!out->ids || out->n_tokens <= 0) {
        free(out->ids);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    if (m->append_terminal_token) {
        if (out->n_tokens == INT_MAX) {
            free(out->ids);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        int *ids = realloc(out->ids, (size_t)(out->n_tokens + 1) * sizeof(*out->ids));
        if (!ids) {
            free(out->ids);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        out->ids = ids;
        out->ids[out->n_tokens++] = m->terminal_token_id;
    }
    return 0;
}

/* Tokenize one embedding input, optionally prepending the model's query
 * instruction. The instruction and input are tokenized in one pass. */
int tokenize_input(
    loaded_model *m, job *j, const char *text, const char *query_instruct, token_buf *out) {
    if (!query_instruct)
        return tokenize_one(m, j, text, out);
    size_t plen = strlen(query_instruct);
    size_t tlen = strlen(text);
    char *buf = (char *)malloc(plen + tlen + 1);
    if (!buf) {
        memset(out, 0, sizeof(*out));
        return -1;
    }
    memcpy(buf, query_instruct, plen);
    memcpy(buf + plen, text, tlen + 1);
    int rc = tokenize_one(m, j, buf, out);
    free(buf);
    return rc;
}

static int late_id_is_skipped(const loaded_model *m, int id) {
    for (int i = 0; i < m->n_late_skip_ids; i++) {
        if (m->late_skip_ids[i] == id)
            return 1;
    }
    return 0;
}

int tokenize_late_text(loaded_model *m, job *j, const char *text, int is_query, late_tokens *out) {
    token_buf raw = {0};
    if (tokenize_one(m, j, text, &raw) != 0)
        return -1;

    int raw_tokens = raw.n_tokens;
    int target = raw_tokens + 1;
    if (is_query && target < EMBED_LATE_QUERY_TOKENS)
        target = EMBED_LATE_QUERY_TOKENS;

    out->ids = xmalloc((size_t)target * sizeof(*out->ids));
    out->ids[0] = raw.ids[0];
    out->ids[1] = is_query ? m->late_query_prefix_id : m->late_document_prefix_id;
    if (raw_tokens > 1) {
        memcpy(out->ids + 2, raw.ids + 1, (size_t)(raw_tokens - 1) * sizeof(*out->ids));
    }
    out->n_tokens = raw_tokens + 1;
    if (is_query) {
        while (out->n_tokens < EMBED_LATE_QUERY_TOKENS)
            out->ids[out->n_tokens++] = m->late_mask_id;
        out->n_keep = out->n_tokens;
    } else {
        out->keep = xmalloc((size_t)out->n_tokens * sizeof(*out->keep));
        for (int i = 0; i < out->n_tokens; i++) {
            if (!late_id_is_skipped(m, out->ids[i]))
                out->keep[out->n_keep++] = i;
        }
    }
    free(raw.ids);
    return out->n_keep > 0 ? 0 : -1;
}

int model_embed_batch(loaded_model *m, const embed_input_t *inputs, int batch, float *out) {
#ifdef USE_MLX
    if (m->mlx_ctx)
        return embed_mlx_encode_batch(m->mlx_ctx, inputs, batch, out);
#endif
#ifdef USE_CUDA
    if (m->cuda_ctx)
        return embed_cuda_encode_batch(m->cuda_ctx, inputs, batch, out);
#endif
    return embed_model_encode_batch(m->cpu_model, m->cpu_ws, inputs, batch, out);
}

int model_embed_spans_batch(loaded_model *m,
                            const embed_context_input_t *inputs,
                            int batch,
                            float *out) {
#ifdef USE_MLX
    if (m->mlx_ctx)
        return embed_mlx_encode_spans_batch(m->mlx_ctx, inputs, batch, out);
#endif
#ifdef USE_CUDA
    if (m->cuda_ctx)
        return embed_cuda_encode_spans_batch(m->cuda_ctx, inputs, batch, out);
#endif
    return embed_model_encode_spans_batch(m->cpu_model, m->cpu_ws, inputs, batch, out);
}

static int model_uses_dense_batches(const loaded_model *m) {
#ifdef USE_MLX
    return m->mlx_ctx != NULL;
#else
    (void)m;
    return 0;
#endif
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

int configure_loaded_model(loaded_model *m, const embed_config_t *config, int token_dim) {
    if (!m || !m->info || !config || config->hidden_size <= 0) {
        if (m && m->info)
            server_log("embed-server: model %s has invalid config", m->info->id);
        return -1;
    }
    if (m->info->kind == MODEL_KIND_LATE) {
        if (token_dim <= 0) {
            server_log("embed-server: late model %s is missing token projection metadata",
                       m->info->id);
            return -1;
        }
    } else if (token_dim != 0) {
        server_log("embed-server: non-late model %s unexpectedly has token projection metadata",
                   m->info->id);
        return -1;
    }

    int min_dim = m->info->min_dim > 0 ? m->info->min_dim : config->hidden_size;
    if (min_dim <= 0 || min_dim > config->hidden_size) {
        server_log("embed-server: model %s has invalid min_dim=%d for dim=%d", m->info->id,
                   min_dim, config->hidden_size);
        return -1;
    }
    m->info->dim = config->hidden_size;
    m->info->min_dim = min_dim;
    m->info->token_dim = token_dim;
    m->info->attention_mode = config->attention_mode;
    m->info->pooling_mode = config->pooling_mode;
    m->info->normalize_embeddings = config->normalize_embeddings;
    m->append_terminal_token = config->append_terminal_token;
    /* m->terminal_token_id is resolved from the tokenizer in load_one_model. */
    m->renormalize_truncated =
        config->normalize_embeddings && config->pooling_mode == EMBED_POOL_LAST_TOKEN;
    return 0;
}

int load_one_model(http_server *s, const embed_server_model_spec_t *spec) {
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
    m->terminal_token_id = -1;
    m->cls_id = -1;
    m->sep_id = -1;
    /* Pick the tokenizer by the files present: WordPiece vocab.txt, then a
     * SentencePiece .model, otherwise the Qwen byte-level BPE vocab.json. */
    char wp_path[1024];
    snprintf(wp_path, sizeof(wp_path), "%s/vocab.txt", path);
    if (access(wp_path, R_OK) == 0) {
        m->wp_tok = wordpiece_tokenizer_load(path);
        if (!m->wp_tok) {
            server_log("embed-server: failed to load WordPiece tokenizer: %s", wp_path);
            return -1;
        }
        m->wp_tok_ws = wordpiece_workspace_new();
        if (!m->wp_tok_ws) {
            server_log("embed-server: failed to allocate WordPiece workspace: %s", path);
            return -1;
        }
        /* [CLS]/[SEP] wrap every input; resolve their ids once from the vocab. */
        m->cls_id = wordpiece_tokenizer_token_id(m->wp_tok, "[CLS]");
        m->sep_id = wordpiece_tokenizer_token_id(m->wp_tok, "[SEP]");
        if (m->cls_id < 0 || m->sep_id < 0) {
            server_log("embed-server: WordPiece vocab missing [CLS]/[SEP]: %s", path);
            return -1;
        }
    } else {
        char sp_path[1024];
        snprintf(sp_path, sizeof(sp_path), "%s/sentencepiece.bpe.model", path);
        if (access(sp_path, R_OK) != 0) {
            snprintf(sp_path, sizeof(sp_path), "%s/tokenizer.model", path);
            if (access(sp_path, R_OK) != 0)
                snprintf(sp_path, sizeof(sp_path), "%s/spiece.model", path);
        }
        if (access(sp_path, R_OK) == 0) {
            m->sp_tok = sentencepiece_tokenizer_load(path);
            if (!m->sp_tok) {
                server_log("embed-server: failed to load SentencePiece tokenizer: %s", sp_path);
                return -1;
            }
            m->sp_tok_ws = sentencepiece_workspace_new();
            if (!m->sp_tok_ws) {
                server_log("embed-server: failed to allocate SentencePiece workspace: %s", path);
                return -1;
            }
            m->cls_id = sentencepiece_tokenizer_token_id(m->sp_tok, "<s>");
            m->sep_id = sentencepiece_tokenizer_token_id(m->sp_tok, "</s>");
            if (m->cls_id < 0 || m->sep_id < 0) {
                server_log("embed-server: SentencePiece vocab missing <s>/</s>: %s", path);
                return -1;
            }
        }
    }
    if (!m->wp_tok && !m->sp_tok) {
        char vocab_path[1024];
        snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", path);
        m->tok = embed_tokenizer_load(vocab_path);
        if (!m->tok) {
            server_log("embed-server: failed to load tokenizer: %s", vocab_path);
            return -1;
        }
        m->tok_ws = embed_tokenizer_workspace_new();
        if (!m->tok_ws) {
            server_log("embed-server: failed to allocate tokenizer workspace: %s", path);
            return -1;
        }
        /* The contextual chunk separator is the tokenizer's <|endoftext|>. The
         * released models keep added special tokens out of vocab.json, so the
         * family constant is the normal case; a vocab that does define the
         * token (e.g. small test models) takes precedence so the id stays
         * inside that model's embedding table. */
        int sep_id = embed_tokenizer_token_id(m->tok, "<|endoftext|>");
        m->context_separator_id = sep_id >= 0 ? sep_id : EMBED_CONTEXT_SEPARATOR_TOKEN_ID;
        /* Qwen3-Embedding pools the last token, the tokenizer's <|endoftext|>
         * suffix - the same token as the separator, not the model's chat
         * eos_token_id (<|im_end|>). Resolve it here from the tokenizer. */
        m->terminal_token_id = m->context_separator_id;
        if (m->info->kind == MODEL_KIND_LATE) {
            int id = embed_tokenizer_token_id(m->tok, "[MASK]");
            m->late_mask_id = id >= 0 ? id : EMBED_LATE_MASK_TOKEN_ID;
            id = embed_tokenizer_token_id(m->tok, "[Q]");
            m->late_query_prefix_id = id >= 0 ? id : EMBED_LATE_QUERY_PREFIX_ID;
            id = embed_tokenizer_token_id(m->tok, "[D]");
            m->late_document_prefix_id = id >= 0 ? id : EMBED_LATE_DOCUMENT_PREFIX_ID;

            const char *punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
            for (const char *p = punct;
                 *p &&
                 m->n_late_skip_ids < (int)(sizeof(m->late_skip_ids) / sizeof(m->late_skip_ids[0]));
                 p++) {
                char text[2] = {*p, '\0'};
                int n_ids = 0;
                int *ids = embed_tokenizer_encode(m->tok, text, &n_ids);
                if (ids && n_ids == 1)
                    m->late_skip_ids[m->n_late_skip_ids++] = ids[0];
                free(ids);
            }
        }
    }
#ifdef USE_MLX
    if (s->use_mlx) {
        /* MLX streams are thread-local. The inference worker loads the MLX
         * model so all MLX arrays/streams are created, used, and freed by the
         * same thread. */
        return 0;
    }
#endif
#ifdef USE_CUDA
    if (s->use_cuda) {
        return 0;
    }
#endif
    const embed_config_t *config = NULL;
    int token_dim = 0;
    if (m->info->kind == MODEL_KIND_LATE) {
        m->cpu_late_model = embed_late_model_load(path);
        if (!m->cpu_late_model) {
            server_log("embed-server: failed to load late model: %s", path);
            return -1;
        }
        m->cpu_late_ws = embed_late_workspace_new(m->cpu_late_model);
        if (!m->cpu_late_ws) {
            server_log("embed-server: failed to allocate late workspace: %s", path);
            return -1;
        }
        config = embed_late_model_config(m->cpu_late_model);
        token_dim = embed_late_model_token_dim(m->cpu_late_model);
    } else {
        m->cpu_model = embed_model_load(path);
        if (!m->cpu_model) {
            server_log("embed-server: failed to load model: %s", path);
            return -1;
        }
        m->cpu_ws = embed_workspace_new(m->cpu_model);
        if (!m->cpu_ws) {
            server_log("embed-server: failed to allocate workspace: %s", path);
            return -1;
        }
        config = embed_model_config(m->cpu_model);
    }
    if (configure_loaded_model(m, config, token_dim) != 0) {
        return -1;
    }
    if (m->info->kind == MODEL_KIND_LATE &&
        (m->late_mask_id >= config->vocab_size || m->late_query_prefix_id >= config->vocab_size ||
         m->late_document_prefix_id >= config->vocab_size)) {
        server_log("embed-server: late special-token ids exceed model vocab");
        return -1;
    }
    return 0;
}

void free_models(http_server *s) {
    for (int i = 0; i < s->n_models; i++) {
        loaded_model *m = &s->models[i];
        free_model_info(m->info);
        free(m->path);
        if (m->tok_ws)
            embed_tokenizer_workspace_free(m->tok_ws);
        if (m->tok)
            embed_tokenizer_free(m->tok);
        if (m->wp_tok_ws)
            wordpiece_workspace_free(m->wp_tok_ws);
        if (m->wp_tok)
            wordpiece_tokenizer_free(m->wp_tok);
        if (m->sp_tok_ws)
            sentencepiece_workspace_free(m->sp_tok_ws);
        if (m->sp_tok)
            sentencepiece_tokenizer_free(m->sp_tok);
        if (m->cpu_ws)
            embed_workspace_free(m->cpu_ws);
        if (m->cpu_model)
            embed_model_free(m->cpu_model);
        if (m->cpu_late_ws)
            embed_late_workspace_free(m->cpu_late_ws);
        if (m->cpu_late_model)
            embed_late_model_free(m->cpu_late_model);
#ifdef USE_MLX
        if (m->mlx_ctx)
            embed_mlx_free(m->mlx_ctx);
        if (m->mlx_late_ctx)
            embed_mlx_late_free(m->mlx_late_ctx);
#endif
#ifdef USE_CUDA
        if (m->cuda_ctx)
            embed_cuda_free(m->cuda_ctx);
        if (m->cuda_late_ctx)
            embed_cuda_late_free(m->cuda_late_ctx);
#endif
    }
    free(s->models);
    s->models = NULL;
    s->n_models = 0;
}
