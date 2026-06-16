#include "server_internal.h"
#include "server_util.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ========================================================================
 * Model registry, load/lifecycle, and tokenize/inference dispatch
 *
 * k_models is the static table of every model the server can serve (name,
 * dimensions, attention/pooling mode, API family). load_one_model resolves a
 * served id to a slot, picks the tokenizer by the files present (WordPiece
 * vocab.txt -> BERT, else byte-level BPE vocab.json), and loads the CPU model
 * (GPU backends load on the worker thread). The tokenize_* and model_embed_*
 * helpers are the single dispatch point the request handlers call, hiding the
 * CPU/MLX/CUDA backend choice behind one signature.
 * ======================================================================== */

const model_info k_models[] = {
    {"pplx-embed-v1-0.6b", MODEL_KIND_STANDARD, 1024, 128, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_MEAN, 0, EMBED_API_PERPLEXITY, NULL},
    {"pplx-embed-v1-4b", MODEL_KIND_STANDARD, 2560, 128, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_MEAN, 0, EMBED_API_PERPLEXITY, NULL},
    {"Qwen3-Embedding-0.6B", MODEL_KIND_STANDARD, 1024, 32, 0, EMBED_ATTENTION_CAUSAL,
     EMBED_POOL_LAST_TOKEN, 1, EMBED_API_OPENAI, EMBED_QWEN3_QUERY_INSTRUCT},
    {"Qwen3-Embedding-4B", MODEL_KIND_STANDARD, 2560, 32, 0, EMBED_ATTENTION_CAUSAL,
     EMBED_POOL_LAST_TOKEN, 1, EMBED_API_OPENAI, EMBED_QWEN3_QUERY_INSTRUCT},
    {"Qwen3-Embedding-8B", MODEL_KIND_STANDARD, 4096, 32, 0, EMBED_ATTENTION_CAUSAL,
     EMBED_POOL_LAST_TOKEN, 1, EMBED_API_OPENAI, EMBED_QWEN3_QUERY_INSTRUCT},
    {"gte-Qwen2-1.5B-instruct", MODEL_KIND_STANDARD, 1536, 1536, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_LAST_TOKEN, 1, EMBED_API_OPENAI, EMBED_GTE_QWEN2_QUERY_INSTRUCT},
    /* BERT encoders: WordPiece tokenizer, learned positions, no instruction
     * prefix. all-MiniLM-L6-v2 mean-pools; bge-small-en-v1.5 pools the [CLS]
     * token. Both emit L2-normalized float32, so they speak the OpenAI API. */
    {"all-MiniLM-L6-v2", MODEL_KIND_STANDARD, 384, 384, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_MEAN, 1, EMBED_API_OPENAI, NULL},
    {"bge-small-en-v1.5", MODEL_KIND_STANDARD, 384, 384, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_CLS, 1, EMBED_API_OPENAI, NULL},
    {"bge-base-en-v1.5", MODEL_KIND_STANDARD, 768, 768, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_CLS, 1, EMBED_API_OPENAI, NULL},
    {"bge-large-en-v1.5", MODEL_KIND_STANDARD, 1024, 1024, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_CLS, 1, EMBED_API_OPENAI, NULL},
    {"pplx-embed-context-v1-0.6b", MODEL_KIND_CONTEXTUAL, 1024, 128, 0,
     EMBED_ATTENTION_BIDIRECTIONAL, EMBED_POOL_MEAN, 0, EMBED_API_PERPLEXITY, NULL},
    {"pplx-embed-context-v1-4b", MODEL_KIND_CONTEXTUAL, 2560, 128, 0, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_MEAN, 0, EMBED_API_PERPLEXITY, NULL},
    {"pplx-embed-v1-late-0.6b", MODEL_KIND_LATE, 1024, 128, 128, EMBED_ATTENTION_BIDIRECTIONAL,
     EMBED_POOL_MEAN, 0, EMBED_API_PERPLEXITY, NULL},
};

model_slot model_slot_for_id(const char *id) {
    if (!id)
        return MODEL_UNKNOWN;
    for (int i = 0; i < MODEL_COUNT; i++) {
        if (!strcmp(id, k_models[i].id))
            return (model_slot)i;
    }
    return MODEL_UNKNOWN;
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
    if (!m || !m->info || !config || config->hidden_size != m->info->dim ||
        token_dim != m->info->token_dim || config->attention_mode != m->info->attention_mode ||
        config->pooling_mode != m->info->pooling_mode ||
        config->normalize_embeddings != m->info->normalize_embeddings) {
        if (m && m->info && config)
            server_log("embed-server: model %s has incompatible config", m->info->id);
        return -1;
    }
    m->append_terminal_token = config->append_terminal_token;
    /* m->terminal_token_id is resolved from the tokenizer in load_one_model. */
    m->renormalize_truncated =
        config->normalize_embeddings && config->pooling_mode == EMBED_POOL_LAST_TOKEN;
    return 0;
}

int load_one_model(http_server *s, model_slot slot, const char *path) {
    loaded_model *m = &s->models[slot];
    m->info = &k_models[slot];
    m->path = xstrdup(path);
    m->terminal_token_id = -1;
    m->cls_id = -1;
    m->sep_id = -1;
    /* Pick the tokenizer by the files present: a WordPiece vocab.txt marks a
     * BERT encoder; otherwise the Qwen byte-level BPE vocab.json. The two never
     * coexist in a model directory, so the probe is unambiguous. */
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
    for (int i = 0; i < MODEL_COUNT; i++) {
        loaded_model *m = &s->models[i];
        free(m->path);
        if (m->tok_ws)
            embed_tokenizer_workspace_free(m->tok_ws);
        if (m->tok)
            embed_tokenizer_free(m->tok);
        if (m->wp_tok_ws)
            wordpiece_workspace_free(m->wp_tok_ws);
        if (m->wp_tok)
            wordpiece_tokenizer_free(m->wp_tok);
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
}
