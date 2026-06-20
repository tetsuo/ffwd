#include "server_internal.h"
#include "util.h"
#include "ffwd.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Endpoint handlers: prepare, batch, execute, render
 *
 * One prepare_* per endpoint validates the request JSON and tokenizes its
 * inputs into a *_request; the worker (schedule.c) groups compatible
 * ready requests and calls the matching execute_*_request_list, which runs the
 * batched forward pass through model_ffwd_* and renders each response via
 * encode.c. Rerank runs MaxSim over late-interaction token vectors.
 * ======================================================================== */

int embedding_request_compatible(const embedding_request *a, const embedding_request *b) {
    return a->ready && b->ready && a->model == b->model && a->dims == b->dims &&
           !strcmp(a->encoding, b->encoding);
}

typedef struct {
    ffwd_input_t input;
    int output_index;
} embedding_batch_item;

static int embedding_batch_item_cmp(const void *a, const void *b) {
    const embedding_batch_item *ia = a;
    const embedding_batch_item *ib = b;
    if (ia->input.n_tokens < ib->input.n_tokens)
        return -1;
    if (ia->input.n_tokens > ib->input.n_tokens)
        return 1;
    return ia->output_index - ib->output_index;
}

int embedding_request_fits_group(const embedding_request *r,
                                 int group_inputs,
                                 int group_tokens,
                                 int max_batch,
                                 int max_batch_tokens) {
    if (group_inputs == 0)
        return 1;
    return r->n_inputs <= max_batch - group_inputs &&
           r->total_tokens <= max_batch_tokens - group_tokens;
}

void execute_embedding_request_list(embedding_request **reqs, int n_reqs) {
    if (n_reqs <= 0)
        return;
    loaded_model *m = reqs[0]->model;
    int dim = m->info->dim;
    int total_inputs = 0;
    for (int i = 0; i < n_reqs; i++)
        total_inputs += reqs[i]->n_inputs;

    embedding_batch_item *items = xmalloc((size_t)total_inputs * sizeof(*items));
    float *embs = xmalloc((size_t)total_inputs * dim * sizeof(float));

    int pos = 0;
    for (int i = 0; i < n_reqs; i++) {
        for (int k = 0; k < reqs[i]->n_inputs; k++) {
            items[pos].input = reqs[i]->inputs[k];
            items[pos].output_index = pos;
            pos++;
        }
    }
    qsort(items, (size_t)total_inputs, sizeof(*items), embedding_batch_item_cmp);

    int max_batch = reqs[0]->j->srv->batch_size > 0 ? reqs[0]->j->srv->batch_size : total_inputs;
    if (max_batch > total_inputs)
        max_batch = total_inputs;
    int max_batch_tokens = reqs[0]->j->srv->max_batch_tokens;
    ffwd_input_t *inputs = xmalloc((size_t)max_batch * sizeof(*inputs));
    float *batch_embs = xmalloc((size_t)max_batch * dim * sizeof(float));

    int failed = 0;
    uint64_t infer_ns = 0;
    for (int start = 0; start < total_inputs;) {
        int cur = 0;
        int tokens = 0;
        while (start + cur < total_inputs && cur < max_batch) {
            int n_tokens = items[start + cur].input.n_tokens;
            if (!inference_batch_accepts_input(m, cur, tokens, n_tokens, max_batch_tokens))
                break;
            inputs[cur] = items[start + cur].input;
            tokens += n_tokens;
            cur++;
        }
        uint64_t t0 = nstime();
        int ffwd_rc = model_ffwd_batch(m, inputs, cur, batch_embs);
        infer_ns += nstime() - t0;
        if (ffwd_rc != 0) {
            failed = 1;
            break;
        }
        for (int i = 0; i < cur; i++)
            memcpy(embs + (size_t)items[start + i].output_index * dim, batch_embs + (size_t)i * dim,
                   (size_t)dim * sizeof(float));
        start += cur;
    }

    pos = 0;
    for (int i = 0; i < n_reqs; i++) {
        reqs[i]->j->infer_ns += infer_ns;
        if (failed) {
            job_set_error(reqs[i]->j, 500, "embedding failed", "server_error");
        } else {
            uint64_t t0 = nstime();
            if (render_embedding_response(reqs[i], embs + (size_t)pos * dim) != 0)
                job_set_error(reqs[i]->j, 500, "embedding normalization failed", "server_error");
            reqs[i]->j->encode_ns += nstime() - t0;
        }
        pos += reqs[i]->n_inputs;
    }

    free(batch_embs);
    free(inputs);
    free(embs);
    free(items);
}

int contextual_request_compatible(const contextual_request *a, const contextual_request *b) {
    return a->ready && b->ready && a->model == b->model && a->dims == b->dims &&
           !strcmp(a->encoding, b->encoding);
}

int contextual_request_fits_group(const contextual_request *r,
                                  int group_docs,
                                  int group_tokens,
                                  int max_batch,
                                  int max_batch_tokens) {
    if (group_docs == 0)
        return 1;
    return r->n_docs <= max_batch - group_docs && r->total_tokens <= max_batch_tokens - group_tokens;
}

typedef struct {
    ffwd_context_input_t input;
    int output_span_index;
    int order;
} contextual_batch_item;

static int contextual_batch_item_cmp(const void *a, const void *b) {
    const contextual_batch_item *ia = a;
    const contextual_batch_item *ib = b;
    if (ia->input.input.n_tokens < ib->input.input.n_tokens)
        return -1;
    if (ia->input.input.n_tokens > ib->input.input.n_tokens)
        return 1;
    return ia->order - ib->order;
}

void execute_contextual_request_list(contextual_request **reqs, int n_reqs) {
    if (n_reqs <= 0)
        return;
    loaded_model *m = reqs[0]->model;
    int dim = m->info->dim;
    int total_docs = 0;
    int total_chunks = 0;
    for (int i = 0; i < n_reqs; i++) {
        if (total_docs > INT_MAX - reqs[i]->n_docs ||
            total_chunks > INT_MAX - reqs[i]->total_chunks) {
            for (int k = 0; k < n_reqs; k++)
                job_set_error(reqs[k]->j, 500, "contextual batch is too large", "server_error");
            return;
        }
        total_docs += reqs[i]->n_docs;
        total_chunks += reqs[i]->total_chunks;
    }
    if ((size_t)total_chunks > SIZE_MAX / (size_t)dim / sizeof(float)) {
        for (int i = 0; i < n_reqs; i++)
            job_set_error(reqs[i]->j, 500, "contextual batch is too large", "server_error");
        return;
    }

    contextual_batch_item *items = xmalloc((size_t)total_docs * sizeof(*items));
    float *embs = xmalloc((size_t)total_chunks * dim * sizeof(*embs));
    int pos = 0;
    int span_offset = 0;
    for (int i = 0; i < n_reqs; i++) {
        for (int d = 0; d < reqs[i]->n_docs; d++) {
            contextual_doc *doc = &reqs[i]->docs[d];
            items[pos].input.input.ids = doc->ids;
            items[pos].input.input.n_tokens = doc->n_tokens;
            items[pos].input.spans = doc->spans;
            items[pos].input.n_spans = doc->n_spans;
            items[pos].output_span_index = span_offset;
            items[pos].order = pos;
            span_offset += doc->n_spans;
            pos++;
        }
    }
    qsort(items, (size_t)total_docs, sizeof(*items), contextual_batch_item_cmp);

    int max_batch = reqs[0]->j->srv->batch_size > 0 ? reqs[0]->j->srv->batch_size : total_docs;
    if (max_batch > total_docs)
        max_batch = total_docs;
    int max_batch_tokens = reqs[0]->j->srv->max_batch_tokens;
    ffwd_context_input_t *inputs = xmalloc((size_t)max_batch * sizeof(*inputs));
    float *batch_embs = xmalloc(sizeof(*batch_embs));
    size_t batch_emb_cap = 0;

    int failed = 0;
    uint64_t infer_ns = 0;
    for (int start = 0; start < total_docs;) {
        int cur = 0;
        int tokens = 0;
        int chunks = 0;
        while (start + cur < total_docs && cur < max_batch) {
            const ffwd_context_input_t *input = &items[start + cur].input;
            if (!inference_batch_accepts_input(m, cur, tokens, input->input.n_tokens,
                                               max_batch_tokens))
                break;
            inputs[cur] = *input;
            tokens += input->input.n_tokens;
            chunks += input->n_spans;
            cur++;
        }
        size_t needed = (size_t)chunks * dim;
        if (needed > batch_emb_cap) {
            batch_embs = xrealloc(batch_embs, needed * sizeof(*batch_embs));
            batch_emb_cap = needed;
        }
        uint64_t t0 = nstime();
        int ffwd_rc = model_ffwd_spans_batch(m, inputs, cur, batch_embs);
        infer_ns += nstime() - t0;
        if (ffwd_rc != 0) {
            failed = 1;
            break;
        }
        int batch_span = 0;
        for (int i = 0; i < cur; i++) {
            int n_spans = inputs[i].n_spans;
            memcpy(embs + (size_t)items[start + i].output_span_index * dim,
                   batch_embs + (size_t)batch_span * dim, (size_t)n_spans * dim * sizeof(*embs));
            batch_span += n_spans;
        }
        start += cur;
    }

    span_offset = 0;
    for (int i = 0; i < n_reqs; i++) {
        reqs[i]->j->infer_ns += infer_ns;
        if (failed) {
            job_set_error(reqs[i]->j, 500, "contextual embedding failed", "server_error");
        } else {
            uint64_t t0 = nstime();
            render_contextual_response(reqs[i], embs + (size_t)span_offset * dim);
            reqs[i]->j->encode_ns += nstime() - t0;
        }
        span_offset += reqs[i]->total_chunks;
    }

    free(batch_embs);
    free(inputs);
    free(embs);
    free(items);
}

static int validate_common(cJSON *root,
                           cJSON *detail,
                           model_kind expected_kind,
                           loaded_model **out_model,
                           int *out_dims,
                           const char **out_encoding,
                           http_server *s) {
    cJSON *model_item = cJSON_GetObjectItemCaseSensitive(root, "model");
    const char *model_id = NULL;
    loaded_model *m = NULL;
    if (!model_item) {
        ve_add(detail, "[\"body\",\"model\"]", "field required", "missing");
    } else if (!cJSON_IsString(model_item) || !model_item->valuestring) {
        ve_add(detail, "[\"body\",\"model\"]", "model must be a string", "type_error.string");
    } else {
        model_id = model_item->valuestring;
        m = loaded_model_for_label(s, model_id);
        if (!m) {
            ve_add(detail, "[\"body\",\"model\"]",
                   "value is not a valid enum member for this endpoint", "enum");
        } else if (m->info->kind != expected_kind) {
            ve_add(detail, "[\"body\",\"model\"]", "model is not available on this endpoint", "enum");
            m = NULL;
        }
    }
    *out_model = m;
    const char *enc = encoding_from_root(root, detail, m ? m->info->api : FFWD_API_OPENAI);
    int min_dim = m ? m->info->min_dim : 128;
    int max_dim = m ? m->info->dim : 2560;
    int dims = dimensions_from_root(root, detail, min_dim, max_dim, enc);
    *out_encoding = enc;
    *out_dims = dims;
    return 0;
}

void prepare_embedding_request(job *j, cJSON *root, http_server *s, embedding_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root;

    cJSON *detail = cJSON_CreateArray();
    loaded_model *m = NULL;
    int dims = 0;
    const char *encoding = NULL;
    validate_common(root, detail, MODEL_KIND_STANDARD, &m, &dims, &encoding, s);
    out->model = m;
    out->dims = dims;
    out->encoding = encoding;

    /* The query instruction, when requested, applies to every input. */
    const char *query_instruct = m && text_type_is_query(root, detail, m->info->query_instruct)
                                     ? m->info->query_instruct
                                     : NULL;

    cJSON *input = cJSON_GetObjectItemCaseSensitive(root, "input");
    int n_inputs = 0;
    if (!input) {
        ve_add(detail, "[\"body\",\"input\"]", "field required", "missing");
    } else if (cJSON_IsString(input)) {
        if (!input->valuestring || input->valuestring[0] == '\0')
            ve_add(detail, "[\"body\",\"input\"]", "input must not be empty", "value_error.empty");
        n_inputs = 1;
    } else if (cJSON_IsArray(input)) {
        n_inputs = cJSON_GetArraySize(input);
        if (n_inputs < 1)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at least 1 item",
                   "value_error.list.min_items");
        else if (n_inputs > FFWD_API_MAX_STANDARD_INPUTS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 512 items",
                   "value_error.list.max_items");
        cJSON *item;
        int i = 0;
        cJSON_ArrayForEach(item, input) {
            char loc[64];
            snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", i);
            if (!cJSON_IsString(item)) {
                ve_add(detail, loc, "input item must be a string", "type_error.string");
            } else if (!item->valuestring || item->valuestring[0] == '\0') {
                ve_add(detail, loc, "input item must not be empty", "value_error.empty");
            }
            i++;
        }
    } else {
        ve_add(detail, "[\"body\",\"input\"]", "input must be a string or an array of strings",
               "type_error");
    }

    if (cJSON_GetArraySize(detail) == 0 && m && input) {
        out->n_inputs = n_inputs;
        out->tokens = xcalloc((size_t)n_inputs, sizeof(*out->tokens));
        out->inputs = xmalloc((size_t)n_inputs * sizeof(*out->inputs));

        int idx = 0;
        if (input && cJSON_IsString(input) && input->valuestring) {
            if (tokenize_input(m, j, input->valuestring, query_instruct, &out->tokens[0]) != 0) {
                ve_add(detail, "[\"body\",\"input\"]", "tokenization failed",
                       "value_error.tokenization");
            } else {
                out->inputs[0].ids = out->tokens[0].ids;
                out->inputs[0].n_tokens = out->tokens[0].n_tokens;
                out->total_tokens = out->tokens[0].n_tokens;
                if (out->tokens[0].n_tokens > FFWD_API_MAX_ITEM_TOKENS)
                    ve_add(detail, "[\"body\",\"input\"]", "input exceeds 32768 token limit",
                           "value_error.context_length");
            }
        } else {
            cJSON *item;
            cJSON_ArrayForEach(item, input) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", idx);
                if (!cJSON_IsString(item) || !item->valuestring) {
                    ve_add(detail, loc, "input item must be a string", "type_error.string");
                    idx++;
                    continue;
                }
                if (tokenize_input(m, j, item->valuestring, query_instruct, &out->tokens[idx]) != 0) {
                    ve_add(detail, loc, "tokenization failed", "value_error.tokenization");
                } else {
                    out->inputs[idx].ids = out->tokens[idx].ids;
                    out->inputs[idx].n_tokens = out->tokens[idx].n_tokens;
                    out->total_tokens += out->tokens[idx].n_tokens;
                    if (out->tokens[idx].n_tokens > FFWD_API_MAX_ITEM_TOKENS)
                        ve_add(detail, loc, "input exceeds 32768 token limit",
                               "value_error.context_length");
                }
                idx++;
            }
        }

        if (out->total_tokens > FFWD_API_MAX_TOTAL_TOKENS)
            ve_add(detail, "[\"body\",\"input\"]", "request exceeds 120000 token limit",
                   "value_error.context_length");
    }

    if (cJSON_GetArraySize(detail) > 0) {
        job_set_422(j, detail);
        embedding_request_free(out);
        out->j = j;
    } else {
        out->ready = 1;
    }
    cJSON_Delete(detail);
}

void prepare_contextual_request(job *j, cJSON *root, http_server *s, contextual_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root;

    cJSON *detail = cJSON_CreateArray();
    loaded_model *m = NULL;
    int dims = 0;
    const char *encoding = NULL;
    validate_common(root, detail, MODEL_KIND_CONTEXTUAL, &m, &dims, &encoding, s);
    out->model = m;
    out->dims = dims;
    out->encoding = encoding;

    cJSON *input = cJSON_GetObjectItemCaseSensitive(root, "input");
    int n_docs = 0;
    int total_chunks = 0;
    if (!input) {
        ve_add(detail, "[\"body\",\"input\"]", "field required", "missing");
    } else if (!cJSON_IsArray(input)) {
        ve_add(detail, "[\"body\",\"input\"]", "input must be an array of document chunk arrays",
               "type_error.array");
    } else {
        n_docs = cJSON_GetArraySize(input);
        if (n_docs < 1)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at least 1 document",
                   "value_error.list.min_items");
        else if (n_docs > FFWD_API_MAX_CONTEXT_DOCS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 512 documents",
                   "value_error.list.max_items");
        cJSON *doc_arr;
        int di = 0;
        cJSON_ArrayForEach(doc_arr, input) {
            if (!cJSON_IsArray(doc_arr)) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", di);
                ve_add(detail, loc, "document must be an array of strings", "type_error.array");
                di++;
                continue;
            }
            int doc_chunks = cJSON_GetArraySize(doc_arr);
            if (doc_chunks > FFWD_API_MAX_CONTEXT_CHUNKS - total_chunks)
                total_chunks = FFWD_API_MAX_CONTEXT_CHUNKS + 1;
            else
                total_chunks += doc_chunks;
            if (doc_chunks < 1) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", di);
                ve_add(detail, loc, "document must contain at least 1 chunk",
                       "value_error.list.min_items");
            }
            cJSON *chunk;
            int ci = 0;
            cJSON_ArrayForEach(chunk, doc_arr) {
                char loc[80];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d,%d]", di, ci);
                if (!cJSON_IsString(chunk)) {
                    ve_add(detail, loc, "chunk must be a string", "type_error.string");
                } else if (!chunk->valuestring || chunk->valuestring[0] == '\0') {
                    ve_add(detail, loc, "chunk must not be empty", "value_error.empty");
                }
                ci++;
            }
            di++;
        }
        if (total_chunks > FFWD_API_MAX_CONTEXT_CHUNKS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 16000 chunks",
                   "value_error.list.max_items");
    }

    if (cJSON_GetArraySize(detail) == 0 && m) {
        out->n_docs = n_docs;
        out->docs = xcalloc((size_t)n_docs, sizeof(*out->docs));
        int64_t request_tokens = 0;
        cJSON *doc_arr;
        int di = 0;
        cJSON_ArrayForEach(doc_arr, input) {
            contextual_doc *doc = &out->docs[di];
            int n_chunks = cJSON_GetArraySize(doc_arr);
            token_buf *chunk_tokens = xcalloc((size_t)n_chunks, sizeof(*chunk_tokens));
            doc->spans = xcalloc((size_t)n_chunks, sizeof(*doc->spans));
            doc->n_spans = n_chunks;
            int64_t doc_tokens = n_chunks > 1 ? n_chunks - 1 : 0;
            int doc_valid = 1;

            cJSON *chunk;
            int ci = 0;
            cJSON_ArrayForEach(chunk, doc_arr) {
                char loc[80];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d,%d]", di, ci);
                if (tokenize_one(m, j, chunk->valuestring, &chunk_tokens[ci]) != 0) {
                    ve_add(detail, loc, "tokenization failed", "value_error.tokenization");
                    doc_valid = 0;
                } else {
                    doc_tokens += chunk_tokens[ci].n_tokens;
                }
                ci++;
            }

            if (doc_tokens > FFWD_API_MAX_ITEM_TOKENS) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", di);
                ve_add(detail, loc, "document exceeds 32768 token limit",
                       "value_error.context_length");
                doc_valid = 0;
            }
            request_tokens += doc_tokens;

            if (doc_valid) {
                doc->n_tokens = (int)doc_tokens;
                doc->ids = xmalloc((size_t)doc->n_tokens * sizeof(*doc->ids));
                int sep_id = ffwd_tok_context_separator_id(m->tok);
                int pos = 0;
                for (ci = 0; ci < n_chunks; ci++) {
                    doc->spans[ci].start = pos;
                    doc->spans[ci].n_tokens = chunk_tokens[ci].n_tokens;
                    /* n_tokens > 0 implies ids != NULL (tokenize_one sets both
                     * together), and the guard avoids a memcpy(dst, NULL, 0) for
                     * an empty chunk that the static analyzer reports as UB. */
                    if (chunk_tokens[ci].n_tokens > 0)
                        memcpy(doc->ids + pos, chunk_tokens[ci].ids,
                               (size_t)chunk_tokens[ci].n_tokens * sizeof(*doc->ids));
                    pos += chunk_tokens[ci].n_tokens;
                    if (ci + 1 < n_chunks) {
                        doc->ids[pos++] = sep_id;
                    }
                }
            }
            free_token_bufs(chunk_tokens, n_chunks);
            free(chunk_tokens);
            di++;
        }
        if (request_tokens > FFWD_API_MAX_TOTAL_TOKENS) {
            ve_add(detail, "[\"body\",\"input\"]", "request exceeds 120000 token limit",
                   "value_error.context_length");
        } else {
            out->total_tokens = (int)request_tokens;
            out->total_chunks = total_chunks;
        }
    }

    if (cJSON_GetArraySize(detail) > 0) {
        job_set_422(j, detail);
        contextual_request_free(out);
        out->j = j;
    } else {
        out->ready = 1;
    }
    cJSON_Delete(detail);
}

static int
validate_rerank_model(cJSON *root, cJSON *detail, http_server *s, loaded_model **out_model) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, "model");
    if (!item) {
        ve_add(detail, "[\"body\",\"model\"]", "field required", "missing");
        return 0;
    }
    if (!cJSON_IsString(item) || !item->valuestring) {
        ve_add(detail, "[\"body\",\"model\"]", "model must be a string", "type_error.string");
        return 0;
    }
    loaded_model *m = loaded_model_for_label(s, item->valuestring);
    if (!m) {
        ve_add(detail, "[\"body\",\"model\"]", "value is not a valid enum member for this endpoint",
               "enum");
        return 0;
    }
    if (m->info->kind != MODEL_KIND_LATE) {
        ve_add(detail, "[\"body\",\"model\"]", "model is not available on this endpoint", "enum");
        return 0;
    }
    *out_model = m;
    return 0;
}

void prepare_rerank_request(job *j, cJSON *root, http_server *s, rerank_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root;

    cJSON *detail = cJSON_CreateArray();
    validate_rerank_model(root, detail, s, &out->model);
    cJSON *query = cJSON_GetObjectItemCaseSensitive(root, "query");
    cJSON *documents = cJSON_GetObjectItemCaseSensitive(root, "documents");
    /* query_str is set only once query passes every check; the tokenize step
     * below runs only when detail is empty, so it is non-NULL there. Hoisting it
     * also keeps the validated value explicit for the static analyzer. */
    const char *query_str = NULL;
    if (!query) {
        ve_add(detail, "[\"body\",\"query\"]", "field required", "missing");
    } else if (!cJSON_IsString(query)) {
        ve_add(detail, "[\"body\",\"query\"]", "query must be a string", "type_error.string");
    } else if (!query->valuestring || !query->valuestring[0]) {
        ve_add(detail, "[\"body\",\"query\"]", "query must not be empty", "value_error.empty");
    } else {
        query_str = query->valuestring;
    }

    int n_documents = 0;
    if (!documents) {
        ve_add(detail, "[\"body\",\"documents\"]", "field required", "missing");
    } else if (!cJSON_IsArray(documents)) {
        ve_add(detail, "[\"body\",\"documents\"]", "documents must be an array of strings",
               "type_error.array");
    } else {
        n_documents = cJSON_GetArraySize(documents);
        if (n_documents < 1)
            ve_add(detail, "[\"body\",\"documents\"]", "documents must contain at least 1 item",
                   "value_error.list.min_items");
        else if (n_documents > FFWD_API_MAX_RERANK_DOCUMENTS)
            ve_add(detail, "[\"body\",\"documents\"]", "documents must contain at most 1000 items",
                   "value_error.list.max_items");
        cJSON *document;
        int i = 0;
        cJSON_ArrayForEach(document, documents) {
            char loc[72];
            snprintf(loc, sizeof(loc), "[\"body\",\"documents\",%d]", i++);
            if (!cJSON_IsString(document))
                ve_add(detail, loc, "document must be a string", "type_error.string");
            else if (!document->valuestring || !document->valuestring[0])
                ve_add(detail, loc, "document must not be empty", "value_error.empty");
        }
    }

    cJSON *top_n = cJSON_GetObjectItemCaseSensitive(root, "top_n");
    cJSON *top_k = cJSON_GetObjectItemCaseSensitive(root, "top_k");
    if (top_n && top_k) {
        ve_add(detail, "[\"body\",\"top_n\"]", "top_n and top_k are aliases; provide only one",
               "value_error.conflict");
    }
    cJSON *top = top_n ? top_n : top_k;
    out->top_n = n_documents;
    if (top) {
        if (!cjson_is_integer(top)) {
            ve_add(detail, top_n ? "[\"body\",\"top_n\"]" : "[\"body\",\"top_k\"]",
                   "top_n must be an integer", "type_error.integer");
        } else {
            out->top_n = top->valueint;
            if (out->top_n < 1 || out->top_n > n_documents)
                ve_add(detail, top_n ? "[\"body\",\"top_n\"]" : "[\"body\",\"top_k\"]",
                       "top_n must be between 1 and the number of documents", "value_error.range");
        }
    }

    cJSON *return_documents = cJSON_GetObjectItemCaseSensitive(root, "return_documents");
    if (return_documents) {
        if (!cJSON_IsBool(return_documents))
            ve_add(detail, "[\"body\",\"return_documents\"]", "return_documents must be a boolean",
                   "type_error.bool");
        else
            out->return_documents = cJSON_IsTrue(return_documents);
    }

    if (cJSON_GetArraySize(detail) == 0 && out->model) {
        out->n_documents = n_documents;
        out->documents = xcalloc((size_t)n_documents, sizeof(*out->documents));
        if (tokenize_late_text(out->model, j, query_str, 1, &out->query) != 0) {
            ve_add(detail, "[\"body\",\"query\"]", "tokenization failed", "value_error.tokenization");
        } else {
            out->query_tokens = out->query.n_tokens;
            if (out->query_tokens > FFWD_API_MAX_ITEM_TOKENS)
                ve_add(detail, "[\"body\",\"query\"]", "query exceeds 32768 token limit",
                       "value_error.context_length");
        }

        cJSON *document;
        int i = 0;
        cJSON_ArrayForEach(document, documents) {
            char loc[72];
            snprintf(loc, sizeof(loc), "[\"body\",\"documents\",%d]", i);
            if (tokenize_late_text(out->model, j, document->valuestring, 0, &out->documents[i]) !=
                0) {
                ve_add(detail, loc, "tokenization failed", "value_error.tokenization");
            } else {
                int n = out->documents[i].n_tokens;
                if (n > FFWD_API_MAX_ITEM_TOKENS)
                    ve_add(detail, loc, "document exceeds 32768 token limit",
                           "value_error.context_length");
                if (out->document_tokens <= INT_MAX - n)
                    out->document_tokens += n;
                else
                    out->document_tokens = INT_MAX;
            }
            i++;
        }
        if (out->query_tokens > FFWD_API_MAX_TOTAL_TOKENS - out->document_tokens) {
            ve_add(detail, "[\"body\",\"documents\"]", "request exceeds 120000 token limit",
                   "value_error.context_length");
        }
    }

    if (cJSON_GetArraySize(detail) > 0) {
        job_set_422(j, detail);
        rerank_request_free(out);
        out->j = j;
    } else {
        out->ready = 1;
    }
    cJSON_Delete(detail);
}

typedef struct {
    int index;
    float score;
} rerank_result;

static int rerank_result_cmp(const void *a, const void *b) {
    const rerank_result *ra = a;
    const rerank_result *rb = b;
    if (ra->score > rb->score)
        return -1;
    if (ra->score < rb->score)
        return 1;
    return ra->index - rb->index;
}

void execute_rerank_request(rerank_request *r) {
    if (!r || !r->model) {
        if (r && r->j)
            job_set_error(r->j, 500, "late-interaction model is unavailable", "server_error");
        return;
    }
    float *scores = xmalloc((size_t)r->n_documents * sizeof(*scores));

    /* Flatten the per-document fields into the arrays the backend rerank takes;
     * it encodes the query and every candidate in one packed forward and scores
     * MaxSim, so N short documents collapse to a single batch. */
    const int **doc_ids = xmalloc((size_t)r->n_documents * sizeof(*doc_ids));
    const int **keep = xmalloc((size_t)r->n_documents * sizeof(*keep));
    int *n_tokens = xmalloc((size_t)r->n_documents * sizeof(*n_tokens));
    int *n_keep = xmalloc((size_t)r->n_documents * sizeof(*n_keep));
    for (int i = 0; i < r->n_documents; i++) {
        doc_ids[i] = r->documents[i].ids;
        keep[i] = r->documents[i].keep;
        n_tokens[i] = r->documents[i].n_tokens;
        n_keep[i] = r->documents[i].n_keep;
    }
    ffwd_rerank_input_t in = {
        .query_ids = r->query.ids,
        .query_n_tokens = r->query.n_tokens,
        .query_n_keep = r->query.n_keep,
        .doc_ids = doc_ids,
        .doc_n_tokens = n_tokens,
        .doc_keep = keep,
        .doc_n_keep = n_keep,
        .n_docs = r->n_documents,
    };
    uint64_t t0 = nstime();
    int rc = ffwd_rerank(r->model->backend, &in, scores);
    r->j->infer_ns += nstime() - t0;
    free(n_keep);
    free(n_tokens);
    free(keep);
    free(doc_ids);
    if (rc != 0) {
        free(scores);
        job_set_error(r->j, 500, "late-interaction reranking failed", "server_error");
        return;
    }

    rerank_result *ranked = xmalloc((size_t)r->n_documents * sizeof(*ranked));
    for (int i = 0; i < r->n_documents; i++) {
        ranked[i].index = i;
        ranked[i].score = scores[i];
    }
    qsort(ranked, (size_t)r->n_documents, sizeof(*ranked), rerank_result_cmp);

    t0 = nstime();
    sbuf b = {0};
    sbuf_printf(&b, "{\"object\":\"list\",\"model\":\"%s\",\"results\":[", r->model->info->id);
    cJSON *documents = cJSON_GetObjectItemCaseSensitive(r->root, "documents");
    for (int i = 0; i < r->top_n; i++) {
        if (i)
            sbuf_putc(&b, ',');
        sbuf_printf(&b, "{\"index\":%d,\"relevance_score\":%.9g", ranked[i].index,
                    (double)ranked[i].score);
        if (r->return_documents) {
            cJSON *document = cJSON_GetArrayItem(documents, ranked[i].index);
            sbuf_puts(&b, ",\"document\":");
            append_json_string(&b, document->valuestring);
        }
        sbuf_putc(&b, '}');
    }
    sbuf_printf(&b,
                "],\"usage\":{\"query_tokens\":%d,\"document_tokens\":%d,"
                "\"total_tokens\":%d}}",
                r->query_tokens, r->document_tokens, r->query_tokens + r->document_tokens);
    set_response_from_buf(r->j, &b);
    r->j->encode_ns += nstime() - t0;
    free(ranked);
    free(scores);
}
