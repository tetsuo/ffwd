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

/* Hand JSON serialization to the renderer thread only when another batch is
 * already queued, so the worker can start that GPU launch while these results
 * serialize. With nothing else queued the worker renders inline, which avoids a
 * thread hand-off that buys no overlap and adds latency to the reply. */
static int render_should_defer(http_server *s) { return worker_has_pending_jobs(s); }

static int embedding_batch_item_cmp(const void *a, const void *b) {
    const embedding_batch_item *ia = a;
    const embedding_batch_item *ib = b;
    if (ia->input.n_tokens < ib->input.n_tokens)
        return -1;
    if (ia->input.n_tokens > ib->input.n_tokens)
        return 1;
    return ia->output_index - ib->output_index;
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

    int defer_render = render_should_defer(reqs[0]->j->srv);
    pos = 0;
    for (int i = 0; i < n_reqs; i++) {
        reqs[i]->j->infer_ns += infer_ns;
        if (failed) {
            job_set_error(reqs[i]->j, 500, "embedding failed", "server_error");
        } else if (!defer_render) {
            uint64_t t0 = nstime();
            if (render_embedding_response(reqs[i], embs + (size_t)pos * dim) != 0)
                job_set_error(reqs[i]->j, 500, "embedding normalization failed", "server_error");
            reqs[i]->j->encode_ns += nstime() - t0;
        } else {
            job_set_embedding_render(reqs[i]->j, reqs[i], embs + (size_t)pos * dim);
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

static int validate_common(yyjson_val *root,
                           yyjson_mut_doc *detail,
                           model_kind expected_kind,
                           loaded_model **out_model,
                           int *out_dims,
                           const char **out_encoding,
                           http_server *s) {
    yyjson_val *model_item = yyjson_obj_get(root, "model");
    const char *model_id = NULL;
    loaded_model *m = NULL;
    if (!model_item) {
        ve_add(detail, "[\"body\",\"model\"]", "field required", "missing");
    } else if (!(model_id = yyjson_get_str(model_item))) {
        ve_add(detail, "[\"body\",\"model\"]", "model must be a string", "type_error.string");
    } else {
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

void prepare_embedding_request(job *j, yyjson_doc *root_doc, http_server *s, embedding_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root_doc;
    yyjson_val *root = yyjson_doc_get_root(root_doc);

    yyjson_mut_doc *detail = ve_new();
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

    yyjson_val *input = yyjson_obj_get(root, "input");
    int n_inputs = 0;
    if (!input) {
        ve_add(detail, "[\"body\",\"input\"]", "field required", "missing");
    } else if (yyjson_is_str(input)) {
        const char *text = yyjson_get_str(input);
        if (!text || text[0] == '\0')
            ve_add(detail, "[\"body\",\"input\"]", "input must not be empty", "value_error.empty");
        n_inputs = 1;
    } else if (yyjson_is_arr(input)) {
        n_inputs = (int)yyjson_arr_size(input);
        if (n_inputs < 1)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at least 1 item",
                   "value_error.list.min_items");
        else if (n_inputs > FFWD_API_MAX_STANDARD_INPUTS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 512 items",
                   "value_error.list.max_items");
        size_t idx, max;
        yyjson_val *item;
        yyjson_arr_foreach(input, idx, max, item) {
            char loc[64];
            snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", (int)idx);
            if (!yyjson_is_str(item)) {
                ve_add(detail, loc, "input item must be a string", "type_error.string");
            } else {
                const char *text = yyjson_get_str(item);
                if (!text || text[0] == '\0')
                    ve_add(detail, loc, "input item must not be empty", "value_error.empty");
            }
        }
    } else {
        ve_add(detail, "[\"body\",\"input\"]", "input must be a string or an array of strings",
               "type_error");
    }

    if (ve_count(detail) == 0 && m && input) {
        out->n_inputs = n_inputs;
        out->tokens = xcalloc((size_t)n_inputs, sizeof(*out->tokens));
        out->inputs = xmalloc((size_t)n_inputs * sizeof(*out->inputs));

        const char *single = yyjson_get_str(input);
        if (single) {
            if (tokenize_input(m, j, single, query_instruct, &out->tokens[0]) != 0) {
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
            size_t idx, max;
            yyjson_val *item;
            yyjson_arr_foreach(input, idx, max, item) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", (int)idx);
                const char *text = yyjson_get_str(item);
                if (!text) {
                    ve_add(detail, loc, "input item must be a string", "type_error.string");
                    continue;
                }
                if (tokenize_input(m, j, text, query_instruct, &out->tokens[idx]) != 0) {
                    ve_add(detail, loc, "tokenization failed", "value_error.tokenization");
                } else {
                    out->inputs[idx].ids = out->tokens[idx].ids;
                    out->inputs[idx].n_tokens = out->tokens[idx].n_tokens;
                    out->total_tokens += out->tokens[idx].n_tokens;
                    if (out->tokens[idx].n_tokens > FFWD_API_MAX_ITEM_TOKENS)
                        ve_add(detail, loc, "input exceeds 32768 token limit",
                               "value_error.context_length");
                }
            }
        }

        if (out->total_tokens > FFWD_API_MAX_TOTAL_TOKENS)
            ve_add(detail, "[\"body\",\"input\"]", "request exceeds 120000 token limit",
                   "value_error.context_length");
    }

    if (ve_count(detail) > 0) {
        job_set_422(j, detail);
        embedding_request_free(out);
        out->j = j;
    } else {
        out->ready = 1;
    }
    yyjson_mut_doc_free(detail);
}

void prepare_contextual_request(job *j,
                                yyjson_doc *root_doc,
                                http_server *s,
                                contextual_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root_doc;
    yyjson_val *root = yyjson_doc_get_root(root_doc);

    yyjson_mut_doc *detail = ve_new();
    loaded_model *m = NULL;
    int dims = 0;
    const char *encoding = NULL;
    validate_common(root, detail, MODEL_KIND_CONTEXTUAL, &m, &dims, &encoding, s);
    out->model = m;
    out->dims = dims;
    out->encoding = encoding;

    yyjson_val *input = yyjson_obj_get(root, "input");
    int n_docs = 0;
    int total_chunks = 0;
    if (!input) {
        ve_add(detail, "[\"body\",\"input\"]", "field required", "missing");
    } else if (!yyjson_is_arr(input)) {
        ve_add(detail, "[\"body\",\"input\"]", "input must be an array of document chunk arrays",
               "type_error.array");
    } else {
        n_docs = (int)yyjson_arr_size(input);
        if (n_docs < 1)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at least 1 document",
                   "value_error.list.min_items");
        else if (n_docs > FFWD_API_MAX_CONTEXT_DOCS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 512 documents",
                   "value_error.list.max_items");
        size_t di, dn;
        yyjson_val *doc_arr;
        yyjson_arr_foreach(input, di, dn, doc_arr) {
            if (!yyjson_is_arr(doc_arr)) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", (int)di);
                ve_add(detail, loc, "document must be an array of strings", "type_error.array");
                continue;
            }
            int doc_chunks = (int)yyjson_arr_size(doc_arr);
            if (doc_chunks > FFWD_API_MAX_CONTEXT_CHUNKS - total_chunks)
                total_chunks = FFWD_API_MAX_CONTEXT_CHUNKS + 1;
            else
                total_chunks += doc_chunks;
            if (doc_chunks < 1) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", (int)di);
                ve_add(detail, loc, "document must contain at least 1 chunk",
                       "value_error.list.min_items");
            }
            size_t ci, cn;
            yyjson_val *chunk;
            yyjson_arr_foreach(doc_arr, ci, cn, chunk) {
                char loc[80];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d,%d]", (int)di, (int)ci);
                if (!yyjson_is_str(chunk)) {
                    ve_add(detail, loc, "chunk must be a string", "type_error.string");
                } else {
                    const char *text = yyjson_get_str(chunk);
                    if (!text || text[0] == '\0')
                        ve_add(detail, loc, "chunk must not be empty", "value_error.empty");
                }
            }
        }
        if (total_chunks > FFWD_API_MAX_CONTEXT_CHUNKS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 16000 chunks",
                   "value_error.list.max_items");
    }

    if (ve_count(detail) == 0 && m) {
        out->n_docs = n_docs;
        out->docs = xcalloc((size_t)n_docs, sizeof(*out->docs));
        int64_t request_tokens = 0;
        size_t di, dn;
        yyjson_val *doc_arr;
        yyjson_arr_foreach(input, di, dn, doc_arr) {
            contextual_doc *doc = &out->docs[di];
            int n_chunks = (int)yyjson_arr_size(doc_arr);
            token_buf *chunk_tokens = xcalloc((size_t)n_chunks, sizeof(*chunk_tokens));
            doc->spans = xcalloc((size_t)n_chunks, sizeof(*doc->spans));
            doc->n_spans = n_chunks;
            int64_t doc_tokens = n_chunks > 1 ? n_chunks - 1 : 0;
            int doc_valid = 1;

            size_t ci, cn;
            yyjson_val *chunk;
            yyjson_arr_foreach(doc_arr, ci, cn, chunk) {
                char loc[80];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d,%d]", (int)di, (int)ci);
                if (tokenize_one(m, j, yyjson_get_str(chunk), &chunk_tokens[ci]) != 0) {
                    ve_add(detail, loc, "tokenization failed", "value_error.tokenization");
                    doc_valid = 0;
                } else {
                    doc_tokens += chunk_tokens[ci].n_tokens;
                }
            }

            if (doc_tokens > FFWD_API_MAX_ITEM_TOKENS) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", (int)di);
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
                for (int k = 0; k < n_chunks; k++) {
                    doc->spans[k].start = pos;
                    doc->spans[k].n_tokens = chunk_tokens[k].n_tokens;
                    /* n_tokens > 0 implies ids != NULL (tokenize_one sets both
                     * together), and the guard avoids a memcpy(dst, NULL, 0) for
                     * an empty chunk that the static analyzer reports as UB. */
                    if (chunk_tokens[k].n_tokens > 0)
                        memcpy(doc->ids + pos, chunk_tokens[k].ids,
                               (size_t)chunk_tokens[k].n_tokens * sizeof(*doc->ids));
                    pos += chunk_tokens[k].n_tokens;
                    if (k + 1 < n_chunks) {
                        doc->ids[pos++] = sep_id;
                    }
                }
            }
            free_token_bufs(chunk_tokens, n_chunks);
            free(chunk_tokens);
        }
        if (request_tokens > FFWD_API_MAX_TOTAL_TOKENS) {
            ve_add(detail, "[\"body\",\"input\"]", "request exceeds 120000 token limit",
                   "value_error.context_length");
        } else {
            out->total_tokens = (int)request_tokens;
            out->total_chunks = total_chunks;
        }
    }

    if (ve_count(detail) > 0) {
        job_set_422(j, detail);
        contextual_request_free(out);
        out->j = j;
    } else {
        out->ready = 1;
    }
    yyjson_mut_doc_free(detail);
}

static int validate_rerank_model(yyjson_val *root,
                                 yyjson_mut_doc *detail,
                                 http_server *s,
                                 loaded_model **out_model) {
    yyjson_val *item = yyjson_obj_get(root, "model");
    if (!item) {
        ve_add(detail, "[\"body\",\"model\"]", "field required", "missing");
        return 0;
    }
    const char *model_id = yyjson_get_str(item);
    if (!model_id) {
        ve_add(detail, "[\"body\",\"model\"]", "model must be a string", "type_error.string");
        return 0;
    }
    loaded_model *m = loaded_model_for_label(s, model_id);
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

void prepare_rerank_request(job *j, yyjson_doc *root_doc, http_server *s, rerank_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root_doc;
    yyjson_val *root = yyjson_doc_get_root(root_doc);

    yyjson_mut_doc *detail = ve_new();
    validate_rerank_model(root, detail, s, &out->model);
    yyjson_val *query = yyjson_obj_get(root, "query");
    yyjson_val *documents = yyjson_obj_get(root, "documents");
    /* query_str is set only once query passes every check; the tokenize step
     * below runs only when detail is empty, so it is non-NULL there. Hoisting it
     * also keeps the validated value explicit for the static analyzer. */
    const char *query_str = NULL;
    if (!query) {
        ve_add(detail, "[\"body\",\"query\"]", "field required", "missing");
    } else if (!yyjson_is_str(query)) {
        ve_add(detail, "[\"body\",\"query\"]", "query must be a string", "type_error.string");
    } else if (!yyjson_get_str(query)[0]) {
        ve_add(detail, "[\"body\",\"query\"]", "query must not be empty", "value_error.empty");
    } else {
        query_str = yyjson_get_str(query);
    }

    int n_documents = 0;
    if (!documents) {
        ve_add(detail, "[\"body\",\"documents\"]", "field required", "missing");
    } else if (!yyjson_is_arr(documents)) {
        ve_add(detail, "[\"body\",\"documents\"]", "documents must be an array of strings",
               "type_error.array");
    } else {
        n_documents = (int)yyjson_arr_size(documents);
        if (n_documents < 1)
            ve_add(detail, "[\"body\",\"documents\"]", "documents must contain at least 1 item",
                   "value_error.list.min_items");
        else if (n_documents > FFWD_API_MAX_RERANK_DOCUMENTS)
            ve_add(detail, "[\"body\",\"documents\"]", "documents must contain at most 1000 items",
                   "value_error.list.max_items");
        size_t i, dmax;
        yyjson_val *document;
        yyjson_arr_foreach(documents, i, dmax, document) {
            char loc[72];
            snprintf(loc, sizeof(loc), "[\"body\",\"documents\",%d]", (int)i);
            if (!yyjson_is_str(document)) {
                ve_add(detail, loc, "document must be a string", "type_error.string");
            } else {
                const char *text = yyjson_get_str(document);
                if (!text || !text[0])
                    ve_add(detail, loc, "document must not be empty", "value_error.empty");
            }
        }
    }

    yyjson_val *top_n = yyjson_obj_get(root, "top_n");
    yyjson_val *top_k = yyjson_obj_get(root, "top_k");
    if (top_n && top_k) {
        ve_add(detail, "[\"body\",\"top_n\"]", "top_n and top_k are aliases; provide only one",
               "value_error.conflict");
    }
    yyjson_val *top = top_n ? top_n : top_k;
    out->top_n = n_documents;
    if (top) {
        if (!json_is_integer(top)) {
            ve_add(detail, top_n ? "[\"body\",\"top_n\"]" : "[\"body\",\"top_k\"]",
                   "top_n must be an integer", "type_error.integer");
        } else {
            out->top_n = (int)yyjson_get_num(top);
            if (out->top_n < 1 || out->top_n > n_documents)
                ve_add(detail, top_n ? "[\"body\",\"top_n\"]" : "[\"body\",\"top_k\"]",
                       "top_n must be between 1 and the number of documents", "value_error.range");
        }
    }

    yyjson_val *return_documents = yyjson_obj_get(root, "return_documents");
    if (return_documents) {
        if (!yyjson_is_bool(return_documents))
            ve_add(detail, "[\"body\",\"return_documents\"]", "return_documents must be a boolean",
                   "type_error.bool");
        else
            out->return_documents = yyjson_get_bool(return_documents);
    }

    if (ve_count(detail) == 0 && out->model) {
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

        size_t i, dmax;
        yyjson_val *document;
        yyjson_arr_foreach(documents, i, dmax, document) {
            char loc[72];
            snprintf(loc, sizeof(loc), "[\"body\",\"documents\",%d]", (int)i);
            if (tokenize_late_text(out->model, j, yyjson_get_str(document), 0, &out->documents[i]) !=
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
        }
        if (out->query_tokens > FFWD_API_MAX_TOTAL_TOKENS - out->document_tokens) {
            ve_add(detail, "[\"body\",\"documents\"]", "request exceeds 120000 token limit",
                   "value_error.context_length");
        }
    }

    if (ve_count(detail) > 0) {
        job_set_422(j, detail);
        rerank_request_free(out);
        out->j = j;
    } else {
        out->ready = 1;
    }
    yyjson_mut_doc_free(detail);
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
    yyjson_val *documents = yyjson_obj_get(yyjson_doc_get_root(r->root), "documents");
    for (int i = 0; i < r->top_n; i++) {
        if (i)
            sbuf_putc(&b, ',');
        sbuf_printf(&b, "{\"index\":%d,\"relevance_score\":%.9g", ranked[i].index,
                    (double)ranked[i].score);
        if (r->return_documents) {
            yyjson_val *document = yyjson_arr_get(documents, (size_t)ranked[i].index);
            sbuf_puts(&b, ",\"document\":");
            append_json_string(&b, yyjson_get_str(document));
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
