#include "server_internal.h"
#include "util.h"
#include "sbuf.h"
#include "base64.h"
#include "ffwd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Embedding output encoding
 *
 * Turns a pooled float vector into the requested wire form. OpenAI-compatible
 * models emit true float32 arrays by default. Perplexity-compatible models
 * emit tanh-quantized int8 by default, and their "float" view is the decoded
 * int8 value for SDK compatibility.
 * ======================================================================== */

signed char quantize_int8_tanh(float x) {
    int v = (int)lrintf(tanhf(x) * 127.0f);
    if (v > 127)
        v = 127;
    if (v < -128)
        v = -128;
    return (signed char)v;
}

char *encode_embedding(const float *emb, int dims, const char *encoding) {
    if (!strcmp(encoding, "base64")) {
        /* OpenAI/DashScope (Qwen3): base64 of the raw little-endian float32
         * vector. Copy through a byte buffer instead of reinterpret-casting the
         * float pointer; x86_64 and aarch64 are little-endian, so the bytes
         * match the wire format with no swap. */
        size_t nbytes = (size_t)dims * sizeof(float);
        unsigned char *buf = xmalloc(nbytes);
        memcpy(buf, emb, nbytes);
        char *out = base64_encode(buf, nbytes);
        free(buf);
        return out;
    }
    if (!strcmp(encoding, "base64_binary")) {
        size_t bytes = (size_t)dims / 8;
        unsigned char *bits = xcalloc(bytes, 1);
        for (int i = 0; i < dims; i++) {
            if (emb[i] >= 0.0f)
                bits[(size_t)i >> 3] |= (unsigned char)(1u << (i & 7));
        }
        char *out = base64_encode(bits, bytes);
        free(bits);
        return out;
    }

    signed char *q = xmalloc((size_t)dims);
    for (int i = 0; i < dims; i++)
        q[i] = quantize_int8_tanh(emb[i]);
    char *out = base64_encode((const unsigned char *)q, (size_t)dims);
    free(q);
    return out;
}

static void append_embedding_object(sbuf *b, int index, const char *embedding) {
    sbuf_printf(b, "{\"object\":\"embedding\",\"index\":%d,\"embedding\":\"", index);
    sbuf_puts(b, embedding);
    sbuf_puts(b, "\"}");
}

void append_embedding_value(
    sbuf *b, int index, const float *emb, int dims, const char *encoding, embedding_api_t api) {
    if (!strcmp(encoding, "float")) {
        sbuf_printf(b, "{\"object\":\"embedding\",\"index\":%d,\"embedding\":[", index);
        for (int i = 0; i < dims; i++) {
            if (i)
                sbuf_putc(b, ',');
            float v =
                api == FFWD_API_PERPLEXITY ? (float)quantize_int8_tanh(emb[i]) / 128.0f : emb[i];
            sbuf_printf(b, "%.9g", (double)v);
        }
        sbuf_puts(b, "]}");
        return;
    }
    char *encoded = encode_embedding(emb, dims, encoding);
    append_embedding_object(b, index, encoded);
    free(encoded);
}

void set_response_from_buf(job *j, sbuf *b) {
    j->status = 200;
    j->content_type = "application/json";
    j->response = b->ptr;
    j->response_len = b->len;
    memset(b, 0, sizeof(*b));
}

int render_embedding_response(embedding_request *r, const float *embs) {
    const int full_dim = r->model->info->dim;
    const float *render_embs = embs;
    float *truncated = NULL;
    if (r->model->renormalize_truncated && r->dims < full_dim) {
        truncated = xmalloc((size_t)r->n_inputs * r->dims * sizeof(*truncated));
        for (int i = 0; i < r->n_inputs; i++) {
            float *dst = truncated + (size_t)i * r->dims;
            memcpy(dst, embs + (size_t)i * full_dim, (size_t)r->dims * sizeof(*dst));
            if (ffwd_l2_normalize(dst, r->dims) != 0) {
                free(truncated);
                return -1;
            }
        }
        render_embs = truncated;
    }
    int render_stride = truncated ? r->dims : full_dim;
    embedding_api_t api = r->model->info->api;

    sbuf b = {0};
    sbuf_puts(&b, "{\"object\":\"list\",\"data\":[");
    for (int i = 0; i < r->n_inputs; i++) {
        if (i)
            sbuf_putc(&b, ',');
        append_embedding_value(&b, i, render_embs + (size_t)i * render_stride, r->dims, r->encoding,
                               api);
    }
    sbuf_printf(&b,
                "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%d,"
                "\"total_tokens\":%d}}",
                r->model->info->id, r->total_tokens, r->total_tokens);
    set_response_from_buf(r->j, &b);

    free(truncated);
    return 0;
}

void job_render_free(job *j) {
    if (!j)
        return;
    free(j->embedding_render.encoding);
    free(j->embedding_render.embs);
    memset(&j->embedding_render, 0, sizeof(j->embedding_render));
    j->render_kind = 0;
}

void job_set_embedding_render(job *j, const embedding_request *r, const float *embs) {
    if (!j || !r || !r->model || !embs)
        return;
    job_render_free(j);
    size_t count = (size_t)r->n_inputs * (size_t)r->model->info->dim;
    j->embedding_render.model = r->model;
    j->embedding_render.encoding = xstrdup(r->encoding);
    j->embedding_render.embs = xmalloc(count * sizeof(*j->embedding_render.embs));
    memcpy(j->embedding_render.embs, embs, count * sizeof(*j->embedding_render.embs));
    j->embedding_render.dims = r->dims;
    j->embedding_render.n_inputs = r->n_inputs;
    j->embedding_render.total_tokens = r->total_tokens;
    j->render_kind = 1;
}

int render_job_response(job *j) {
    if (!j || j->render_kind == 0)
        return 0;
    if (j->render_kind != 1 || !j->embedding_render.model || !j->embedding_render.encoding ||
        !j->embedding_render.embs)
        return -1;

    embedding_request r;
    memset(&r, 0, sizeof(r));
    r.j = j;
    r.model = j->embedding_render.model;
    r.dims = j->embedding_render.dims;
    r.encoding = j->embedding_render.encoding;
    r.n_inputs = j->embedding_render.n_inputs;
    r.total_tokens = j->embedding_render.total_tokens;
    return render_embedding_response(&r, j->embedding_render.embs);
}

void render_contextual_response(contextual_request *r, const float *embs) {
    sbuf b = {0};
    sbuf_puts(&b, "{\"object\":\"list\",\"data\":[");
    int span_offset = 0;
    for (int di = 0; di < r->n_docs; di++) {
        if (di)
            sbuf_putc(&b, ',');
        sbuf_printf(&b, "{\"object\":\"list\",\"index\":%d,\"data\":[", di);
        for (int ci = 0; ci < r->docs[di].n_spans; ci++) {
            if (ci)
                sbuf_putc(&b, ',');
            const float *emb = embs + (size_t)span_offset * r->model->info->dim;
            append_embedding_value(&b, ci, emb, r->dims, r->encoding, r->model->info->api);
            span_offset++;
        }
        sbuf_puts(&b, "]}");
    }
    sbuf_printf(&b, "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}",
                r->model->info->id, r->total_tokens, r->total_tokens);
    set_response_from_buf(r->j, &b);
}
