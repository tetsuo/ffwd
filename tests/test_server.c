/* tests/test_server.c - tests for embed_server.c.
 * ds4-style: includes the whole server translation unit with the real main
 * compiled out, so statics are directly testable. Unit-tests the encoding
 * helpers, then boots the real server in-process (tiny 1024-dim hermetic
 * model + byte-vocab tokenizer fixture) and drives the full HTTP request
 * path over loopback sockets. Runs via `make test`. */

#define EMBED_SERVER_TEST
#include "../src/server.c"

#include "tiny_model.h"
#include "tok_fixture.h"

#include <arpa/inet.h>
#include <netinet/in.h>

static int test_failures = 0;

#define TEST_ASSERT(cond)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            test_failures++;                                                \
        }                                                                   \
    } while (0)

static void test_base64_rfc4648(void) {
    /* RFC 4648 test vectors. */
    static const struct {
        const char *in, *out;
    } cases[] = {
        {"", ""},
        {"f", "Zg=="},
        {"fo", "Zm8="},
        {"foo", "Zm9v"},
        {"foob", "Zm9vYg=="},
        {"fooba", "Zm9vYmE="},
        {"foobar", "Zm9vYmFy"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *got = base64_encode((const unsigned char *)cases[i].in, strlen(cases[i].in));
        TEST_ASSERT(strcmp(got, cases[i].out) == 0);
        free(got);
    }
}

static void test_quantize_int8_tanh(void) {
    TEST_ASSERT(quantize_int8_tanh(0.0f) == 0);
    /* tanh saturates to +-1, scaled by 127. */
    TEST_ASSERT(quantize_int8_tanh(100.0f) == 127);
    TEST_ASSERT(quantize_int8_tanh(-100.0f) == -127);
    /* Symmetry. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == -quantize_int8_tanh(-0.5f));
    /* round(tanh(0.5) * 127) = round(58.69) = 59. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == 59);
}

static void test_encode_embedding_int8(void) {
    float emb[4] = {0.0f, 0.5f, -0.5f, 100.0f};
    signed char expect[4] = {0, 59, -59, 127};
    char *got = encode_embedding(emb, 4, "base64_int8");
    char *want = base64_encode((const unsigned char *)expect, 4);
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);
}

static void test_encode_embedding_binary(void) {
    /* Sign bits pack LSB-first: bit i of byte i/8 is 1 when emb[i] >= 0. */
    float emb[8] = {1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
    unsigned char expect = 0x85; /* bits 0, 2, 7 set */
    char *got = encode_embedding(emb, 8, "base64_binary");
    char *want = base64_encode(&expect, 1);
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);

    /* Zero counts as non-negative (sign bit set). */
    float zeros[8] = {0};
    unsigned char all = 0xff;
    got = encode_embedding(zeros, 8, "base64_binary");
    want = base64_encode(&all, 1);
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);
}

static void test_encode_embedding_base64_float32(void) {
    /* OpenAI/DashScope "base64" is base64 of the raw little-endian float32
     * vector - lossless, unlike the int8 formats. */
    float emb[4] = {0.0f, 0.5f, -0.5f, 100.0f};
    /* Reference bytes via memcpy: a float->byte reinterpret cast confuses the
     * static analyzer's uninitialized-read model. */
    unsigned char bytes[sizeof emb];
    memcpy(bytes, emb, sizeof emb);
    char *want = base64_encode(bytes, sizeof emb);
    char *got = encode_embedding(emb, 4, "base64");
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);
}

static void test_append_embedding_value_float(void) {
    float emb[1] = {0.5f};

    /* OpenAI/DashScope (Qwen3): the true float32 value. */
    sbuf b = {0};
    append_embedding_value(&b, 0, emb, 1, "float", EMBED_API_OPENAI);
    TEST_ASSERT(strstr(b.ptr, "\"embedding\":[0.5]") != NULL);
    sbuf_free(&b);

    /* Perplexity: the int8-decoded view, round(tanh(0.5)*127)/128 = 59/128. */
    sbuf b2 = {0};
    append_embedding_value(&b2, 0, emb, 1, "float", EMBED_API_PERPLEXITY);
    TEST_ASSERT(strstr(b2.ptr, "0.4609375") != NULL);
    sbuf_free(&b2);
}

static void test_encoding_from_root_family(void) {
    cJSON *detail = cJSON_CreateArray();

    /* Default encoding differs by family when the field is absent. */
    cJSON *empty = cJSON_CreateObject();
    TEST_ASSERT(strcmp(encoding_from_root(empty, detail, EMBED_API_OPENAI), "float") == 0);
    TEST_ASSERT(strcmp(encoding_from_root(empty, detail, EMBED_API_PERPLEXITY), "base64_int8") ==
                0);
    cJSON_Delete(empty);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 0);

    /* OpenAI rejects int8 formats; Perplexity rejects the OpenAI base64. */
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "encoding_format", "base64_int8");
    encoding_from_root(o, detail, EMBED_API_OPENAI);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 1);
    cJSON_Delete(o);

    cJSON *p = cJSON_CreateObject();
    cJSON_AddStringToObject(p, "encoding_format", "base64");
    encoding_from_root(p, detail, EMBED_API_PERPLEXITY);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 2);
    cJSON_Delete(p);

    cJSON_Delete(detail);
}

static void test_text_type(void) {
    cJSON *detail = cJSON_CreateArray();

    /* Absent: no-op for both model types, no error. */
    cJSON *empty = cJSON_CreateObject();
    TEST_ASSERT(text_type_is_query(empty, detail, EMBED_QWEN3_QUERY_INSTRUCT) == 0);
    TEST_ASSERT(text_type_is_query(empty, detail, NULL) == 0);
    cJSON_Delete(empty);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 0);

    /* Qwen3: query -> 1, document -> 0, both accepted. */
    cJSON *q = cJSON_CreateObject();
    cJSON_AddStringToObject(q, "text_type", "query");
    TEST_ASSERT(text_type_is_query(q, detail, EMBED_QWEN3_QUERY_INSTRUCT) == 1);
    cJSON_Delete(q);
    cJSON *d = cJSON_CreateObject();
    cJSON_AddStringToObject(d, "text_type", "document");
    TEST_ASSERT(text_type_is_query(d, detail, EMBED_QWEN3_QUERY_INSTRUCT) == 0);
    cJSON_Delete(d);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 0);

    /* Invalid enum value on Qwen3 is reported. */
    cJSON *bad = cJSON_CreateObject();
    cJSON_AddStringToObject(bad, "text_type", "passage");
    TEST_ASSERT(text_type_is_query(bad, detail, EMBED_QWEN3_QUERY_INSTRUCT) == 0);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 1);
    cJSON_Delete(bad);

    /* text_type on a Perplexity-family model is rejected. */
    cJSON *p = cJSON_CreateObject();
    cJSON_AddStringToObject(p, "text_type", "query");
    TEST_ASSERT(text_type_is_query(p, detail, NULL) == 0);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 2);
    cJSON_Delete(p);

    cJSON_Delete(detail);
}

static void test_model_registry(void) {
    model_slot slot = model_slot_for_id("gte-Qwen2-1.5B-instruct");
    TEST_ASSERT(slot == MODEL_GTE_QWEN2_15);
    if (slot == MODEL_UNKNOWN)
        return;
    TEST_ASSERT(k_models[slot].dim == 1536);
    TEST_ASSERT(k_models[slot].min_dim == 1536);
    TEST_ASSERT(k_models[slot].attention_mode == EMBED_ATTENTION_BIDIRECTIONAL);
    TEST_ASSERT(k_models[slot].pooling_mode == EMBED_POOL_LAST_TOKEN);
    TEST_ASSERT(k_models[slot].normalize_embeddings == 1);
    TEST_ASSERT(strcmp(k_models[slot].query_instruct, EMBED_GTE_QWEN2_QUERY_INSTRUCT) == 0);
}

static void test_collect_job_batch_deadline(void) {
    http_server s;
    memset(&s, 0, sizeof(s));
    s.batch_size = 32;
    s.batch_wait_us = 1000;
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);

    uint64_t first_ns = nstime() - 5000000u;
    job first = {.srv = &s, .created_ns = first_ns};
    job before_deadline = {.srv = &s, .created_ns = first_ns + 500000u};
    job after_deadline = {.srv = &s, .created_ns = first_ns + 2000000u};
    snprintf(first.path, sizeof(first.path), "%s", "/v1/embeddings");
    snprintf(before_deadline.path, sizeof(before_deadline.path), "%s", "/v1/embeddings");
    snprintf(after_deadline.path, sizeof(after_deadline.path), "%s", "/v1/embeddings");
    enqueue_job(&first);
    enqueue_job(&before_deadline);
    enqueue_job(&after_deadline);

    job *batch[3] = {0};
    TEST_ASSERT(collect_job_batch(&s, batch, 3) == 2);
    TEST_ASSERT(batch[0] == &first);
    TEST_ASSERT(batch[1] == &before_deadline);
    TEST_ASSERT(s.job_head == &after_deadline);
    TEST_ASSERT(s.job_tail == &after_deadline);

    s.job_head = NULL;
    s.job_tail = NULL;
    pthread_cond_destroy(&s.cv);
    pthread_mutex_destroy(&s.mu);
}

static void test_collect_job_batch_zero_wait(void) {
    http_server s;
    memset(&s, 0, sizeof(s));
    s.batch_size = 32;
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);

    job first = {.srv = &s, .created_ns = nstime()};
    job second = {.srv = &s, .created_ns = nstime()};
    snprintf(first.path, sizeof(first.path), "%s", "/v1/embeddings");
    snprintf(second.path, sizeof(second.path), "%s", "/v1/embeddings");
    enqueue_job(&first);
    enqueue_job(&second);

    job *batch[2] = {0};
    TEST_ASSERT(collect_job_batch(&s, batch, 2) == 2);
    TEST_ASSERT(batch[0] == &first);
    TEST_ASSERT(batch[1] == &second);
    TEST_ASSERT(s.job_head == NULL);
    TEST_ASSERT(s.job_tail == NULL);

    pthread_cond_destroy(&s.cv);
    pthread_mutex_destroy(&s.mu);
}

/* ====================================================================
 * HTTP request-path tests against an in-process server
 * ==================================================================== */

#define TEST_API_KEY "tt-test-key"

static int find_free_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t alen = sizeof(a);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0 ||
        getsockname(fd, (struct sockaddr *)&a, &alen) != 0) {
        close(fd);
        return -1;
    }
    int port = ntohs(a.sin_port);
    close(fd);
    return port;
}

static int tcp_connect(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    struct sockaddr_in a;
    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int send_all(int fd, const char *buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, buf + off, n - off);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        off += (size_t)w;
    }
    return 0;
}

/* Send one request; raw == NULL builds a normal request, otherwise `raw`
 * bytes go on the wire verbatim (for malformed-HTTP checks). */
static int http_send(int fd,
                     const char *method,
                     const char *path,
                     const char *auth,
                     const char *body,
                     const char *raw) {
    if (raw)
        return send_all(fd, raw, strlen(raw));
    char head[512];
    size_t blen = body ? strlen(body) : 0;
    int n = snprintf(head, sizeof(head),
                     "%s %s HTTP/1.1\r\n"
                     "Host: 127.0.0.1\r\n"
                     "%s%s%s"
                     "Content-Type: application/json\r\n"
                     "Content-Length: %zu\r\n"
                     "\r\n",
                     method, path, auth ? "Authorization: Bearer " : "", auth ? auth : "",
                     auth ? "\r\n" : "", blen);
    if (n < 0 || (size_t)n >= sizeof(head))
        return -1;
    if (send_all(fd, head, (size_t)n) != 0)
        return -1;
    return blen ? send_all(fd, body, blen) : 0;
}

/* Read one response: returns the status code (or -1) and the body, which
 * the caller frees. Reads exactly Content-Length body bytes so it works
 * with keep-alive connections. */
static int http_recv(int fd, char **out_body) {
    *out_body = NULL;
    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    if (!buf)
        return -1;
    char *hdr_end = NULL;
    while (!hdr_end) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) {
                free(buf);
                return -1;
            }
            buf = nb;
        }
        ssize_t r = read(fd, buf + len, cap - len - 1);
        if (r <= 0) {
            free(buf);
            return -1;
        }
        len += (size_t)r;
        buf[len] = '\0';
        hdr_end = strstr(buf, "\r\n\r\n");
    }

    int status = -1;
    if (sscanf(buf, "HTTP/1.1 %d", &status) != 1) {
        free(buf);
        return -1;
    }
    size_t content_length = 0;
    const char *cl = strstr(buf, "Content-Length:");
    if (cl && cl < hdr_end)
        content_length = strtoul(cl + 15, NULL, 10);

    size_t body_off = (size_t)(hdr_end - buf) + 4;
    size_t need = body_off + content_length;
    if (need + 1 > cap) {
        char *nb = realloc(buf, need + 1);
        if (!nb) {
            free(buf);
            return -1;
        }
        buf = nb;
    }
    while (len < need) {
        ssize_t r = read(fd, buf + len, need - len);
        if (r <= 0) {
            free(buf);
            return -1;
        }
        len += (size_t)r;
    }
    buf[need] = '\0';
    *out_body = strdup(buf + body_off);
    free(buf);
    return *out_body ? status : -1;
}

/* One-shot request helper; returns status, body via out_body (caller frees,
 * may be NULL when only the status matters). */
static int http_req(int port,
                    const char *method,
                    const char *path,
                    const char *auth,
                    const char *body,
                    const char *raw,
                    char **out_body) {
    int fd = tcp_connect(port);
    if (fd < 0)
        return -1;
    char *resp = NULL;
    int status = -1;
    if (http_send(fd, method, path, auth, body, raw) == 0)
        status = http_recv(fd, &resp);
    close(fd);
    if (out_body)
        *out_body = resp;
    else
        free(resp);
    return status;
}

typedef struct {
    embed_server_model_spec_t spec[5];
    embed_server_config_t cfg;
    int rc;
} srv_ctx;

static void *srv_thread_main(void *arg) {
    srv_ctx *s = (srv_ctx *)arg;
    s->rc = embed_run_server(&s->cfg);
    return NULL;
}

static cJSON *parse_embeddings_body(const char *body, int want_items) {
    cJSON *root = cJSON_Parse(body);
    TEST_ASSERT(root != NULL);
    if (!root)
        return NULL;
    cJSON *data = cJSON_GetObjectItemCaseSensitive(root, "data");
    TEST_ASSERT(cJSON_IsArray(data));
    TEST_ASSERT(cJSON_GetArraySize(data) == want_items);
    if (!cJSON_IsArray(data) || cJSON_GetArraySize(data) != want_items) {
        cJSON_Delete(root);
        return NULL;
    }
    return root;
}

static void test_http_embeddings(int port) {
    char *body = NULL;

    /* Routing. */
    TEST_ASSERT(http_req(port, "OPTIONS", "/v1/embeddings", NULL, NULL, NULL, NULL) == 204);
    TEST_ASSERT(http_req(port, "GET", "/v1/embeddings", TEST_API_KEY, NULL, NULL, NULL) == 404);
    TEST_ASSERT(http_req(port, "POST", "/nope", TEST_API_KEY, "{}", NULL, NULL) == 404);

    /* Auth: required, and the exact key. */
    const char *ok_req = "{\"model\":\"pplx-embed-v1-0.6b\",\"input\":\"hello world\","
                         "\"encoding_format\":\"float\"}";
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", NULL, ok_req, NULL, NULL) == 401);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", "wrong-key", ok_req, NULL, NULL) == 401);

    /* Malformed HTTP (400) and malformed JSON (422 detail). */
    TEST_ASSERT(http_req(port, NULL, NULL, NULL, NULL, "GARBAGE\r\n\r\n", NULL) == 400);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY, "{nope", NULL, NULL) == 422);

    /* Validation: missing/unknown model, wrong endpoint family, valid but
     * unloaded id, bad encoding enum, dimensions out of range; the
     * API answers validation errors with 422 detail arrays. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY, "{\"input\":\"hi\"}", NULL,
                         NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"foo\",\"input\":\"hi\"}", NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":\"hi\"}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-4b\",\"input\":\"hi\"}", NULL, NULL) == 503);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\",\"input\":\"hi\","
                         "\"encoding_format\":\"yaml\"}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\",\"input\":\"hi\","
                         "\"dimensions\":64}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":[]}",
                         NULL, NULL) == 422);

    /* Contextual endpoint: standard id is the wrong enum, the 4B contextual
     * id is valid but not loaded. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":[[\"a\"]]}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-4b\","
                         "\"input\":[[\"a\"]]}",
                         NULL, NULL) == 503);

    /* Happy path, float encoding, single string input. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY, ok_req, NULL, &body) == 200);
    cJSON *root = body ? parse_embeddings_body(body, 1) : NULL;
    if (root) {
        cJSON *data = cJSON_GetObjectItemCaseSensitive(root, "data");
        cJSON *item = cJSON_GetArrayItem(data, 0);
        cJSON *emb = cJSON_GetObjectItemCaseSensitive(item, "embedding");
        cJSON *idx = cJSON_GetObjectItemCaseSensitive(item, "index");
        TEST_ASSERT(cJSON_IsArray(emb));
        TEST_ASSERT(cJSON_GetArraySize(emb) == 1024);
        TEST_ASSERT(cJSON_IsNumber(idx) && idx->valueint == 0);
        int nonzero = 0;
        cJSON *v;
        cJSON_ArrayForEach(v, emb) {
            TEST_ASSERT(cJSON_IsNumber(v));
            if (v->valuedouble != 0.0)
                nonzero = 1;
        }
        TEST_ASSERT(nonzero);
        cJSON *usage = cJSON_GetObjectItemCaseSensitive(root, "usage");
        cJSON *total = usage ? cJSON_GetObjectItemCaseSensitive(usage, "total_tokens") : NULL;
        TEST_ASSERT(total && cJSON_IsNumber(total) && total->valueint >= 1);
        cJSON_Delete(root);
    }
    free(body);
    body = NULL;

    /* Batch order: one embedding per input, indexes 0..n-1, and different
     * texts produce different vectors. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":[\"hello\",\"world\",\"held\"],"
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    root = body ? parse_embeddings_body(body, 3) : NULL;
    if (root) {
        cJSON *data = cJSON_GetObjectItemCaseSensitive(root, "data");
        cJSON *e0 = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(data, 0), "embedding");
        cJSON *e1 = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(data, 1), "embedding");
        for (int i = 0; i < 3; i++) {
            cJSON *idx = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(data, i), "index");
            TEST_ASSERT(cJSON_IsNumber(idx) && idx->valueint == i);
        }
        int differ = 0;
        for (int i = 0; i < 1024 && !differ; i++) {
            if (cJSON_GetArrayItem(e0, i)->valuedouble != cJSON_GetArrayItem(e1, i)->valuedouble)
                differ = 1;
        }
        TEST_ASSERT(differ);
        cJSON_Delete(root);
    }
    free(body);
    body = NULL;

    /* Default encoding is base64_int8: a string of 1024 quantized bytes. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":\"hello\"}",
                         NULL, &body) == 200);
    root = body ? parse_embeddings_body(body, 1) : NULL;
    if (root) {
        cJSON *emb = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(root, "data"), 0), "embedding");
        TEST_ASSERT(cJSON_IsString(emb));
        TEST_ASSERT(strlen(emb->valuestring) == ((1024 + 2) / 3) * 4);
        cJSON_Delete(root);
    }
    free(body);
    body = NULL;

    /* Matryoshka truncation via dimensions. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":\"hello\",\"dimensions\":128,"
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    root = body ? parse_embeddings_body(body, 1) : NULL;
    if (root) {
        cJSON *emb = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(root, "data"), 0), "embedding");
        TEST_ASSERT(cJSON_IsArray(emb) && cJSON_GetArraySize(emb) == 128);
        cJSON_Delete(root);
    }
    free(body);
    body = NULL;

    /* Qwen3 appends its terminal token, supports 32D MRL output, and
     * re-normalizes the truncated prefix. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"Qwen3-Embedding-0.6B\","
                         "\"input\":\"hello\",\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *embed_full = body ? parse_embeddings_body(body, 1) : NULL;
    free(body);
    body = NULL;
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"Qwen3-Embedding-0.6B\","
                         "\"input\":\"hello\",\"dimensions\":32,"
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *embed_mrl = body ? parse_embeddings_body(body, 1) : NULL;
    if (embed_full && embed_mrl) {
        cJSON *full_data = cJSON_GetObjectItemCaseSensitive(embed_full, "data");
        cJSON *mrl_data = cJSON_GetObjectItemCaseSensitive(embed_mrl, "data");
        cJSON *full_emb =
            cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(full_data, 0), "embedding");
        cJSON *mrl_emb =
            cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(mrl_data, 0), "embedding");
        TEST_ASSERT(cJSON_IsArray(full_emb) && cJSON_GetArraySize(full_emb) == 1024);
        TEST_ASSERT(cJSON_IsArray(mrl_emb) && cJSON_GetArraySize(mrl_emb) == 32);
        double prefix_diff = 0.0;
        for (int i = 0; i < 32; i++) {
            double d = fabs(cJSON_GetArrayItem(full_emb, i)->valuedouble -
                            cJSON_GetArrayItem(mrl_emb, i)->valuedouble);
            if (d > prefix_diff)
                prefix_diff = d;
        }
        TEST_ASSERT(prefix_diff > 0.0);
        cJSON *usage = cJSON_GetObjectItemCaseSensitive(embed_mrl, "usage");
        cJSON *total = usage ? cJSON_GetObjectItemCaseSensitive(usage, "total_tokens") : NULL;
        TEST_ASSERT(total && total->valueint == 2);
    }
    cJSON_Delete(embed_full);
    cJSON_Delete(embed_mrl);
    free(body);
    body = NULL;

    /* text_type=query must equal manually prepending the Qwen3 instruction with
     * text_type=document: both produce the same tokens, hence the same vector. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"Qwen3-Embedding-0.6B\",\"input\":\"hello\","
                         "\"text_type\":\"query\",\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *tt_query = body ? parse_embeddings_body(body, 1) : NULL;
    free(body);
    body = NULL;
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"Qwen3-Embedding-0.6B\",\"input\":\"Instruct: Given a web "
                         "search query, retrieve relevant passages that answer the "
                         "query\\nQuery:hello\",\"text_type\":\"document\","
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *tt_manual = body ? parse_embeddings_body(body, 1) : NULL;
    free(body);
    body = NULL;
    if (tt_query && tt_manual) {
        cJSON *qe = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(tt_query, "data"), 0), "embedding");
        cJSON *me = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(tt_manual, "data"), 0),
            "embedding");
        TEST_ASSERT(cJSON_IsArray(qe) && cJSON_IsArray(me) &&
                    cJSON_GetArraySize(qe) == cJSON_GetArraySize(me));
        double tt_diff = 0.0;
        int n = cJSON_GetArraySize(qe);
        for (int i = 0; i < n; i++) {
            double d = fabs(cJSON_GetArrayItem(qe, i)->valuedouble -
                            cJSON_GetArrayItem(me, i)->valuedouble);
            if (d > tt_diff)
                tt_diff = d;
        }
        TEST_ASSERT(tt_diff < 1e-6);
    }
    cJSON_Delete(tt_query);
    cJSON_Delete(tt_manual);

    /* GTE-Qwen2 uses the same retrieval task text but publishes a space after
     * "Query:". Verify the registry-specific prompt through the full path. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"gte-Qwen2-1.5B-instruct\",\"input\":\"hello\","
                         "\"text_type\":\"query\",\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *gte_query = body ? parse_embeddings_body(body, 1) : NULL;
    free(body);
    body = NULL;
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"gte-Qwen2-1.5B-instruct\",\"input\":\"Instruct: Given a "
                         "web search query, retrieve relevant passages that answer the "
                         "query\\nQuery: hello\",\"text_type\":\"document\","
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *gte_manual = body ? parse_embeddings_body(body, 1) : NULL;
    free(body);
    body = NULL;
    if (gte_query && gte_manual) {
        cJSON *qe = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(gte_query, "data"), 0),
            "embedding");
        cJSON *me = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(gte_manual, "data"), 0),
            "embedding");
        TEST_ASSERT(cJSON_IsArray(qe) && cJSON_IsArray(me));
        TEST_ASSERT(cJSON_GetArraySize(qe) == 1536 && cJSON_GetArraySize(me) == 1536);
        double gte_diff = 0.0;
        for (int i = 0; i < cJSON_GetArraySize(qe); i++) {
            double d = fabs(cJSON_GetArrayItem(qe, i)->valuedouble -
                            cJSON_GetArrayItem(me, i)->valuedouble);
            if (d > gte_diff)
                gte_diff = d;
        }
        TEST_ASSERT(gte_diff < 1e-6);
    }
    cJSON_Delete(gte_query);
    cJSON_Delete(gte_manual);

    /* text_type is rejected on a Perplexity-family model, and bad values 422. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\",\"input\":\"hello\","
                         "\"text_type\":\"query\"}",
                         NULL, &body) == 422);
    free(body);
    body = NULL;
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"Qwen3-Embedding-0.6B\",\"input\":\"hello\","
                         "\"text_type\":\"passage\"}",
                         NULL, &body) == 422);
    free(body);
    body = NULL;

    /* Concurrent clients: more requests in flight than the micro-batch cap
     * (-b 2) so the worker has to assemble several batches. */
    enum { N_CONC = 6 };
    int fds[N_CONC];
    for (int i = 0; i < N_CONC; i++) {
        fds[i] = tcp_connect(port);
        TEST_ASSERT(fds[i] >= 0);
        if (fds[i] >= 0)
            TEST_ASSERT(http_send(fds[i], "POST", "/v1/embeddings", TEST_API_KEY, ok_req, NULL) ==
                        0);
    }
    for (int i = 0; i < N_CONC; i++) {
        if (fds[i] < 0)
            continue;
        char *resp = NULL;
        TEST_ASSERT(http_recv(fds[i], &resp) == 200);
        free(resp);
        close(fds[i]);
    }
}

/* data[di].data[ci].embedding from a contextual response. */
static cJSON *ctx_chunk_emb(cJSON *root, int di, int ci) {
    cJSON *docs = cJSON_GetObjectItemCaseSensitive(root, "data");
    cJSON *doc = cJSON_GetArrayItem(docs, di);
    cJSON *chunks = cJSON_GetObjectItemCaseSensitive(doc, "data");
    cJSON *item = cJSON_GetArrayItem(chunks, ci);
    return cJSON_GetObjectItemCaseSensitive(item, "embedding");
}

static double emb_max_absdiff(cJSON *a, cJSON *b) {
    if (!cJSON_IsArray(a) || !cJSON_IsArray(b))
        return 1e9;
    int n = cJSON_GetArraySize(a);
    if (n != cJSON_GetArraySize(b))
        return 1e9;
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d =
            fabs(cJSON_GetArrayItem(a, i)->valuedouble - cJSON_GetArrayItem(b, i)->valuedouble);
        if (d > m)
            m = d;
    }
    return m;
}

static void test_http_contextual(int port) {
    char *body = NULL;

    /* Input must be an array of chunk arrays; chunks must be non-empty. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":\"hi\"}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":[[\"\"]]}",
                         NULL, NULL) == 422);

    /* Happy path: chunks concatenate per document with one separator token
     * between them, one embedding per chunk in document order. Fixture
     * token counts: hello=1, world=3, held=2, one separator in doc 0, so
     * usage must report 7. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":[[\"hello\",\"world\"],[\"held\"]],"
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *multi = body ? cJSON_Parse(body) : NULL;
    TEST_ASSERT(multi != NULL);
    free(body);
    body = NULL;
    if (multi) {
        cJSON *docs = cJSON_GetObjectItemCaseSensitive(multi, "data");
        TEST_ASSERT(cJSON_IsArray(docs) && cJSON_GetArraySize(docs) == 2);
        for (int di = 0; di < 2; di++) {
            cJSON *doc = cJSON_GetArrayItem(docs, di);
            cJSON *idx = cJSON_GetObjectItemCaseSensitive(doc, "index");
            cJSON *chunks = cJSON_GetObjectItemCaseSensitive(doc, "data");
            TEST_ASSERT(idx && idx->valueint == di);
            TEST_ASSERT(cJSON_IsArray(chunks));
            TEST_ASSERT(cJSON_GetArraySize(chunks) == (di == 0 ? 2 : 1));
            for (int ci = 0; ci < cJSON_GetArraySize(chunks); ci++) {
                cJSON *item = cJSON_GetArrayItem(chunks, ci);
                cJSON *cidx = cJSON_GetObjectItemCaseSensitive(item, "index");
                cJSON *emb = ctx_chunk_emb(multi, di, ci);
                TEST_ASSERT(cidx && cidx->valueint == ci);
                TEST_ASSERT(cJSON_IsArray(emb) && cJSON_GetArraySize(emb) == 1024);
            }
        }
        cJSON *usage = cJSON_GetObjectItemCaseSensitive(multi, "usage");
        cJSON *total = usage ? cJSON_GetObjectItemCaseSensitive(usage, "total_tokens") : NULL;
        TEST_ASSERT(total && total->valueint == 7);
    }

    /* A single-chunk document has no separators and its pooling span covers
     * the whole sequence, so it must match the standard embedding of the
     * same text from the same weights. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":[[\"hello\"]],"
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *single = body ? cJSON_Parse(body) : NULL;
    TEST_ASSERT(single != NULL);
    free(body);
    body = NULL;
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":\"hello\","
                         "\"encoding_format\":\"float\"}",
                         NULL, &body) == 200);
    cJSON *std = body ? cJSON_Parse(body) : NULL;
    TEST_ASSERT(std != NULL);
    free(body);
    body = NULL;

    if (multi && single && std) {
        cJSON *std_emb = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(std, "data"), 0), "embedding");
        cJSON *alone = ctx_chunk_emb(single, 0, 0);
        cJSON *in_doc = ctx_chunk_emb(multi, 0, 0);
        TEST_ASSERT(emb_max_absdiff(std_emb, alone) < 1e-4);
        /* Whole-document attention: the same chunk next to a neighbor must
         * pool to a different vector. */
        TEST_ASSERT(emb_max_absdiff(alone, in_doc) > 1e-3);
    }
    cJSON_Delete(multi);
    cJSON_Delete(single);
    cJSON_Delete(std);
}

static void test_http_rerank(int port) {
    char *body = NULL;

    TEST_ASSERT(http_req(port, "POST", "/v1/rerank", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"query\":\"hello\",\"documents\":[\"world\"]}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/rerank", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-late-0.6b\","
                         "\"query\":\"\",\"documents\":[\"world\"]}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/rerank", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-late-0.6b\","
                         "\"query\":\"hello\",\"documents\":[]}",
                         NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/rerank", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-late-0.6b\","
                         "\"query\":\"hello\",\"documents\":[\"world\"],"
                         "\"top_n\":1,\"top_k\":1}",
                         NULL, NULL) == 422);

    const char *request = "{\"model\":\"pplx-embed-v1-late-0.6b\","
                          "\"query\":\"hello\",\"documents\":["
                          "\"hello!\",\"world\",\"quoted \\\"document\\\"\"],"
                          "\"top_n\":2,\"return_documents\":true}";
    TEST_ASSERT(http_req(port, "POST", "/v1/rerank", TEST_API_KEY, request, NULL, &body) == 200);
    cJSON *root = body ? cJSON_Parse(body) : NULL;
    TEST_ASSERT(root != NULL);
    if (root) {
        cJSON *object = cJSON_GetObjectItemCaseSensitive(root, "object");
        cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");
        cJSON *results = cJSON_GetObjectItemCaseSensitive(root, "results");
        TEST_ASSERT(cJSON_IsString(object) && strcmp(object->valuestring, "list") == 0);
        TEST_ASSERT(cJSON_IsString(model) &&
                    strcmp(model->valuestring, "pplx-embed-v1-late-0.6b") == 0);
        TEST_ASSERT(cJSON_IsArray(results) && cJSON_GetArraySize(results) == 2);
        double previous = INFINITY;
        int seen[3] = {0};
        cJSON *result;
        cJSON_ArrayForEach(result, results) {
            cJSON *index = cJSON_GetObjectItemCaseSensitive(result, "index");
            cJSON *score = cJSON_GetObjectItemCaseSensitive(result, "relevance_score");
            cJSON *document = cJSON_GetObjectItemCaseSensitive(result, "document");
            TEST_ASSERT(cJSON_IsNumber(index) && index->valueint >= 0 && index->valueint < 3);
            TEST_ASSERT(cJSON_IsNumber(score) && score->valuedouble <= previous);
            TEST_ASSERT(cJSON_IsString(document));
            if (cJSON_IsNumber(index) && index->valueint >= 0 && index->valueint < 3)
                seen[index->valueint]++;
            if (cJSON_IsNumber(score))
                previous = score->valuedouble;
        }
        TEST_ASSERT(seen[0] + seen[1] + seen[2] == 2);

        cJSON *usage = cJSON_GetObjectItemCaseSensitive(root, "usage");
        cJSON *query_tokens =
            usage ? cJSON_GetObjectItemCaseSensitive(usage, "query_tokens") : NULL;
        cJSON *document_tokens =
            usage ? cJSON_GetObjectItemCaseSensitive(usage, "document_tokens") : NULL;
        cJSON *total_tokens =
            usage ? cJSON_GetObjectItemCaseSensitive(usage, "total_tokens") : NULL;
        TEST_ASSERT(query_tokens && document_tokens && total_tokens);
        if (query_tokens && document_tokens && total_tokens) {
            TEST_ASSERT(cJSON_IsNumber(query_tokens));
            TEST_ASSERT(cJSON_IsNumber(document_tokens));
            TEST_ASSERT(cJSON_IsNumber(total_tokens));
            TEST_ASSERT(query_tokens->valueint == 32);
            TEST_ASSERT(document_tokens->valueint > 0);
            TEST_ASSERT(total_tokens->valueint ==
                        query_tokens->valueint + document_tokens->valueint);
        }
        cJSON_Delete(root);
    }
    free(body);
    body = NULL;

    TEST_ASSERT(http_req(port, "POST", "/v1/rerank", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-late-0.6b\","
                         "\"query\":\"hello\","
                         "\"documents\":[\"hello\",\"world\"],\"top_k\":1}",
                         NULL, &body) == 200);
    root = body ? cJSON_Parse(body) : NULL;
    TEST_ASSERT(root != NULL);
    if (root) {
        cJSON *results = cJSON_GetObjectItemCaseSensitive(root, "results");
        TEST_ASSERT(cJSON_IsArray(results) && cJSON_GetArraySize(results) == 1);
        cJSON *document =
            cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(results, 0), "document");
        TEST_ASSERT(document == NULL);
        cJSON_Delete(root);
    }
    free(body);
}

static void test_http_server(void) {
    char dir[1024], late_dir[1024], embed_dir[1024], gte_dir[1024];
    snprintf(dir, sizeof(dir), "%s/embed-srv-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(dir)) {
        fprintf(stderr, "FAIL: mkdtemp\n");
        test_failures++;
        return;
    }
    snprintf(late_dir, sizeof(late_dir), "%s/embed-srv-late-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(late_dir)) {
        fprintf(stderr, "FAIL: late mkdtemp\n");
        test_failures++;
        return;
    }
    snprintf(embed_dir, sizeof(embed_dir), "%s/embed-srv-qwen-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(embed_dir)) {
        fprintf(stderr, "FAIL: Qwen mkdtemp\n");
        test_failures++;
        return;
    }
    snprintf(gte_dir, sizeof(gte_dir), "%s/embed-srv-gte-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(gte_dir)) {
        fprintf(stderr, "FAIL: GTE mkdtemp\n");
        test_failures++;
        return;
    }
    /* The 0.6b server slot requires hidden_size 1024; everything else stays
     * tiny. The model vocab covers every tokenizer fixture id. */
    tm_dims_t dims = {1024, 2, 1, 64, 8, TF_VOCAB_SIZE};
    tm_dims_t gte_dims = {1536, 2, 1, 64, 8, TF_VOCAB_SIZE};
    if (tf_write_vocab(dir) != 0 || tm_write_model_dims(dir, "F32", &dims) != 0 ||
        tf_write_vocab(late_dir) != 0 || tm_write_model_dims(late_dir, "F32", &dims) != 0 ||
        tm_write_late_projection(late_dir, "F32", 128, 1024) != 0 ||
        tf_write_vocab(embed_dir) != 0 ||
        tm_write_qwen3_model_dims(embed_dir, "F32", &dims, TF_EOT_ID) != 0 ||
        tf_write_vocab(gte_dir) != 0 ||
        tm_write_qwen2_model_dims(gte_dir, "F32", &gte_dims, TF_EOT_ID, 0) != 0) {
        fprintf(stderr, "FAIL: fixture write\n");
        test_failures++;
        return;
    }

    int port = find_free_port();
    TEST_ASSERT(port > 0);

    srv_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    /* Both 0.6b slots serve the same tiny weights; the contextual slot
     * resolves its separator from the fixture's <|endoftext|>. */
    ctx.spec[0].id = "pplx-embed-v1-0.6b";
    ctx.spec[0].path = dir;
    ctx.spec[1].id = "pplx-embed-context-v1-0.6b";
    ctx.spec[1].path = dir;
    ctx.spec[2].id = "pplx-embed-v1-late-0.6b";
    ctx.spec[2].path = late_dir;
    ctx.spec[3].id = "Qwen3-Embedding-0.6B";
    ctx.spec[3].path = embed_dir;
    ctx.spec[4].id = "gte-Qwen2-1.5B-instruct";
    ctx.spec[4].path = gte_dir;
    ctx.cfg.models = ctx.spec;
    ctx.cfg.n_models = 5;
    ctx.cfg.port = port;
    ctx.cfg.batch_size = 2;
    ctx.cfg.batch_wait_us = 1000;
    ctx.cfg.api_key = TEST_API_KEY;

    pthread_t th;
    if (pthread_create(&th, NULL, srv_thread_main, &ctx) != 0) {
        fprintf(stderr, "FAIL: server thread\n");
        test_failures++;
        return;
    }

    int up = 0;
    for (int i = 0; i < 200; i++) { /* up to ~10 s */
        int fd = tcp_connect(port);
        if (fd >= 0) {
            close(fd);
            up = 1;
            break;
        }
        usleep(50 * 1000);
    }
    TEST_ASSERT(up);
    if (up) {
        test_http_embeddings(port);
        test_http_contextual(port);
        test_http_rerank(port);
    }

    stop_signal_handler(SIGTERM);
    pthread_join(th, NULL);
    TEST_ASSERT(ctx.rc == 0);
}

int main(void) {
    test_base64_rfc4648();
    test_quantize_int8_tanh();
    test_encode_embedding_int8();
    test_encode_embedding_binary();
    test_encode_embedding_base64_float32();
    test_append_embedding_value_float();
    test_encoding_from_root_family();
    test_text_type();
    test_model_registry();
    test_collect_job_batch_deadline();
    test_collect_job_batch_zero_wait();
    test_http_server();
    if (test_failures) {
        fprintf(stderr, "server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ok: server unit and HTTP request-path tests passed");
    return 0;
}
