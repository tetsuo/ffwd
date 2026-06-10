/* tests/test_server.c - tests for embed_server.c.
 * ds4-style: includes the whole server translation unit with the real main
 * compiled out, so statics are directly testable. Unit-tests the encoding
 * helpers, then boots the real server in-process (tiny 1024-dim hermetic
 * model + byte-vocab tokenizer fixture) and drives the full HTTP request
 * path over loopback sockets. Runs via `make test`. */

#define EMBED_SERVER_TEST
#include "../embed_server.c"

#include "tiny_model.h"
#include "tok_fixture.h"

#include <arpa/inet.h>
#include <netinet/in.h>

static int test_failures = 0;

#define TEST_ASSERT(cond) do {                                        \
    if (!(cond)) {                                                    \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        test_failures++;                                              \
    }                                                                 \
} while (0)

static void test_base64_rfc4648(void)
{
    /* RFC 4648 test vectors. */
    static const struct { const char *in, *out; } cases[] = {
        {"", ""}, {"f", "Zg=="}, {"fo", "Zm8="}, {"foo", "Zm9v"},
        {"foob", "Zm9vYg=="}, {"fooba", "Zm9vYmE="}, {"foobar", "Zm9vYmFy"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *got = base64_encode((const unsigned char *)cases[i].in,
                                  strlen(cases[i].in));
        TEST_ASSERT(strcmp(got, cases[i].out) == 0);
        free(got);
    }
}

static void test_quantize_int8_tanh(void)
{
    TEST_ASSERT(quantize_int8_tanh(0.0f) == 0);
    /* tanh saturates to +-1, scaled by 127. */
    TEST_ASSERT(quantize_int8_tanh(100.0f) == 127);
    TEST_ASSERT(quantize_int8_tanh(-100.0f) == -127);
    /* Symmetry. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == -quantize_int8_tanh(-0.5f));
    /* round(tanh(0.5) * 127) = round(58.69) = 59. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == 59);
}

static void test_encode_embedding_int8(void)
{
    float emb[4] = {0.0f, 0.5f, -0.5f, 100.0f};
    signed char expect[4] = {0, 59, -59, 127};
    char *got = encode_embedding(emb, 4, "base64_int8");
    char *want = base64_encode((const unsigned char *)expect, 4);
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);
}

static void test_encode_embedding_binary(void)
{
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

/* ====================================================================
 * HTTP request-path tests against an in-process server
 * ==================================================================== */

#define TEST_API_KEY "tt-test-key"

static int find_free_port(void)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
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

static int tcp_connect(int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
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

static int send_all(int fd, const char *buf, size_t n)
{
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, buf + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        off += (size_t)w;
    }
    return 0;
}

/* Send one request; raw == NULL builds a normal request, otherwise `raw`
 * bytes go on the wire verbatim (for malformed-HTTP checks). */
static int http_send(int fd, const char *method, const char *path,
                     const char *auth, const char *body, const char *raw)
{
    if (raw) return send_all(fd, raw, strlen(raw));
    char head[512];
    size_t blen = body ? strlen(body) : 0;
    int n = snprintf(head, sizeof(head),
                     "%s %s HTTP/1.1\r\n"
                     "Host: 127.0.0.1\r\n"
                     "%s%s%s"
                     "Content-Type: application/json\r\n"
                     "Content-Length: %zu\r\n"
                     "\r\n",
                     method, path,
                     auth ? "Authorization: Bearer " : "",
                     auth ? auth : "", auth ? "\r\n" : "",
                     blen);
    if (n < 0 || (size_t)n >= sizeof(head)) return -1;
    if (send_all(fd, head, (size_t)n) != 0) return -1;
    return blen ? send_all(fd, body, blen) : 0;
}

/* Read one response: returns the status code (or -1) and the body, which
 * the caller frees. Reads exactly Content-Length body bytes so it works
 * with keep-alive connections. */
static int http_recv(int fd, char **out_body)
{
    *out_body = NULL;
    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    if (!buf) return -1;
    char *hdr_end = NULL;
    while (!hdr_end) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) { free(buf); return -1; }
            buf = nb;
        }
        ssize_t r = read(fd, buf + len, cap - len - 1);
        if (r <= 0) { free(buf); return -1; }
        len += (size_t)r;
        buf[len] = '\0';
        hdr_end = strstr(buf, "\r\n\r\n");
    }

    int status = -1;
    if (sscanf(buf, "HTTP/1.1 %d", &status) != 1) { free(buf); return -1; }
    size_t content_length = 0;
    const char *cl = strstr(buf, "Content-Length:");
    if (cl && cl < hdr_end) content_length = strtoul(cl + 15, NULL, 10);

    size_t body_off = (size_t)(hdr_end - buf) + 4;
    size_t need = body_off + content_length;
    if (need + 1 > cap) {
        char *nb = realloc(buf, need + 1);
        if (!nb) { free(buf); return -1; }
        buf = nb;
    }
    while (len < need) {
        ssize_t r = read(fd, buf + len, need - len);
        if (r <= 0) { free(buf); return -1; }
        len += (size_t)r;
    }
    buf[need] = '\0';
    *out_body = strdup(buf + body_off);
    free(buf);
    return *out_body ? status : -1;
}

/* One-shot request helper; returns status, body via out_body (caller frees,
 * may be NULL when only the status matters). */
static int http_req(int port, const char *method, const char *path,
                    const char *auth, const char *body, const char *raw,
                    char **out_body)
{
    int fd = tcp_connect(port);
    if (fd < 0) return -1;
    char *resp = NULL;
    int status = -1;
    if (http_send(fd, method, path, auth, body, raw) == 0)
        status = http_recv(fd, &resp);
    close(fd);
    if (out_body) *out_body = resp;
    else free(resp);
    return status;
}

typedef struct {
    pplx_server_model_spec_t spec[2];
    pplx_server_config_t cfg;
    int rc;
} srv_ctx;

static void *srv_thread_main(void *arg)
{
    srv_ctx *s = (srv_ctx *)arg;
    s->rc = pplx_run_server(&s->cfg);
    return NULL;
}

static cJSON *parse_embeddings_body(const char *body, int want_items)
{
    cJSON *root = cJSON_Parse(body);
    TEST_ASSERT(root != NULL);
    if (!root) return NULL;
    cJSON *data = cJSON_GetObjectItemCaseSensitive(root, "data");
    TEST_ASSERT(cJSON_IsArray(data));
    TEST_ASSERT(cJSON_GetArraySize(data) == want_items);
    if (!cJSON_IsArray(data) || cJSON_GetArraySize(data) != want_items) {
        cJSON_Delete(root);
        return NULL;
    }
    return root;
}

static void test_http_embeddings(int port)
{
    char *body = NULL;

    /* CORS preflight and routing. */
    TEST_ASSERT(http_req(port, "OPTIONS", "/v1/embeddings", NULL, NULL, NULL,
                         NULL) == 204);
    TEST_ASSERT(http_req(port, "GET", "/v1/embeddings", TEST_API_KEY, NULL,
                         NULL, NULL) == 404);
    TEST_ASSERT(http_req(port, "POST", "/nope", TEST_API_KEY, "{}", NULL,
                         NULL) == 404);

    /* Auth: required, and the exact key. */
    const char *ok_req =
        "{\"model\":\"pplx-embed-v1-0.6b\",\"input\":\"hello world\","
        "\"encoding_format\":\"float\"}";
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", NULL, ok_req, NULL,
                         NULL) == 401);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", "wrong-key", ok_req,
                         NULL, NULL) == 401);

    /* Malformed HTTP (400) and malformed JSON (422 detail). */
    TEST_ASSERT(http_req(port, NULL, NULL, NULL, NULL,
                         "GARBAGE\r\n\r\n", NULL) == 400);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{nope", NULL, NULL) == 422);

    /* Validation: missing/unknown model, wrong endpoint family, valid but
     * unloaded id, bad encoding enum, dimensions out of range; the
     * API answers validation errors with 422 detail arrays. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"input\":\"hi\"}", NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"foo\",\"input\":\"hi\"}", NULL,
                         NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":\"hi\"}", NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-4b\",\"input\":\"hi\"}",
                         NULL, NULL) == 503);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\",\"input\":\"hi\","
                         "\"encoding_format\":\"yaml\"}", NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\",\"input\":\"hi\","
                         "\"dimensions\":64}", NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":[]}", NULL, NULL) == 422);

    /* Contextual endpoint: standard id is the wrong enum, the 4B contextual
     * id is valid but not loaded. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings",
                         TEST_API_KEY,
                         "{\"model\":\"pplx-embed-v1-0.6b\","
                         "\"input\":[[\"a\"]]}", NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings",
                         TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-4b\","
                         "\"input\":[[\"a\"]]}", NULL, NULL) == 503);

    /* Happy path, float encoding, single string input. */
    TEST_ASSERT(http_req(port, "POST", "/v1/embeddings", TEST_API_KEY, ok_req,
                         NULL, &body) == 200);
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
            if (v->valuedouble != 0.0) nonzero = 1;
        }
        TEST_ASSERT(nonzero);
        cJSON *usage = cJSON_GetObjectItemCaseSensitive(root, "usage");
        cJSON *total = usage
            ? cJSON_GetObjectItemCaseSensitive(usage, "total_tokens") : NULL;
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
        cJSON *e0 = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(data, 0), "embedding");
        cJSON *e1 = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(data, 1), "embedding");
        for (int i = 0; i < 3; i++) {
            cJSON *idx = cJSON_GetObjectItemCaseSensitive(
                cJSON_GetArrayItem(data, i), "index");
            TEST_ASSERT(cJSON_IsNumber(idx) && idx->valueint == i);
        }
        int differ = 0;
        for (int i = 0; i < 1024 && !differ; i++) {
            if (cJSON_GetArrayItem(e0, i)->valuedouble !=
                cJSON_GetArrayItem(e1, i)->valuedouble)
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
                         "\"input\":\"hello\"}", NULL, &body) == 200);
    root = body ? parse_embeddings_body(body, 1) : NULL;
    if (root) {
        cJSON *emb = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(root, "data"),
                               0), "embedding");
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
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(root, "data"),
                               0), "embedding");
        TEST_ASSERT(cJSON_IsArray(emb) && cJSON_GetArraySize(emb) == 128);
        cJSON_Delete(root);
    }
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
            TEST_ASSERT(http_send(fds[i], "POST", "/v1/embeddings",
                                  TEST_API_KEY, ok_req, NULL) == 0);
    }
    for (int i = 0; i < N_CONC; i++) {
        if (fds[i] < 0) continue;
        char *resp = NULL;
        TEST_ASSERT(http_recv(fds[i], &resp) == 200);
        free(resp);
        close(fds[i]);
    }
}

/* data[di].data[ci].embedding from a contextual response. */
static cJSON *ctx_chunk_emb(cJSON *root, int di, int ci)
{
    cJSON *docs = cJSON_GetObjectItemCaseSensitive(root, "data");
    cJSON *doc = cJSON_GetArrayItem(docs, di);
    cJSON *chunks = cJSON_GetObjectItemCaseSensitive(doc, "data");
    cJSON *item = cJSON_GetArrayItem(chunks, ci);
    return cJSON_GetObjectItemCaseSensitive(item, "embedding");
}

static double emb_max_absdiff(cJSON *a, cJSON *b)
{
    if (!cJSON_IsArray(a) || !cJSON_IsArray(b)) return 1e9;
    int n = cJSON_GetArraySize(a);
    if (n != cJSON_GetArraySize(b)) return 1e9;
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(cJSON_GetArrayItem(a, i)->valuedouble -
                        cJSON_GetArrayItem(b, i)->valuedouble);
        if (d > m) m = d;
    }
    return m;
}

static void test_http_contextual(int port)
{
    char *body = NULL;

    /* Input must be an array of chunk arrays; chunks must be non-empty. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings",
                         TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":\"hi\"}", NULL, NULL) == 422);
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings",
                         TEST_API_KEY,
                         "{\"model\":\"pplx-embed-context-v1-0.6b\","
                         "\"input\":[[\"\"]]}", NULL, NULL) == 422);

    /* Happy path: chunks concatenate per document with one separator token
     * between them, one embedding per chunk in document order. Fixture
     * token counts: hello=1, world=3, held=2, one separator in doc 0, so
     * usage must report 7. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings",
                         TEST_API_KEY,
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
                TEST_ASSERT(cJSON_IsArray(emb) &&
                            cJSON_GetArraySize(emb) == 1024);
            }
        }
        cJSON *usage = cJSON_GetObjectItemCaseSensitive(multi, "usage");
        cJSON *total = usage
            ? cJSON_GetObjectItemCaseSensitive(usage, "total_tokens") : NULL;
        TEST_ASSERT(total && total->valueint == 7);
    }

    /* A single-chunk document has no separators and its pooling span covers
     * the whole sequence, so it must match the standard embedding of the
     * same text from the same weights. */
    TEST_ASSERT(http_req(port, "POST", "/v1/contextualizedembeddings",
                         TEST_API_KEY,
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
            cJSON_GetArrayItem(cJSON_GetObjectItemCaseSensitive(std, "data"),
                               0), "embedding");
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

static void test_http_server(void)
{
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/pplx-srv-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(dir)) {
        fprintf(stderr, "FAIL: mkdtemp\n");
        test_failures++;
        return;
    }
    /* The 0.6b server slot requires hidden_size 1024; everything else stays
     * tiny. The model vocab covers every tokenizer fixture id. */
    tm_dims_t dims = {1024, 2, 1, 64, 8, TF_VOCAB_SIZE};
    if (tf_write_vocab(dir) != 0 ||
        tm_write_model_dims(dir, "F32", &dims) != 0) {
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
    ctx.cfg.models = ctx.spec;
    ctx.cfg.n_models = 2;
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
    for (int i = 0; i < 200; i++) {        /* up to ~10 s */
        int fd = tcp_connect(port);
        if (fd >= 0) { close(fd); up = 1; break; }
        usleep(50 * 1000);
    }
    TEST_ASSERT(up);
    if (up) {
        test_http_embeddings(port);
        test_http_contextual(port);
    }

    stop_signal_handler(SIGTERM);
    pthread_join(th, NULL);
    TEST_ASSERT(ctx.rc == 0);
}

int main(void)
{
    test_base64_rfc4648();
    test_quantize_int8_tanh();
    test_encode_embedding_int8();
    test_encode_embedding_binary();
    test_http_server();
    if (test_failures) {
        fprintf(stderr, "server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ok: server unit and HTTP request-path tests passed");
    return 0;
}
