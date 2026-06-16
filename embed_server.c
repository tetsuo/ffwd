#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "embed_build.h"
#include "embed_server.h"
#include "embed.h"
#include "qwen_safetensors.h"
#include "qwen_kernels.h"
#include "qwen_tokenizer.h"
#include "wordpiece_tokenizer.h"

#ifdef USE_MLX
#include "embed_mlx.h"
#endif
#ifdef USE_CUDA
#include "embed_cuda.h"
#endif

#include "deps/ae/ae.h"
#include "deps/ae/anet.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <cjson/cJSON.h>

#define EMBED_HTTP_IO_BUF            8192
#define EMBED_HTTP_MAX_HEADER        (64u * 1024u)
#define EMBED_HTTP_MAX_BODY          (64u * 1024u * 1024u)
#define EMBED_HTTP_MAX_PATH          255u
#define EMBED_HTTP_BACKLOG           128
#define EMBED_HTTP_SETSIZE           10240
#define EMBED_HTTP_CLIENT_TIMEOUT_MS 30000
#define EMBED_HTTP_SWEEP_MS          1000

#define EMBED_API_MAX_STANDARD_INPUTS  512
#define EMBED_API_MAX_CONTEXT_DOCS     512
#define EMBED_API_MAX_CONTEXT_CHUNKS   16000
#define EMBED_API_MAX_RERANK_DOCUMENTS 1000
#define EMBED_API_MAX_ITEM_TOKENS      32768
#define EMBED_API_MAX_TOTAL_TOKENS     120000
#define EMBED_LATE_QUERY_TOKENS        32
#define EMBED_LATE_MASK_TOKEN_ID       151642
#define EMBED_LATE_QUERY_PREFIX_ID     151669
#define EMBED_LATE_DOCUMENT_PREFIX_ID  151670
/* `text_type: query` prepends the model's published retrieval instruction.
 * Documents and the default pass through unchanged. Callers needing a custom
 * instruction prepend it themselves and leave text_type at "document". */
#define EMBED_QWEN3_QUERY_INSTRUCT \
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
#define EMBED_GTE_QWEN2_QUERY_INSTRUCT                                                \
    "Instruct: Given a web search query, retrieve relevant passages that answer the " \
    "query\nQuery: "
/* Chosen from the L4 scheduler sweep: -b 32 gained ~24% concurrent
 * short-request throughput over 8 with no long-document penalty, while 128
 * was no faster and inflated long-document tail latency. */
#define EMBED_SERVER_DEFAULT_BATCH_SIZE       32
#define EMBED_SERVER_DEFAULT_MAX_BATCH_TOKENS 16384
/* Microseconds the worker waits for more requests before dispatching a batch.
 * CUDA uses a 1 ms window so arrivals group into one launch and keep the GPU
 * busy; MLX and CPU dispatch immediately, where waiting only adds latency.
 * Override with --batch-wait-us. */
#define EMBED_SERVER_BATCH_WAIT_US       0
#define EMBED_SERVER_CUDA_BATCH_WAIT_US  1000
#define EMBED_SERVER_MICROBATCH_MAX_JOBS 128
#define EMBED_MLX_MEMORY_BUDGET_PERCENT  90
#define EMBED_MLX_RESIDENT_MULTIPLIER    2

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} sbuf;

static void die_oom(void) {
    fprintf(stderr, "embed-server: out of memory\n");
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p)
        die_oom();
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p)
        die_oom();
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p)
        die_oom();
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static uint64_t mstime(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000u + (uint64_t)tv.tv_usec / 1000u;
}

static uint64_t nstime(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000000u + (uint64_t)tv.tv_usec * 1000u;
}

static double ns_to_ms(uint64_t ns) { return (double)ns / 1000000.0; }

static void server_log(const char *fmt, ...) {
    char ts[32];
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof(ts), "%m%d %H:%M:%S", &tmv);
    fprintf(stderr, "%s ", ts);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void sbuf_reserve(sbuf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1)
        die_oom();
    size_t need = b->len + add + 1;
    if (need <= b->cap)
        return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2)
            die_oom();
        cap *= 2;
    }
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void sbuf_append(sbuf *b, const void *p, size_t n) {
    sbuf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void sbuf_putc(sbuf *b, char c) { sbuf_append(b, &c, 1); }
static void sbuf_puts(sbuf *b, const char *s) { sbuf_append(b, s, strlen(s)); }

static void sbuf_clear(sbuf *b) {
    b->len = 0;
    if (b->ptr)
        b->ptr[0] = '\0';
}

static void sbuf_printf(sbuf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0)
        die_oom();
    sbuf_reserve(b, (size_t)n);
    int n2 = vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    if (n2 < 0 || n2 != n)
        die_oom();
    b->len += (size_t)n;
}

static void sbuf_free(sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

typedef enum {
    MODEL_STD_06,
    MODEL_STD_4,
    MODEL_QWEN3_06,
    MODEL_QWEN3_4,
    MODEL_QWEN3_8,
    MODEL_GTE_QWEN2_15,
    MODEL_MINILM_L6,
    MODEL_BGE_SMALL,
    MODEL_BGE_BASE,
    MODEL_BGE_LARGE,
    MODEL_CTX_06,
    MODEL_CTX_4,
    MODEL_LATE_06,
    MODEL_COUNT,
    MODEL_UNKNOWN
} model_slot;

typedef enum { MODEL_KIND_STANDARD, MODEL_KIND_CONTEXTUAL, MODEL_KIND_LATE } model_kind;

/* Which embedding API a model speaks. pplx-embed models follow the Perplexity
 * API: the canonical output is the tanh int8 quantization, so the default
 * encoding is base64_int8 and "float" is the int8-decoded view (int8/128).
 * Instruction embedding models follow the OpenAI-compatible API: "float" is
 * the true float32 vector and "base64" is base64 of little-endian float32.
 * int8 quantization underuses the range of L2-normalized vectors, so it is not
 * used for these models. */
typedef enum {
    EMBED_API_PERPLEXITY = 0,
    EMBED_API_OPENAI = 1,
} embedding_api_t;

typedef struct {
    const char *id;
    model_kind kind;
    int dim;
    int min_dim;
    int token_dim;
    embed_attention_mode_t attention_mode;
    embed_pooling_mode_t pooling_mode;
    int normalize_embeddings;
    embedding_api_t api;
    const char *query_instruct;
} model_info;

static const model_info k_models[] = {
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

static model_slot model_slot_for_id(const char *id) {
    if (!id)
        return MODEL_UNKNOWN;
    for (int i = 0; i < MODEL_COUNT; i++) {
        if (!strcmp(id, k_models[i].id))
            return (model_slot)i;
    }
    return MODEL_UNKNOWN;
}

typedef struct {
    const model_info *info;
    char *path;
    qwen_tokenizer_t *tok;
    qwen_tokenizer_workspace_t *tok_ws;
    /* BERT-family WordPiece tokenizer, selected by file probe in load_one_model.
     * When wp_tok is set, tokenize_one wraps the ids with [CLS]/[SEP] and the
     * Qwen byte-level BPE fields above stay NULL. */
    wordpiece_tokenizer_t *wp_tok;
    wordpiece_workspace_t *wp_tok_ws;
    int cls_id;
    int sep_id;
    int context_separator_id;
    int late_mask_id;
    int late_query_prefix_id;
    int late_document_prefix_id;
    int late_skip_ids[64];
    int n_late_skip_ids;
    int append_terminal_token;
    int terminal_token_id;
    int renormalize_truncated;
    embed_model_t *cpu_model;
    embed_workspace_t *cpu_ws;
    embed_late_model_t *cpu_late_model;
    embed_late_workspace_t *cpu_late_ws;
#ifdef USE_MLX
    embed_mlx_ctx_t *mlx_ctx;
    embed_mlx_late_ctx_t *mlx_late_ctx;
#endif
#ifdef USE_CUDA
    embed_cuda_ctx_t *cuda_ctx;
    embed_cuda_late_ctx_t *cuda_late_ctx;
#endif
} loaded_model;

typedef struct client client;
typedef struct job job;

typedef struct {
    aeEventLoop *loop;
    int listen_fd;
    int completion_pipe[2];
    int signal_pipe[2];
    const char *host;
    int port;
    int batch_size;
    int max_batch_tokens;
    int batch_wait_us;
    char *api_key;
    loaded_model models[MODEL_COUNT];
    int use_mlx;
    int use_cuda;
    int mlx_quantize_bits;
    int mlx_quantize_group_size;

    pthread_t worker;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int stopping;
    int worker_ready;
    int worker_init_rc;
    job *job_head;
    job *job_tail;
    job *done_head;
    job *done_tail;

    client *clients;
    int n_clients;
} http_server;

typedef struct {
    char method[8];
    char path[EMBED_HTTP_MAX_PATH + 1];
    char version[16];
    char *body;
    size_t body_len;
    char *auth;
} http_request;

struct client {
    http_server *srv;
    int fd;
    int refcount;
    int cancelled;
    int linked;
    uint64_t last_active_ms;
    sbuf in;
    sbuf out;
    size_t sent;
    int header_done;
    size_t header_len;
    size_t content_length;
    int close_after_write;
    http_request req;
    client *prev;
    client *next;
};

struct job {
    http_server *srv;
    client *c;
    char method[8];
    char path[EMBED_HTTP_MAX_PATH + 1];
    char *body;
    size_t body_len;
    char *auth;
    int status;
    char *content_type;
    char *extra_headers;
    char *response;
    size_t response_len;
    uint64_t created_ns;
    uint64_t started_ns;
    uint64_t parse_ns;
    uint64_t tokenize_ns;
    uint64_t infer_ns;
    uint64_t encode_ns;
    job *next;
};

static int queue_write(client *c);
static void dispatch_request(client *c);
static void read_cb(aeEventLoop *loop, int fd, void *clientData, int mask);

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_signal_wfd = -1;

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested)
        _exit(130);
    int saved = errno;
    g_stop_requested = 1;
    int fd = (int)g_signal_wfd;
    if (fd >= 0) {
        const unsigned char byte = 1;
        if (write(fd, &byte, 1) < 0) {
            /* best-effort stop wakeup; a failed self-pipe write is ignored */
        }
    }
    errno = saved;
}

static int set_nonblock(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0)
        return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static void http_request_free(http_request *r) {
    free(r->body);
    free(r->auth);
    memset(r, 0, sizeof(*r));
}

static void client_free(client *c) {
    if (!c)
        return;
    http_request_free(&c->req);
    sbuf_free(&c->in);
    sbuf_free(&c->out);
    free(c);
}

static void client_incref(client *c) {
    pthread_mutex_lock(&c->srv->mu);
    c->refcount++;
    pthread_mutex_unlock(&c->srv->mu);
}

static void client_decref(client *c) {
    int free_now = 0;
    pthread_mutex_lock(&c->srv->mu);
    if (--c->refcount == 0)
        free_now = 1;
    pthread_mutex_unlock(&c->srv->mu);
    if (free_now)
        client_free(c);
}

static void client_decref_n(client *c, int n) {
    int free_now = 0;
    if (n <= 0)
        return;
    pthread_mutex_lock(&c->srv->mu);
    c->refcount -= n;
    if (c->refcount <= 0)
        free_now = 1;
    pthread_mutex_unlock(&c->srv->mu);
    if (free_now)
        client_free(c);
}

static void client_unlink(client *c) {
    http_server *s = c->srv;
    if (!c->linked)
        return;
    if (c->prev)
        c->prev->next = c->next;
    else
        s->clients = c->next;
    if (c->next)
        c->next->prev = c->prev;
    c->prev = c->next = NULL;
    c->linked = 0;
    if (s->n_clients > 0)
        s->n_clients--;
}

static int close_client_unlink(client *c) {
    if (!c || c->cancelled)
        return 0;
    c->cancelled = 1;
    if (c->fd >= 0) {
        aeDeleteFileEvent(c->srv->loop, c->fd, AE_READABLE);
        aeDeleteFileEvent(c->srv->loop, c->fd, AE_WRITABLE);
        close(c->fd);
        c->fd = -1;
    }
    client_unlink(c);
    return 1;
}

static void close_client(client *c) {
    if (!close_client_unlink(c))
        return;
    client_decref(c);
}

static void append_http_response_ex(client *c,
                                    int status,
                                    const char *ctype,
                                    const char *extra_headers,
                                    const char *body,
                                    size_t body_len) {
    const char *reason = "OK";
    if (status == 204)
        reason = "No Content";
    else if (status == 400)
        reason = "Bad Request";
    else if (status == 401)
        reason = "Unauthorized";
    else if (status == 404)
        reason = "Not Found";
    else if (status == 405)
        reason = "Method Not Allowed";
    else if (status == 422)
        reason = "Unprocessable Entity";
    else if (status == 503)
        reason = "Service Unavailable";
    else if (status >= 500)
        reason = "Internal Server Error";

    sbuf_printf(&c->out,
                "HTTP/1.1 %d %s\r\n"
                "Content-Length: %zu\r\n"
                "Connection: %s\r\n",
                status, reason, body_len, c->close_after_write ? "close" : "keep-alive");
    if (ctype)
        sbuf_printf(&c->out, "Content-Type: %s\r\n", ctype);

    if (extra_headers)
        sbuf_puts(&c->out, extra_headers);
    sbuf_puts(&c->out, "\r\n");
    if (body_len)
        sbuf_append(&c->out, body, body_len);
}

static void
append_http_response(client *c, int status, const char *ctype, const char *body, size_t body_len) {
    append_http_response_ex(c, status, ctype, NULL, body, body_len);
}

static void append_json_error(sbuf *b, const char *message, const char *type) {
    sbuf_puts(b, "{\"error\":{\"message\":\"");
    for (const char *p = message ? message : ""; *p; p++) {
        if (*p == '"' || *p == '\\') {
            sbuf_putc(b, '\\');
            sbuf_putc(b, *p);
        } else {
            sbuf_putc(b, *p);
        }
    }
    sbuf_puts(b, "\",\"type\":\"");
    sbuf_puts(b, type ? type : "server_error");
    sbuf_puts(b, "\"}}");
}

static void append_json_string(sbuf *b, const char *s) {
    sbuf_putc(b, '"');
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
        case '"':
            sbuf_puts(b, "\\\"");
            break;
        case '\\':
            sbuf_puts(b, "\\\\");
            break;
        case '\b':
            sbuf_puts(b, "\\b");
            break;
        case '\f':
            sbuf_puts(b, "\\f");
            break;
        case '\n':
            sbuf_puts(b, "\\n");
            break;
        case '\r':
            sbuf_puts(b, "\\r");
            break;
        case '\t':
            sbuf_puts(b, "\\t");
            break;
        default:
            if (*p < 0x20)
                sbuf_printf(b, "\\u%04x", (unsigned)*p);
            else
                sbuf_putc(b, (char)*p);
        }
    }
    sbuf_putc(b, '"');
}

static char *json_error_body(const char *message, const char *type, size_t *len) {
    sbuf b = {0};
    append_json_error(&b, message, type);
    *len = b.len;
    return b.ptr;
}

static void job_set_error(job *j, int status, const char *message, const char *type) {
    free(j->response);
    j->status = status;
    j->content_type = "application/json";
    j->response = json_error_body(message, type, &j->response_len);
}

static bool auth_ok(http_server *s, const char *auth) {
    if (!s->api_key || !s->api_key[0])
        return true;
    if (!auth)
        return false;
    const char prefix[] = "Bearer ";
    return !strncmp(auth, prefix, sizeof(prefix) - 1) &&
           !strcmp(auth + sizeof(prefix) - 1, s->api_key);
}

static ssize_t find_header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n')
            return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n')
            return (ssize_t)(i + 1);
    }
    return -1;
}

static char *header_value_dup(const char *h, size_t n, const char *name) {
    size_t name_len = strlen(name);
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n')
            p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r')
            len--;
        if (len > name_len && !strncasecmp(line, name, name_len) && line[name_len] == ':') {
            const char *v = line + name_len + 1;
            while (v < line + len && isspace((unsigned char)*v))
                v++;
            const char *e = line + len;
            while (e > v && isspace((unsigned char)e[-1]))
                e--;
            return xstrndup(v, (size_t)(e - v));
        }
        if (p < end)
            p++;
    }
    return NULL;
}

static int header_has_token(const char *value, const char *token) {
    size_t token_len = strlen(token);
    const char *p = value;
    while (p && *p) {
        while (*p == ',' || isspace((unsigned char)*p))
            p++;
        const char *start = p;
        while (*p && *p != ',')
            p++;
        const char *end = p;
        while (end > start && isspace((unsigned char)end[-1]))
            end--;
        if ((size_t)(end - start) == token_len && !strncasecmp(start, token, token_len))
            return 1;
    }
    return 0;
}

static int parse_request_headers(client *c) {
    char line[512];
    size_t i = 0;
    while (i < c->header_len && c->in.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = c->in.ptr[i];
        i++;
    }
    line[i] = '\0';
    int fields = sscanf(line, "%7s %255s %15s", c->req.method, c->req.path, c->req.version);
    if (fields < 2)
        return -1;
    if (fields < 3)
        snprintf(c->req.version, sizeof(c->req.version), "HTTP/1.0");
    char *q = strchr(c->req.path, '?');
    if (q)
        *q = '\0';

    c->close_after_write = strcasecmp(c->req.version, "HTTP/1.1") != 0;
    char *conn = header_value_dup(c->in.ptr, c->header_len, "Connection");
    if (conn) {
        if (header_has_token(conn, "close"))
            c->close_after_write = 1;
        else if (header_has_token(conn, "keep-alive"))
            c->close_after_write = 0;
        free(conn);
    }

    char *cl = header_value_dup(c->in.ptr, c->header_len, "Content-Length");
    if (cl) {
        char *end = NULL;
        errno = 0;
        unsigned long long v = strtoull(cl, &end, 10);
        if (errno || !end || *end || v > EMBED_HTTP_MAX_BODY) {
            free(cl);
            return -1;
        }
        c->content_length = (size_t)v;
        free(cl);
    }
    c->req.auth = header_value_dup(c->in.ptr, c->header_len, "Authorization");
    return 0;
}

static int complete_request(client *c) {
    if (!c->header_done) {
        ssize_t hend = find_header_end(c->in.ptr, c->in.len);
        if (hend < 0) {
            if (c->in.len > EMBED_HTTP_MAX_HEADER)
                return -1;
            return 0;
        }
        c->header_done = 1;
        c->header_len = (size_t)hend;
        if (parse_request_headers(c) != 0)
            return -1;
    }
    if (c->content_length > EMBED_HTTP_MAX_BODY)
        return -1;
    if (c->in.len < c->header_len + c->content_length)
        return 0;
    c->req.body_len = c->content_length;
    c->req.body = xmalloc(c->content_length + 1);
    memcpy(c->req.body, c->in.ptr + c->header_len, c->content_length);
    c->req.body[c->content_length] = '\0';
    return 1;
}

static void send_bad_http_request(client *c) {
    size_t len;
    char *body = json_error_body("bad HTTP request", "invalid_request_error", &len);
    c->close_after_write = 1;
    append_http_response(c, 400, "application/json", body, len);
    free(body);
    aeDeleteFileEvent(c->srv->loop, c->fd, AE_READABLE);
    if (queue_write(c) != AE_OK)
        close_client(c);
}

static void reset_client_for_next_request(client *c) {
    size_t consumed = c->header_len + c->content_length;
    if (consumed > c->in.len)
        consumed = c->in.len;
    size_t remain = c->in.len - consumed;
    if (remain)
        memmove(c->in.ptr, c->in.ptr + consumed, remain);
    c->in.len = remain;
    if (c->in.ptr)
        c->in.ptr[remain] = '\0';

    http_request_free(&c->req);
    sbuf_clear(&c->out);
    c->sent = 0;
    c->header_done = 0;
    c->header_len = 0;
    c->content_length = 0;
    c->close_after_write = 0;
}

static void arm_next_request(client *c) {
    reset_client_for_next_request(c);
    if (c->cancelled || c->fd < 0)
        return;

    if (c->in.len) {
        int done = complete_request(c);
        if (done < 0) {
            send_bad_http_request(c);
            return;
        }
        if (done > 0) {
            dispatch_request(c);
            return;
        }
    }

    if (aeCreateFileEvent(c->srv->loop, c->fd, AE_READABLE, read_cb, c) != AE_OK)
        close_client(c);
}

static void write_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop;
    (void)mask;
    client *c = clientData;
    while (c->sent < c->out.len) {
        ssize_t n = write(fd, c->out.ptr + c->sent, c->out.len - c->sent);
        if (n > 0) {
            c->sent += (size_t)n;
            c->last_active_ms = mstime();
            continue;
        }
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return;
        close_client(c);
        return;
    }
    aeDeleteFileEvent(c->srv->loop, fd, AE_WRITABLE);
    if (c->close_after_write)
        close_client(c);
    else
        arm_next_request(c);
}

static int queue_write(client *c) {
    if (!c || c->cancelled || c->fd < 0)
        return -1;
    return aeCreateFileEvent(c->srv->loop, c->fd, AE_WRITABLE, write_cb, c);
}

static void enqueue_done(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->done_tail)
        s->done_tail->next = j;
    else
        s->done_head = j;
    s->done_tail = j;
    pthread_mutex_unlock(&s->mu);
    const unsigned char byte = 1;
    if (write(s->completion_pipe[1], &byte, 1) < 0) {
        /* best-effort wakeup; the event loop also drains the done queue */
    }
}

static job *pop_job_locked(http_server *s) {
    job *j = s->job_head;
    if (j) {
        s->job_head = j->next;
        if (!s->job_head)
            s->job_tail = NULL;
        j->next = NULL;
    }
    return j;
}

static int cond_wait_until_ns(pthread_cond_t *cv, pthread_mutex_t *mu, uint64_t deadline_ns) {
    uint64_t now_ns = nstime();
    if (now_ns >= deadline_ns)
        return ETIMEDOUT;

    uint64_t remaining_ns = deadline_ns - now_ns;
    struct timespec abs;
    if (clock_gettime(CLOCK_REALTIME, &abs) != 0) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        abs.tv_sec = tv.tv_sec;
        abs.tv_nsec = (long)tv.tv_usec * 1000L;
    }
    abs.tv_sec += (time_t)(remaining_ns / 1000000000u);
    abs.tv_nsec += (long)(remaining_ns % 1000000000u);
    if (abs.tv_nsec >= 1000000000L) {
        abs.tv_sec++;
        abs.tv_nsec -= 1000000000L;
    }
    return pthread_cond_timedwait(cv, mu, &abs);
}

static int collect_job_batch(http_server *s, job **jobs, int max_jobs) {
    if (max_jobs <= 0)
        return 0;

    pthread_mutex_lock(&s->mu);
    while (!s->job_head && !s->stopping)
        pthread_cond_wait(&s->cv, &s->mu);

    job *first = pop_job_locked(s);
    if (!first) {
        pthread_mutex_unlock(&s->mu);
        return 0;
    }

    int n = 1;
    jobs[0] = first;
    int batchable = s->batch_size > 1 && (!strcmp(first->path, "/v1/embeddings") ||
                                          !strcmp(first->path, "/v1/contextualizedembeddings"));
    if (!batchable) {
        pthread_mutex_unlock(&s->mu);
        return n;
    }

    /* Exact item and token budgets are applied after worker-side tokenization;
     * this queue stage only defines which arrival window may be grouped. */
    uint64_t wait_ns = (uint64_t)s->batch_wait_us * 1000u;
    uint64_t deadline_ns =
        first->created_ns > UINT64_MAX - wait_ns ? UINT64_MAX : first->created_ns + wait_ns;
    while (n < max_jobs) {
        while (!s->job_head && !s->stopping && wait_ns > 0) {
            int rc = cond_wait_until_ns(&s->cv, &s->mu, deadline_ns);
            if (rc == ETIMEDOUT)
                break;
        }
        if (!s->job_head)
            break;
        if (wait_ns > 0 && s->job_head->created_ns > deadline_ns)
            break;
        jobs[n++] = pop_job_locked(s);
        if (wait_ns == 0 && !s->job_head)
            break;
    }
    pthread_mutex_unlock(&s->mu);
    return n;
}

static void enqueue_job(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->job_tail)
        s->job_tail->next = j;
    else
        s->job_head = j;
    s->job_tail = j;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
}

static job *pop_done(http_server *s) {
    pthread_mutex_lock(&s->mu);
    job *j = s->done_head;
    if (j) {
        s->done_head = j->next;
        if (!s->done_head)
            s->done_tail = NULL;
        j->next = NULL;
    }
    pthread_mutex_unlock(&s->mu);
    return j;
}

static void job_free(job *j) {
    if (!j)
        return;
    free(j->body);
    free(j->auth);
    free(j->extra_headers);
    free(j->response);
    free(j);
}

static void completion_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop;
    (void)mask;
    http_server *s = clientData;
    unsigned char tmp[128];
    while (read(fd, tmp, sizeof(tmp)) > 0) {
    }

    job *j;
    while ((j = pop_done(s)) != NULL) {
        client *c = j->c;
        int refs_to_drop = 1; /* job reference */
        if (!c->cancelled) {
            append_http_response_ex(c, j->status, j->content_type, j->extra_headers,
                                    j->response ? j->response : "",
                                    j->response ? j->response_len : 0);
            if (queue_write(c) != AE_OK)
                refs_to_drop += close_client_unlink(c);
        }
        client_decref_n(c, refs_to_drop);
        job_free(j);
    }
}

static bool route_is_inference(const char *method, const char *path) {
    return !strcmp(method, "POST") &&
           (!strcmp(path, "/v1/embeddings") || !strcmp(path, "/v1/contextualizedembeddings") ||
            !strcmp(path, "/v1/rerank"));
}

static void dispatch_request(client *c) {
    aeDeleteFileEvent(c->srv->loop, c->fd, AE_READABLE);

    if (!strcmp(c->req.method, "OPTIONS")) {
        append_http_response(c, 204, NULL, "", 0);
        if (queue_write(c) != AE_OK)
            close_client(c);
        return;
    }

    if (!route_is_inference(c->req.method, c->req.path)) {
        size_t len;
        char *body = json_error_body("unknown endpoint", "invalid_request_error", &len);
        append_http_response(c, 404, "application/json", body, len);
        free(body);
        if (queue_write(c) != AE_OK)
            close_client(c);
        return;
    }

    if (!auth_ok(c->srv, c->req.auth)) {
        size_t len;
        char *body =
            json_error_body("missing or invalid bearer token", "authentication_error", &len);
        append_http_response(c, 401, "application/json", body, len);
        free(body);
        if (queue_write(c) != AE_OK)
            close_client(c);
        return;
    }

    job *j = xcalloc(1, sizeof(*j));
    j->srv = c->srv;
    j->c = c;
    snprintf(j->method, sizeof(j->method), "%s", c->req.method);
    snprintf(j->path, sizeof(j->path), "%s", c->req.path);
    j->body = xstrndup(c->req.body ? c->req.body : "", c->req.body_len);
    j->body_len = c->req.body_len;
    j->auth = c->req.auth ? xstrdup(c->req.auth) : NULL;
    j->created_ns = nstime();
    client_incref(c);
    enqueue_job(j);
}

static void read_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop;
    (void)mask;
    client *c = clientData;
    char tmp[EMBED_HTTP_IO_BUF];
    for (;;) {
        ssize_t n = read(fd, tmp, sizeof(tmp));
        if (n > 0) {
            sbuf_append(&c->in, tmp, (size_t)n);
            c->last_active_ms = mstime();
            int done = complete_request(c);
            if (done < 0) {
                send_bad_http_request(c);
                return;
            }
            if (done > 0) {
                dispatch_request(c);
                return;
            }
            continue;
        }
        if (n == 0) {
            close_client(c);
            return;
        }
        if (errno == EINTR)
            continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK)
            return;
        close_client(c);
        return;
    }
}

static void accept_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop;
    (void)mask;
    http_server *s = clientData;
    char cip[64];
    int cport = 0;
    for (;;) {
        int cfd = anetTcpAccept(NULL, fd, cip, sizeof(cip), &cport);
        if (cfd == ANET_ERR) {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return;
            if (errno == EINTR)
                continue;
            return;
        }
        set_nonblock(cfd);
        client *c = xcalloc(1, sizeof(*c));
        c->srv = s;
        c->fd = cfd;
        c->refcount = 1;
        c->last_active_ms = mstime();
        c->linked = 1;
        c->next = s->clients;
        if (s->clients)
            s->clients->prev = c;
        s->clients = c;
        s->n_clients++;
        if (aeCreateFileEvent(s->loop, cfd, AE_READABLE, read_cb, c) != AE_OK)
            close_client(c);
    }
}

static int timeout_cb(aeEventLoop *loop, long long id, void *clientData) {
    (void)loop;
    (void)id;
    http_server *s = clientData;
    uint64_t now = mstime();
    client *c = s->clients;
    while (c) {
        client *next = c->next;
        /* refcount > 1 means a request from this client is queued or being
         * processed by the worker; long batch queues (e.g. 4B long-document
         * batches) must not be reaped as idle. */
        if (!c->cancelled && c->refcount <= 1 &&
            now - c->last_active_ms > EMBED_HTTP_CLIENT_TIMEOUT_MS)
            close_client(c);
        c = next;
    }
    return EMBED_HTTP_SWEEP_MS;
}

static void signal_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)fd;
    (void)mask;
    http_server *s = clientData;
    unsigned char tmp[128];
    while (read(s->signal_pipe[0], tmp, sizeof(tmp)) > 0) {
    }
    aeStop(loop);
}

static bool ve_add(cJSON *detail, const char *loc_json, const char *msg, const char *type) {
    cJSON *obj = cJSON_CreateObject();
    cJSON *loc = cJSON_Parse(loc_json);
    if (!obj || !loc) {
        cJSON_Delete(obj);
        cJSON_Delete(loc);
        return false;
    }
    cJSON_AddItemToObject(obj, "loc", loc);
    cJSON_AddStringToObject(obj, "msg", msg ? msg : "");
    cJSON_AddStringToObject(obj, "type", type ? type : "");
    cJSON_AddItemToArray(detail, obj);
    return true;
}

static void job_set_422(job *j, cJSON *detail) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "detail", cJSON_Duplicate(detail, 1));
    char *s = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!s) {
        job_set_error(j, 500, "failed to render validation error", "server_error");
        return;
    }
    j->status = 422;
    j->content_type = "application/json";
    j->response = xstrdup(s);
    j->response_len = strlen(j->response);
    cJSON_free(s);
}

static void job_set_timing_header(job *j) {
    if (!j || j->extra_headers || !j->created_ns)
        return;
    uint64_t done_ns = nstime();
    uint64_t queue_ns = j->started_ns > j->created_ns ? j->started_ns - j->created_ns : 0;
    uint64_t worker_ns = j->started_ns && done_ns > j->started_ns ? done_ns - j->started_ns : 0;

    char buf[384];
    int n = snprintf(buf, sizeof(buf),
                     "Server-Timing: queue;dur=%.3f, parse;dur=%.3f, "
                     "tokenize;dur=%.3f, infer;dur=%.3f, encode;dur=%.3f, "
                     "worker;dur=%.3f\r\n",
                     ns_to_ms(queue_ns), ns_to_ms(j->parse_ns), ns_to_ms(j->tokenize_ns),
                     ns_to_ms(j->infer_ns), ns_to_ms(j->encode_ns), ns_to_ms(worker_ns));
    if (n > 0 && (size_t)n < sizeof(buf))
        j->extra_headers = xstrdup(buf);
}

static cJSON *parse_json_body(job *j, cJSON *detail) {
    if (!j->body || j->body_len == 0) {
        ve_add(detail, "[\"body\"]", "body required", "missing");
        return NULL;
    }
    const char *end = NULL;
    cJSON *root = cJSON_ParseWithLengthOpts(j->body, j->body_len, &end, false);
    if (!root) {
        ve_add(detail, "[\"body\"]", "invalid JSON", "json_invalid");
        return NULL;
    }
    const char *body_end = j->body + j->body_len;
    while (end && end < body_end && isspace((unsigned char)*end))
        end++;
    if (!end || end != body_end) {
        ve_add(detail, "[\"body\"]", "invalid JSON", "json_invalid");
        cJSON_Delete(root);
        return NULL;
    }
    if (!cJSON_IsObject(root)) {
        ve_add(detail, "[\"body\"]", "body must be a JSON object", "type_error.object");
        cJSON_Delete(root);
        return NULL;
    }
    return root;
}

static bool cjson_is_integer(cJSON *item) {
    if (!cJSON_IsNumber(item))
        return false;
    double d = item->valuedouble;
    int i = item->valueint;
    return d >= (double)INT_MIN && d <= (double)INT_MAX && d == (double)i;
}

static const char *encoding_from_root(cJSON *root, cJSON *detail, embedding_api_t api) {
    /* Default and accepted encodings follow the model's API family: Perplexity
     * defaults to base64_int8 and accepts {base64_int8, base64_binary, float};
     * OpenAI/DashScope (Qwen3) defaults to float and accepts {float, base64}. */
    const char *dflt = api == EMBED_API_OPENAI ? "float" : "base64_int8";
    cJSON *encoding = cJSON_GetObjectItemCaseSensitive(root, "encoding_format");
    if (!encoding)
        return dflt;
    if (!cJSON_IsString(encoding) || !encoding->valuestring) {
        ve_add(detail, "[\"body\",\"encoding_format\"]", "encoding_format must be a string",
               "type_error.string");
        return dflt;
    }
    const char *v = encoding->valuestring;
    int ok = api == EMBED_API_OPENAI ? (!strcmp(v, "float") || !strcmp(v, "base64"))
                                     : (!strcmp(v, "base64_int8") || !strcmp(v, "base64_binary") ||
                                        !strcmp(v, "float"));
    if (!ok) {
        ve_add(detail, "[\"body\",\"encoding_format\"]",
               api == EMBED_API_OPENAI
                   ? "value is not a valid enum member; permitted: 'float', 'base64'"
                   : "value is not a valid enum member; permitted: "
                     "'base64_int8', 'base64_binary', 'float'",
               "enum");
        return dflt;
    }
    return v;
}

/* Parse the `text_type` hint. Returns 1 when the model's query instruction
 * should be prepended, 0 otherwise (document / absent). Models without a
 * published query instruction reject the field. */
static int text_type_is_query(cJSON *root, cJSON *detail, const char *query_instruct) {
    cJSON *tt = cJSON_GetObjectItemCaseSensitive(root, "text_type");
    if (!tt)
        return 0;
    if (!query_instruct) {
        ve_add(detail, "[\"body\",\"text_type\"]",
               "text_type is only supported for instruction embedding models", "extra_fields");
        return 0;
    }
    if (!cJSON_IsString(tt) || !tt->valuestring) {
        ve_add(detail, "[\"body\",\"text_type\"]", "text_type must be a string",
               "type_error.string");
        return 0;
    }
    const char *v = tt->valuestring;
    if (!strcmp(v, "query"))
        return 1;
    if (!strcmp(v, "document"))
        return 0;
    ve_add(detail, "[\"body\",\"text_type\"]",
           "value is not a valid enum member; permitted: 'query', 'document'", "enum");
    return 0;
}

static int
dimensions_from_root(cJSON *root, cJSON *detail, int min_dim, int max_dim, const char *encoding) {
    int dims = max_dim;
    cJSON *dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    if (dimensions) {
        if (!cjson_is_integer(dimensions)) {
            ve_add(detail, "[\"body\",\"dimensions\"]", "dimensions must be an integer",
                   "type_error.integer");
        } else {
            dims = dimensions->valueint;
            if (dims < min_dim || dims > max_dim) {
                char msg[96];
                snprintf(msg, sizeof(msg), "dimensions must be between %d and %d", min_dim,
                         max_dim);
                ve_add(detail, "[\"body\",\"dimensions\"]", msg, "value_error.range");
            }
        }
    }
    if (!strcmp(encoding, "base64_binary") && dims % 8 != 0)
        ve_add(detail, "[\"body\",\"dimensions\"]",
               "dimensions must be divisible by 8 for base64_binary", "value_error.divisible");
    return dims;
}

static char *base64_encode(const unsigned char *src, size_t n) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t out_len = ((n + 2) / 3) * 4;
    char *out = xmalloc(out_len + 1);
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = (unsigned)src[i] << 16;
        int remain = (int)(n - i);
        if (remain > 1)
            v |= (unsigned)src[i + 1] << 8;
        if (remain > 2)
            v |= (unsigned)src[i + 2];
        out[o++] = tbl[(v >> 18) & 63];
        out[o++] = tbl[(v >> 12) & 63];
        out[o++] = remain > 1 ? tbl[(v >> 6) & 63] : '=';
        out[o++] = remain > 2 ? tbl[v & 63] : '=';
    }
    out[o] = '\0';
    return out;
}

static signed char quantize_int8_tanh(float x) {
    int v = (int)lrintf(tanhf(x) * 127.0f);
    if (v > 127)
        v = 127;
    if (v < -128)
        v = -128;
    return (signed char)v;
}

static char *encode_embedding(const float *emb, int dims, const char *encoding) {
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

typedef struct {
    int *ids;
    int n_tokens;
} token_buf;

typedef struct {
    job *j;
    cJSON *root;
    loaded_model *model;
    int dims;
    const char *encoding;
    int n_inputs;
    token_buf *tokens;
    embed_input_t *inputs;
    int total_tokens;
    int ready;
} embedding_request;

typedef struct {
    int *ids;
    int n_tokens;
    embed_span_t *spans;
    int n_spans;
} contextual_doc;

typedef struct {
    job *j;
    cJSON *root;
    loaded_model *model;
    int dims;
    const char *encoding;
    contextual_doc *docs;
    int n_docs;
    int total_chunks;
    int total_tokens;
    int ready;
} contextual_request;

typedef struct {
    int *ids;
    int n_tokens;
    int *keep;
    int n_keep;
} late_tokens;

typedef struct {
    job *j;
    cJSON *root;
    loaded_model *model;
    late_tokens query;
    late_tokens *documents;
    int n_documents;
    int top_n;
    int return_documents;
    int query_tokens;
    int document_tokens;
    int ready;
} rerank_request;

static void free_token_bufs(token_buf *t, int n) {
    if (!t)
        return;
    for (int i = 0; i < n; i++)
        free(t[i].ids);
}

static void embedding_request_free(embedding_request *r) {
    if (!r)
        return;
    free_token_bufs(r->tokens, r->n_inputs);
    free(r->tokens);
    free(r->inputs);
    if (r->root)
        cJSON_Delete(r->root);
    memset(r, 0, sizeof(*r));
}

static void contextual_request_free(contextual_request *r) {
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

static void late_tokens_free(late_tokens *t) {
    if (!t)
        return;
    free(t->ids);
    free(t->keep);
    memset(t, 0, sizeof(*t));
}

static void rerank_request_free(rerank_request *r) {
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

static int tokenize_one(loaded_model *m, job *j, const char *text, token_buf *out) {
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
    out->ids = qwen_tokenizer_encode_with_workspace(m->tok, m->tok_ws, text, &out->n_tokens);
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
static int tokenize_input(
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

static int
tokenize_late_text(loaded_model *m, job *j, const char *text, int is_query, late_tokens *out) {
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

static int model_embed_batch(loaded_model *m, const embed_input_t *inputs, int batch, float *out) {
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

static int model_embed_spans_batch(loaded_model *m,
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
static int inference_batch_accepts_input(
    const loaded_model *m, int batch, int packed_tokens, int next_tokens, int max_batch_tokens) {
    if (batch == 0)
        return 1;
    if (model_uses_dense_batches(m))
        return next_tokens <= max_batch_tokens / (batch + 1);
    return next_tokens <= max_batch_tokens - packed_tokens;
}

static void append_embedding_object(sbuf *b, int index, const char *embedding) {
    sbuf_printf(b, "{\"object\":\"embedding\",\"index\":%d,\"embedding\":\"", index);
    sbuf_puts(b, embedding);
    sbuf_puts(b, "\"}");
}

/* Emit one embedding into the response, following the model's API family.
 * "float" renders a JSON array: the true float32 vector for OpenAI/DashScope
 * (Qwen3), or the int8-decoded view (int8/128) for Perplexity (pplx). Every
 * other encoding renders a base64 string via encode_embedding(). */
static void append_embedding_value(
    sbuf *b, int index, const float *emb, int dims, const char *encoding, embedding_api_t api) {
    if (!strcmp(encoding, "float")) {
        sbuf_printf(b, "{\"object\":\"embedding\",\"index\":%d,\"embedding\":[", index);
        for (int i = 0; i < dims; i++) {
            if (i)
                sbuf_putc(b, ',');
            float v = api == EMBED_API_OPENAI ? emb[i] : (float)quantize_int8_tanh(emb[i]) / 128.0f;
            sbuf_printf(b, "%.9g", (double)v);
        }
        sbuf_puts(b, "]}");
        return;
    }
    char *encoded = encode_embedding(emb, dims, encoding);
    append_embedding_object(b, index, encoded);
    free(encoded);
}

static void set_response_from_buf(job *j, sbuf *b) {
    j->status = 200;
    j->content_type = "application/json";
    j->response = b->ptr;
    j->response_len = b->len;
    memset(b, 0, sizeof(*b));
}

static int render_embedding_response(embedding_request *r, const float *embs) {
    const int full_dim = r->model->info->dim;
    const float *render_embs = embs;
    float *truncated = NULL;
    if (r->model->renormalize_truncated && r->dims < full_dim) {
        truncated = xmalloc((size_t)r->n_inputs * r->dims * sizeof(*truncated));
        for (int i = 0; i < r->n_inputs; i++) {
            float *dst = truncated + (size_t)i * r->dims;
            memcpy(dst, embs + (size_t)i * full_dim, (size_t)r->dims * sizeof(*dst));
            if (embed_l2_normalize(dst, r->dims) != 0) {
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

static int embedding_request_compatible(const embedding_request *a, const embedding_request *b) {
    return a->ready && b->ready && a->model == b->model && a->dims == b->dims &&
           !strcmp(a->encoding, b->encoding);
}

typedef struct {
    embed_input_t input;
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

static int embedding_request_fits_group(const embedding_request *r,
                                        int group_inputs,
                                        int group_tokens,
                                        int max_batch,
                                        int max_batch_tokens) {
    if (group_inputs == 0)
        return 1;
    return r->n_inputs <= max_batch - group_inputs &&
           r->total_tokens <= max_batch_tokens - group_tokens;
}

static void execute_embedding_request_list(embedding_request **reqs, int n_reqs) {
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
    embed_input_t *inputs = xmalloc((size_t)max_batch * sizeof(*inputs));
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
        int embed_rc = model_embed_batch(m, inputs, cur, batch_embs);
        infer_ns += nstime() - t0;
        if (embed_rc != 0) {
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

static void render_contextual_response(contextual_request *r, const float *embs) {
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

static int contextual_request_compatible(const contextual_request *a, const contextual_request *b) {
    return a->ready && b->ready && a->model == b->model && a->dims == b->dims &&
           !strcmp(a->encoding, b->encoding);
}

static int contextual_request_fits_group(const contextual_request *r,
                                         int group_docs,
                                         int group_tokens,
                                         int max_batch,
                                         int max_batch_tokens) {
    if (group_docs == 0)
        return 1;
    return r->n_docs <= max_batch - group_docs &&
           r->total_tokens <= max_batch_tokens - group_tokens;
}

typedef struct {
    embed_context_input_t input;
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

static void execute_contextual_request_list(contextual_request **reqs, int n_reqs) {
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
    embed_context_input_t *inputs = xmalloc((size_t)max_batch * sizeof(*inputs));
    float *batch_embs = xmalloc(sizeof(*batch_embs));
    size_t batch_emb_cap = 0;

    int failed = 0;
    uint64_t infer_ns = 0;
    for (int start = 0; start < total_docs;) {
        int cur = 0;
        int tokens = 0;
        int chunks = 0;
        while (start + cur < total_docs && cur < max_batch) {
            const embed_context_input_t *input = &items[start + cur].input;
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
        int embed_rc = model_embed_spans_batch(m, inputs, cur, batch_embs);
        infer_ns += nstime() - t0;
        if (embed_rc != 0) {
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
    if (!model_item) {
        ve_add(detail, "[\"body\",\"model\"]", "field required", "missing");
    } else if (!cJSON_IsString(model_item) || !model_item->valuestring) {
        ve_add(detail, "[\"body\",\"model\"]", "model must be a string", "type_error.string");
    } else {
        model_id = model_item->valuestring;
        model_slot slot = model_slot_for_id(model_id);
        if (slot == MODEL_UNKNOWN || k_models[slot].kind != expected_kind) {
            ve_add(detail, "[\"body\",\"model\"]",
                   "value is not a valid enum member for this endpoint", "enum");
        } else {
            loaded_model *m = &s->models[slot];
            if (!m->info) {
                *out_model = NULL;
            } else {
                *out_model = m;
            }
        }
    }
    model_slot slot = model_id ? model_slot_for_id(model_id) : MODEL_UNKNOWN;
    embedding_api_t api = slot != MODEL_UNKNOWN ? k_models[slot].api : EMBED_API_PERPLEXITY;
    const char *enc = encoding_from_root(root, detail, api);
    int min_dim = slot != MODEL_UNKNOWN ? k_models[slot].min_dim : 128;
    int max_dim = slot != MODEL_UNKNOWN ? k_models[slot].dim : 2560;
    int dims = dimensions_from_root(root, detail, min_dim, max_dim, enc);
    *out_encoding = enc;
    *out_dims = dims;
    return model_id && model_slot_for_id(model_id) != MODEL_UNKNOWN &&
                   !s->models[model_slot_for_id(model_id)].info
               ? 503
               : 0;
}

static void prepare_embedding_request(job *j, cJSON *root, http_server *s, embedding_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root;

    cJSON *detail = cJSON_CreateArray();
    loaded_model *m = NULL;
    int dims = 0;
    const char *encoding = NULL;
    int unloaded = validate_common(root, detail, MODEL_KIND_STANDARD, &m, &dims, &encoding, s);
    out->model = m;
    out->dims = dims;
    out->encoding = encoding;

    /* Resolve request behavior from the model registry independent of load
     * state. The query instruction, when requested, applies to every input. */
    cJSON *model_item = cJSON_GetObjectItemCaseSensitive(root, "model");
    model_slot api_slot = (model_item && cJSON_IsString(model_item) && model_item->valuestring)
                              ? model_slot_for_id(model_item->valuestring)
                              : MODEL_UNKNOWN;
    const char *query_instruct =
        api_slot != MODEL_UNKNOWN &&
                text_type_is_query(root, detail, k_models[api_slot].query_instruct)
            ? k_models[api_slot].query_instruct
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
        else if (n_inputs > EMBED_API_MAX_STANDARD_INPUTS)
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

    if (cJSON_GetArraySize(detail) == 0 && unloaded) {
        job_set_error(j, 503, "requested model is valid but not loaded", "model_not_loaded");
        cJSON_Delete(detail);
        return;
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
                if (out->tokens[0].n_tokens > EMBED_API_MAX_ITEM_TOKENS)
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
                if (tokenize_input(m, j, item->valuestring, query_instruct, &out->tokens[idx]) !=
                    0) {
                    ve_add(detail, loc, "tokenization failed", "value_error.tokenization");
                } else {
                    out->inputs[idx].ids = out->tokens[idx].ids;
                    out->inputs[idx].n_tokens = out->tokens[idx].n_tokens;
                    out->total_tokens += out->tokens[idx].n_tokens;
                    if (out->tokens[idx].n_tokens > EMBED_API_MAX_ITEM_TOKENS)
                        ve_add(detail, loc, "input exceeds 32768 token limit",
                               "value_error.context_length");
                }
                idx++;
            }
        }

        if (out->total_tokens > EMBED_API_MAX_TOTAL_TOKENS)
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

static void
prepare_contextual_request(job *j, cJSON *root, http_server *s, contextual_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root;

    cJSON *detail = cJSON_CreateArray();
    loaded_model *m = NULL;
    int dims = 0;
    const char *encoding = NULL;
    int unloaded = validate_common(root, detail, MODEL_KIND_CONTEXTUAL, &m, &dims, &encoding, s);
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
        else if (n_docs > EMBED_API_MAX_CONTEXT_DOCS)
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
            if (doc_chunks > EMBED_API_MAX_CONTEXT_CHUNKS - total_chunks)
                total_chunks = EMBED_API_MAX_CONTEXT_CHUNKS + 1;
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
        if (total_chunks > EMBED_API_MAX_CONTEXT_CHUNKS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 16000 chunks",
                   "value_error.list.max_items");
    }

    if (cJSON_GetArraySize(detail) == 0 && unloaded) {
        job_set_error(j, 503, "requested model is valid but not loaded", "model_not_loaded");
        cJSON_Delete(detail);
        return;
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

            if (doc_tokens > EMBED_API_MAX_ITEM_TOKENS) {
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
                        doc->ids[pos++] = m->context_separator_id;
                    }
                }
            }
            free_token_bufs(chunk_tokens, n_chunks);
            free(chunk_tokens);
            di++;
        }
        if (request_tokens > EMBED_API_MAX_TOTAL_TOKENS) {
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
    model_slot slot = model_slot_for_id(item->valuestring);
    if (slot == MODEL_UNKNOWN || k_models[slot].kind != MODEL_KIND_LATE) {
        ve_add(detail, "[\"body\",\"model\"]", "value is not a valid enum member for this endpoint",
               "enum");
        return 0;
    }
    *out_model = s->models[slot].info ? &s->models[slot] : NULL;
    return *out_model ? 0 : 503;
}

static void prepare_rerank_request(job *j, cJSON *root, http_server *s, rerank_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root;

    cJSON *detail = cJSON_CreateArray();
    int unloaded = validate_rerank_model(root, detail, s, &out->model);
    cJSON *query = cJSON_GetObjectItemCaseSensitive(root, "query");
    cJSON *documents = cJSON_GetObjectItemCaseSensitive(root, "documents");
    if (!query) {
        ve_add(detail, "[\"body\",\"query\"]", "field required", "missing");
    } else if (!cJSON_IsString(query)) {
        ve_add(detail, "[\"body\",\"query\"]", "query must be a string", "type_error.string");
    } else if (!query->valuestring || !query->valuestring[0]) {
        ve_add(detail, "[\"body\",\"query\"]", "query must not be empty", "value_error.empty");
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
        else if (n_documents > EMBED_API_MAX_RERANK_DOCUMENTS)
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

    if (cJSON_GetArraySize(detail) == 0 && unloaded) {
        job_set_error(j, 503, "requested model is valid but not loaded", "model_not_loaded");
        cJSON_Delete(detail);
        return;
    }

    if (cJSON_GetArraySize(detail) == 0 && out->model) {
        out->n_documents = n_documents;
        out->documents = xcalloc((size_t)n_documents, sizeof(*out->documents));
        if (tokenize_late_text(out->model, j, query->valuestring, 1, &out->query) != 0) {
            ve_add(detail, "[\"body\",\"query\"]", "tokenization failed",
                   "value_error.tokenization");
        } else {
            out->query_tokens = out->query.n_tokens;
            if (out->query_tokens > EMBED_API_MAX_ITEM_TOKENS)
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
                if (n > EMBED_API_MAX_ITEM_TOKENS)
                    ve_add(detail, loc, "document exceeds 32768 token limit",
                           "value_error.context_length");
                if (out->document_tokens <= INT_MAX - n)
                    out->document_tokens += n;
                else
                    out->document_tokens = INT_MAX;
            }
            i++;
        }
        if (out->query_tokens > EMBED_API_MAX_TOTAL_TOKENS - out->document_tokens) {
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

static int execute_rerank_cpu(rerank_request *r, float *scores) {
    loaded_model *m = r->model;
    int dim = m->info->token_dim;
    int total_doc_vecs = 0;
    for (int i = 0; i < r->n_documents; i++)
        total_doc_vecs += r->documents[i].n_keep;

    float *query = xmalloc((size_t)r->query.n_tokens * dim * sizeof(*query));
    float *docs = xmalloc((size_t)total_doc_vecs * dim * sizeof(*docs));
    int *offsets = xmalloc((size_t)(r->n_documents + 1) * sizeof(*offsets));

    /* Encode every candidate in one block-diagonal forward. The encoder packs
     * each document's kept token vectors back-to-back and fills offsets, so the
     * cost of N short documents collapses to a single packed forward. */
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

    int rc = embed_late_model_encode_tokens(m->cpu_late_model, m->cpu_late_ws, r->query.ids,
                                            r->query.n_tokens, 1, query);
    if (rc == 0)
        rc = embed_late_model_encode_docs(m->cpu_late_model, m->cpu_late_ws, doc_ids, n_tokens,
                                          keep, n_keep, r->n_documents, 1, docs, offsets);
    if (rc == 0)
        rc = embed_late_maxsim_batch(query, r->query.n_keep, docs, offsets, r->n_documents, dim,
                                     scores);
    free(n_keep);
    free(n_tokens);
    free(keep);
    free(doc_ids);
    free(offsets);
    free(docs);
    free(query);
    return rc;
}

#ifdef USE_MLX
static int execute_rerank_mlx(rerank_request *r, float *scores) {
    loaded_model *m = r->model;
    embed_mlx_late_vectors_t *query =
        embed_mlx_late_encode_tokens_device(m->mlx_late_ctx, r->query.ids, r->query.n_tokens, 1);

    /* Encode every candidate in one padded transformer pass. The encoder packs
     * each document's kept token vectors back-to-back and fills offsets, so the
     * cost of N short documents collapses to a single batched forward. */
    const int **doc_ids = xmalloc((size_t)r->n_documents * sizeof(*doc_ids));
    const int **keep = xmalloc((size_t)r->n_documents * sizeof(*keep));
    int *n_tokens = xmalloc((size_t)r->n_documents * sizeof(*n_tokens));
    int *n_keep = xmalloc((size_t)r->n_documents * sizeof(*n_keep));
    int *offsets = xmalloc((size_t)(r->n_documents + 1) * sizeof(*offsets));
    for (int i = 0; i < r->n_documents; i++) {
        doc_ids[i] = r->documents[i].ids;
        keep[i] = r->documents[i].keep;
        n_tokens[i] = r->documents[i].n_tokens;
        n_keep[i] = r->documents[i].n_keep;
    }
    embed_mlx_late_vectors_t *packed =
        query ? embed_mlx_late_encode_docs_device(m->mlx_late_ctx, doc_ids, n_tokens, keep, n_keep,
                                                  r->n_documents, 1, offsets)
              : NULL;
    int rc = packed ? embed_mlx_late_maxsim_batch_device(m->mlx_late_ctx, query, packed, offsets,
                                                         r->n_documents, scores)
                    : -1;
    embed_mlx_late_vectors_free(packed);
    embed_mlx_late_vectors_free(query);
    free(offsets);
    free(n_keep);
    free(n_tokens);
    free(keep);
    free(doc_ids);
    return rc;
}
#endif

#ifdef USE_CUDA
static int execute_rerank_cuda(rerank_request *r, float *scores) {
    loaded_model *m = r->model;
    int dim = m->info->token_dim;
    int total_doc_vecs = 0;
    for (int i = 0; i < r->n_documents; i++)
        total_doc_vecs += r->documents[i].n_keep;

    float *query = xmalloc((size_t)r->query.n_tokens * dim * sizeof(*query));
    float *docs = xmalloc((size_t)total_doc_vecs * dim * sizeof(*docs));
    int *offsets = xmalloc((size_t)(r->n_documents + 1) * sizeof(*offsets));

    /* GPU encodes the query and every candidate (one packed forward); MaxSim
     * runs on the host, where the grouped-GEMM scorer beats a device graph. */
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

    int rc =
        embed_cuda_late_encode_tokens(m->cuda_late_ctx, r->query.ids, r->query.n_tokens, 1, query);
    if (rc == 0)
        rc = embed_cuda_late_encode_docs(m->cuda_late_ctx, doc_ids, n_tokens, keep, n_keep,
                                         r->n_documents, 1, docs, offsets);
    if (rc == 0)
        rc = embed_late_maxsim_batch(query, r->query.n_keep, docs, offsets, r->n_documents, dim,
                                     scores);
    free(n_keep);
    free(n_tokens);
    free(keep);
    free(doc_ids);
    free(offsets);
    free(docs);
    free(query);
    return rc;
}
#endif

static void execute_rerank_request(rerank_request *r) {
    if (!r || !r->model) {
        if (r && r->j)
            job_set_error(r->j, 500, "late-interaction model is unavailable", "server_error");
        return;
    }
    float *scores = xmalloc((size_t)r->n_documents * sizeof(*scores));
    uint64_t t0 = nstime();
#if defined(USE_MLX)
    int rc = r->model->mlx_late_ctx ? execute_rerank_mlx(r, scores) : execute_rerank_cpu(r, scores);
#elif defined(USE_CUDA)
    int rc =
        r->model->cuda_late_ctx ? execute_rerank_cuda(r, scores) : execute_rerank_cpu(r, scores);
#else
    int rc = execute_rerank_cpu(r, scores);
#endif
    r->j->infer_ns += nstime() - t0;
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

static int parse_job_root(job *j, cJSON **out_root) {
    cJSON *detail = cJSON_CreateArray();
    cJSON *root = parse_json_body(j, detail);
    if (!root) {
        job_set_422(j, detail);
        cJSON_Delete(detail);
        *out_root = NULL;
        return -1;
    }
    cJSON_Delete(detail);
    *out_root = root;
    return 0;
}

static void process_unknown_job(job *j) {
    cJSON *root = NULL;
    if (!j->started_ns)
        j->started_ns = nstime();
    uint64_t t0 = nstime();
    int rc = parse_job_root(j, &root);
    j->parse_ns += nstime() - t0;
    if (rc != 0)
        return;
    job_set_error(j, 404, "unknown endpoint", "invalid_request_error");
    cJSON_Delete(root);
}

static void process_job_group(http_server *s, job **jobs, int n_jobs) {
    embedding_request *std_reqs = xcalloc((size_t)n_jobs, sizeof(*std_reqs));
    contextual_request *ctx_reqs = xcalloc((size_t)n_jobs, sizeof(*ctx_reqs));
    rerank_request *rerank_reqs = xcalloc((size_t)n_jobs, sizeof(*rerank_reqs));
    int *kind = xcalloc((size_t)n_jobs, sizeof(*kind));
    int *done = xcalloc((size_t)n_jobs, sizeof(*done));
    embedding_request **std_group = xmalloc((size_t)n_jobs * sizeof(*std_group));
    contextual_request **ctx_group = xmalloc((size_t)n_jobs * sizeof(*ctx_group));

    for (int i = 0; i < n_jobs; i++) {
        cJSON *root = NULL;
        if (!jobs[i]->started_ns)
            jobs[i]->started_ns = nstime();
        if (!strcmp(jobs[i]->path, "/v1/embeddings")) {
            kind[i] = 1;
        } else if (!strcmp(jobs[i]->path, "/v1/contextualizedembeddings")) {
            kind[i] = 2;
        } else if (!strcmp(jobs[i]->path, "/v1/rerank")) {
            kind[i] = 3;
        } else {
            continue;
        }
        uint64_t t0 = nstime();
        int parse_rc = parse_job_root(jobs[i], &root);
        jobs[i]->parse_ns += nstime() - t0;
        if (parse_rc != 0)
            continue;
        if (kind[i] == 1)
            prepare_embedding_request(jobs[i], root, s, &std_reqs[i]);
        else if (kind[i] == 2)
            prepare_contextual_request(jobs[i], root, s, &ctx_reqs[i]);
        else
            prepare_rerank_request(jobs[i], root, s, &rerank_reqs[i]);
    }

    for (int i = 0; i < n_jobs; i++) {
        if (done[i])
            continue;

        if (!kind[i]) {
            process_unknown_job(jobs[i]);
            job_set_timing_header(jobs[i]);
            enqueue_done(jobs[i]);
            done[i] = 1;
            continue;
        }

        if ((kind[i] == 1 && !std_reqs[i].ready) || (kind[i] == 2 && !ctx_reqs[i].ready) ||
            (kind[i] == 3 && !rerank_reqs[i].ready)) {
            job_set_timing_header(jobs[i]);
            enqueue_done(jobs[i]);
            done[i] = 1;
            continue;
        }

        if (kind[i] == 3) {
            execute_rerank_request(&rerank_reqs[i]);
            job_set_timing_header(jobs[i]);
            enqueue_done(jobs[i]);
            done[i] = 1;
        } else if (kind[i] == 1) {
            int group_n = 1;
            int group_inputs = std_reqs[i].n_inputs;
            int group_tokens = std_reqs[i].total_tokens;
            std_group[0] = &std_reqs[i];
            for (int k = i + 1; k < n_jobs; k++) {
                if (!done[k] && kind[k] == 1 && std_reqs[k].ready &&
                    embedding_request_compatible(&std_reqs[i], &std_reqs[k]) &&
                    embedding_request_fits_group(&std_reqs[k], group_inputs, group_tokens,
                                                 s->batch_size, s->max_batch_tokens)) {
                    std_group[group_n++] = &std_reqs[k];
                    group_inputs += std_reqs[k].n_inputs;
                    group_tokens += std_reqs[k].total_tokens;
                }
            }
            execute_embedding_request_list(std_group, group_n);
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(std_group[k] - std_reqs);
                job_set_timing_header(jobs[idx]);
                enqueue_done(jobs[idx]);
                done[idx] = 1;
            }
        } else {
            int group_n = 1;
            int group_docs = ctx_reqs[i].n_docs;
            int group_tokens = ctx_reqs[i].total_tokens;
            ctx_group[0] = &ctx_reqs[i];
            for (int k = i + 1; k < n_jobs; k++) {
                if (!done[k] && kind[k] == 2 && ctx_reqs[k].ready &&
                    contextual_request_compatible(&ctx_reqs[i], &ctx_reqs[k]) &&
                    contextual_request_fits_group(&ctx_reqs[k], group_docs, group_tokens,
                                                  s->batch_size, s->max_batch_tokens)) {
                    ctx_group[group_n++] = &ctx_reqs[k];
                    group_docs += ctx_reqs[k].n_docs;
                    group_tokens += ctx_reqs[k].total_tokens;
                }
            }
            execute_contextual_request_list(ctx_group, group_n);
            for (int k = 0; k < group_n; k++) {
                int idx = (int)(ctx_group[k] - ctx_reqs);
                job_set_timing_header(jobs[idx]);
                enqueue_done(jobs[idx]);
                done[idx] = 1;
            }
        }
    }

    for (int i = 0; i < n_jobs; i++) {
        embedding_request_free(&std_reqs[i]);
        contextual_request_free(&ctx_reqs[i]);
        rerank_request_free(&rerank_reqs[i]);
    }
    free(ctx_group);
    free(std_group);
    free(done);
    free(kind);
    free(rerank_reqs);
    free(ctx_reqs);
    free(std_reqs);
}

static int configure_loaded_model(loaded_model *m, const embed_config_t *config, int token_dim) {
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

static void *worker_main(void *arg) {
    http_server *s = arg;
#ifdef USE_MLX
    if (s->use_mlx) {
        int rc = 0;
        for (int i = 0; i < MODEL_COUNT; i++) {
            loaded_model *m = &s->models[i];
            if (!m->info)
                continue;
            embed_mlx_options_t mlx_opts = {
                .quantize_bits = s->mlx_quantize_bits,
                .quantize_group_size = s->mlx_quantize_group_size,
            };
            if (m->info->kind == MODEL_KIND_LATE)
                m->mlx_late_ctx = embed_mlx_late_load_with_options(m->path, &mlx_opts);
            else
                m->mlx_ctx = embed_mlx_load_with_options(m->path, &mlx_opts);
            if ((!m->mlx_ctx && m->info->kind != MODEL_KIND_LATE) ||
                (!m->mlx_late_ctx && m->info->kind == MODEL_KIND_LATE)) {
                server_log("embed-server: failed to load MLX model on worker: %s", m->path);
                rc = 1;
                break;
            }
            const embed_config_t *config = m->info->kind == MODEL_KIND_LATE
                                               ? embed_mlx_late_config(m->mlx_late_ctx)
                                               : embed_mlx_config(m->mlx_ctx);
            int token_dim =
                m->info->kind == MODEL_KIND_LATE ? embed_mlx_late_token_dim(m->mlx_late_ctx) : 0;
            if (configure_loaded_model(m, config, token_dim) != 0 ||
                (m->info->kind == MODEL_KIND_LATE &&
                 (m->late_mask_id >= config->vocab_size ||
                  m->late_query_prefix_id >= config->vocab_size ||
                  m->late_document_prefix_id >= config->vocab_size))) {
                rc = 1;
                break;
            }
        }
        pthread_mutex_lock(&s->mu);
        s->worker_init_rc = rc;
        s->worker_ready = 1;
        if (rc)
            s->stopping = 1;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
        if (rc) {
            for (int i = 0; i < MODEL_COUNT; i++) {
                loaded_model *m = &s->models[i];
                if (m->mlx_ctx) {
                    embed_mlx_free(m->mlx_ctx);
                    m->mlx_ctx = NULL;
                }
                if (m->mlx_late_ctx) {
                    embed_mlx_late_free(m->mlx_late_ctx);
                    m->mlx_late_ctx = NULL;
                }
            }
            return NULL;
        }
    } else
#endif
#ifdef USE_CUDA
        if (s->use_cuda) {
        int rc = 0;
        for (int i = 0; i < MODEL_COUNT; i++) {
            loaded_model *m = &s->models[i];
            if (!m->info)
                continue;
            if (m->info->kind == MODEL_KIND_LATE)
                m->cuda_late_ctx = embed_cuda_late_load(m->path);
            else
                m->cuda_ctx = embed_cuda_load(m->path);
            if ((!m->cuda_ctx && m->info->kind != MODEL_KIND_LATE) ||
                (!m->cuda_late_ctx && m->info->kind == MODEL_KIND_LATE)) {
                server_log("embed-server: failed to load CUDA model on worker: %s", m->path);
                rc = 1;
                break;
            }
            const embed_config_t *config = m->info->kind == MODEL_KIND_LATE
                                               ? embed_cuda_late_config(m->cuda_late_ctx)
                                               : embed_cuda_config(m->cuda_ctx);
            int token_dim =
                m->info->kind == MODEL_KIND_LATE ? embed_cuda_late_token_dim(m->cuda_late_ctx) : 0;
            if (configure_loaded_model(m, config, token_dim) != 0 ||
                (m->info->kind == MODEL_KIND_LATE &&
                 (m->late_mask_id >= config->vocab_size ||
                  m->late_query_prefix_id >= config->vocab_size ||
                  m->late_document_prefix_id >= config->vocab_size))) {
                rc = 1;
                break;
            }
        }
        pthread_mutex_lock(&s->mu);
        s->worker_init_rc = rc;
        s->worker_ready = 1;
        if (rc)
            s->stopping = 1;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
        if (rc) {
            for (int i = 0; i < MODEL_COUNT; i++) {
                loaded_model *m = &s->models[i];
                if (m->cuda_ctx) {
                    embed_cuda_free(m->cuda_ctx);
                    m->cuda_ctx = NULL;
                }
                if (m->cuda_late_ctx) {
                    embed_cuda_late_free(m->cuda_late_ctx);
                    m->cuda_late_ctx = NULL;
                }
            }
            return NULL;
        }
    } else
#endif
    {
        pthread_mutex_lock(&s->mu);
        s->worker_ready = 1;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
    }

    for (;;) {
        job *jobs[EMBED_SERVER_MICROBATCH_MAX_JOBS];
        int n_jobs = collect_job_batch(s, jobs, EMBED_SERVER_MICROBATCH_MAX_JOBS);
        if (n_jobs == 0)
            break;
        process_job_group(s, jobs, n_jobs);
    }
#ifdef USE_MLX
    if (s->use_mlx) {
        for (int i = 0; i < MODEL_COUNT; i++) {
            loaded_model *m = &s->models[i];
            if (m->mlx_ctx) {
                embed_mlx_free(m->mlx_ctx);
                m->mlx_ctx = NULL;
            }
            if (m->mlx_late_ctx) {
                embed_mlx_late_free(m->mlx_late_ctx);
                m->mlx_late_ctx = NULL;
            }
        }
    }
#endif
#ifdef USE_CUDA
    if (s->use_cuda) {
        for (int i = 0; i < MODEL_COUNT; i++) {
            loaded_model *m = &s->models[i];
            if (m->cuda_ctx) {
                embed_cuda_free(m->cuda_ctx);
                m->cuda_ctx = NULL;
            }
            if (m->cuda_late_ctx) {
                embed_cuda_late_free(m->cuda_late_ctx);
                m->cuda_late_ctx = NULL;
            }
        }
    }
#endif
    return NULL;
}

static int listen_on(const char *host, int port) {
    char portbuf[32];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    char err[ANET_ERR_LEN];
    int fd = anetTcpServer(err, port, (char *)host, EMBED_HTTP_BACKLOG);
    (void)portbuf;
    if (fd == ANET_ERR) {
        server_log("embed-server: listen failed on %s:%d: %s", host, port, err);
        return -1;
    }
    set_nonblock(fd);
    return fd;
}

static int load_one_model(http_server *s, model_slot slot, const char *path) {
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
        m->tok = qwen_tokenizer_load(vocab_path);
        if (!m->tok) {
            server_log("embed-server: failed to load tokenizer: %s", vocab_path);
            return -1;
        }
        m->tok_ws = qwen_tokenizer_workspace_new();
        if (!m->tok_ws) {
            server_log("embed-server: failed to allocate tokenizer workspace: %s", path);
            return -1;
        }
        /* The contextual chunk separator is the tokenizer's <|endoftext|>. The
         * released models keep added special tokens out of vocab.json, so the
         * family constant is the normal case; a vocab that does define the
         * token (e.g. small test models) takes precedence so the id stays
         * inside that model's embedding table. */
        int sep_id = qwen_tokenizer_token_id(m->tok, "<|endoftext|>");
        m->context_separator_id = sep_id >= 0 ? sep_id : EMBED_CONTEXT_SEPARATOR_TOKEN_ID;
        /* Qwen3-Embedding pools the last token, the tokenizer's <|endoftext|>
         * suffix - the same token as the separator, not the model's chat
         * eos_token_id (<|im_end|>). Resolve it here from the tokenizer. */
        m->terminal_token_id = m->context_separator_id;
        if (m->info->kind == MODEL_KIND_LATE) {
            int id = qwen_tokenizer_token_id(m->tok, "[MASK]");
            m->late_mask_id = id >= 0 ? id : EMBED_LATE_MASK_TOKEN_ID;
            id = qwen_tokenizer_token_id(m->tok, "[Q]");
            m->late_query_prefix_id = id >= 0 ? id : EMBED_LATE_QUERY_PREFIX_ID;
            id = qwen_tokenizer_token_id(m->tok, "[D]");
            m->late_document_prefix_id = id >= 0 ? id : EMBED_LATE_DOCUMENT_PREFIX_ID;

            const char *punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
            for (const char *p = punct;
                 *p &&
                 m->n_late_skip_ids < (int)(sizeof(m->late_skip_ids) / sizeof(m->late_skip_ids[0]));
                 p++) {
                char text[2] = {*p, '\0'};
                int n_ids = 0;
                int *ids = qwen_tokenizer_encode(m->tok, text, &n_ids);
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

static void free_models(http_server *s) {
    for (int i = 0; i < MODEL_COUNT; i++) {
        loaded_model *m = &s->models[i];
        free(m->path);
        if (m->tok_ws)
            qwen_tokenizer_workspace_free(m->tok_ws);
        if (m->tok)
            qwen_tokenizer_free(m->tok);
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

static uint64_t physical_memory_nbytes(void) {
#ifdef __APPLE__
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) == 0 && len == sizeof(bytes) && bytes > 0)
        return bytes;
#endif
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0 || (uint64_t)pages > UINT64_MAX / (uint64_t)page_size)
        return 0;
    return (uint64_t)pages * (uint64_t)page_size;
}

static int validate_model_specs(const embed_server_config_t *cfg) {
    int seen[MODEL_COUNT] = {0};
    for (int i = 0; i < cfg->n_models; i++) {
        const embed_server_model_spec_t *spec = &cfg->models[i];
        model_slot slot = model_slot_for_id(spec->id);
        if (slot == MODEL_UNKNOWN) {
            fprintf(stderr, "embed-server: unknown model id: %s\n", spec->id ? spec->id : "<null>");
            return -1;
        }
        if (!spec->path || !spec->path[0]) {
            fprintf(stderr, "embed-server: model %s has an empty path\n", spec->id);
            return -1;
        }
        if (seen[slot]) {
            fprintf(stderr, "embed-server: duplicate model id: %s\n", spec->id);
            return -1;
        }
        seen[slot] = 1;
    }
    return 0;
}

static int mlx_memory_preflight(const embed_server_config_t *cfg) {
    if (!cfg->use_mlx)
        return 0;

    uint64_t payload = 0;
    for (int i = 0; i < cfg->n_models; i++) {
        multi_safetensors_t *ms = multi_safetensors_open(cfg->models[i].path);
        size_t model_payload = 0;
        if (!ms || multi_safetensors_data_nbytes(ms, &model_payload) != 0) {
            multi_safetensors_close(ms);
            fprintf(stderr, "embed-server: failed to inspect MLX model: %s\n", cfg->models[i].path);
            return -1;
        }
        multi_safetensors_close(ms);
        if ((uint64_t)model_payload > UINT64_MAX - payload) {
            fprintf(stderr, "embed-server: MLX model payload size overflow\n");
            return -1;
        }
        payload += (uint64_t)model_payload;
    }

    if (payload > UINT64_MAX / EMBED_MLX_RESIDENT_MULTIPLIER) {
        fprintf(stderr, "embed-server: MLX resident-memory estimate overflow\n");
        return -1;
    }
    uint64_t estimated = payload * EMBED_MLX_RESIDENT_MULTIPLIER;
    uint64_t physical = physical_memory_nbytes();
    if (physical == 0) {
        server_log("embed-server: warning: could not determine physical memory; "
                   "skipping MLX memory preflight");
        return 0;
    }
    double utilization = cfg->memory_utilization > 0.0 ? cfg->memory_utilization
                                                       : EMBED_MLX_MEMORY_BUDGET_PERCENT / 100.0;
    uint64_t budget = (uint64_t)((double)physical * utilization);
    const double gib = 1024.0 * 1024.0 * 1024.0;
    server_log("embed-server: MLX memory preflight: %.1f GiB tensors, "
               "%.1f GiB estimated resident, %.1f GiB budget "
               "(%.2f of physical memory)",
               (double)payload / gib, (double)estimated / gib, (double)budget / gib, utilization);
    if (cfg->mlx_quantize_bits) {
        server_log("embed-server: MLX %d-bit quantization enabled; preflight "
                   "uses source payload as a conservative peak estimate",
                   cfg->mlx_quantize_bits);
    }
    if (estimated <= budget)
        return 0;
    server_log("embed-server: refusing MLX model set above the host-memory "
               "budget");
    server_log("embed-server: use BF16 artifacts, load fewer models, or raise "
               "--memory-utilization (values above 1.0 overcommit)");
    return -1;
}

int embed_run_server(const embed_server_config_t *cfg) {
    if (!cfg || !cfg->models || cfg->n_models <= 0) {
        fprintf(stderr, "embed-server: at least one --model MODEL_ID=PATH is required\n");
        return 1;
    }
    if (validate_model_specs(cfg) != 0 || mlx_memory_preflight(cfg) != 0)
        return 1;

    http_server s;
    memset(&s, 0, sizeof(s));
    s.host = cfg->host && cfg->host[0] ? cfg->host : "127.0.0.1";
    s.port = cfg->port > 0 ? cfg->port : 8000;
    s.batch_size = cfg->batch_size > 0 ? cfg->batch_size : EMBED_SERVER_DEFAULT_BATCH_SIZE;
    s.max_batch_tokens =
        cfg->max_batch_tokens > 0 ? cfg->max_batch_tokens : EMBED_SERVER_DEFAULT_MAX_BATCH_TOKENS;
    s.batch_wait_us = cfg->batch_wait_us >= 0 ? cfg->batch_wait_us
                                              : (cfg->use_cuda ? EMBED_SERVER_CUDA_BATCH_WAIT_US
                                                               : EMBED_SERVER_BATCH_WAIT_US);
    s.use_mlx = cfg->use_mlx;
    s.use_cuda = cfg->use_cuda;
    s.mlx_quantize_bits = cfg->mlx_quantize_bits;
    s.mlx_quantize_group_size =
        cfg->mlx_quantize_group_size > 0 ? cfg->mlx_quantize_group_size : 64;
    if (cfg->api_key && cfg->api_key[0])
        s.api_key = xstrdup(cfg->api_key);
    else {
        const char *env_key = getenv("EMBED_API_KEY");
        if (env_key && env_key[0])
            s.api_key = xstrdup(env_key);
    }

    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);

    for (int i = 0; i < cfg->n_models; i++) {
        model_slot slot = model_slot_for_id(cfg->models[i].id);
        if (slot == MODEL_UNKNOWN) {
            fprintf(stderr, "embed-server: unknown model id: %s\n", cfg->models[i].id);
            free_models(&s);
            return 1;
        }
        if (load_one_model(&s, slot, cfg->models[i].path) != 0) {
            free_models(&s);
            return 1;
        }
    }

    if (pipe(s.completion_pipe) != 0 || pipe(s.signal_pipe) != 0) {
        perror("pipe");
        free_models(&s);
        return 1;
    }
    set_nonblock(s.completion_pipe[0]);
    set_nonblock(s.completion_pipe[1]);
    set_nonblock(s.signal_pipe[0]);
    set_nonblock(s.signal_pipe[1]);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
    g_signal_wfd = s.signal_pipe[1];

    s.loop = aeCreateEventLoop(EMBED_HTTP_SETSIZE);
    if (!s.loop) {
        fprintf(stderr, "embed-server: failed to create event loop\n");
        free_models(&s);
        return 1;
    }

    if (pthread_create(&s.worker, NULL, worker_main, &s) != 0) {
        fprintf(stderr, "embed-server: failed to start worker\n");
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        return 1;
    }
    pthread_mutex_lock(&s.mu);
    while (!s.worker_ready)
        pthread_cond_wait(&s.cv, &s.mu);
    int worker_init_rc = s.worker_init_rc;
    pthread_mutex_unlock(&s.mu);
    if (worker_init_rc) {
        pthread_join(s.worker, NULL);
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
        return 1;
    }

    s.listen_fd = listen_on(s.host, s.port);
    if (s.listen_fd < 0) {
        pthread_mutex_lock(&s.mu);
        s.stopping = 1;
        pthread_cond_signal(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.worker, NULL);
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        return 1;
    }

    aeCreateFileEvent(s.loop, s.listen_fd, AE_READABLE, accept_cb, &s);
    aeCreateFileEvent(s.loop, s.completion_pipe[0], AE_READABLE, completion_cb, &s);
    aeCreateFileEvent(s.loop, s.signal_pipe[0], AE_READABLE, signal_cb, &s);
    aeCreateTimeEvent(s.loop, EMBED_HTTP_SWEEP_MS, timeout_cb, &s, NULL);

    server_log("embed-server: listening on http://%s:%d", s.host, s.port);
    aeMain(s.loop);

    server_log("embed-server: shutdown requested");
    if (s.listen_fd >= 0)
        close(s.listen_fd);
    client *c = s.clients;
    while (c) {
        client *next = c->next;
        close_client(c);
        c = next;
    }

    pthread_mutex_lock(&s.mu);
    s.stopping = 1;
    pthread_cond_signal(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(s.worker, NULL);

    aeDeleteEventLoop(s.loop);
    close(s.completion_pipe[0]);
    close(s.completion_pipe[1]);
    close(s.signal_pipe[0]);
    close(s.signal_pipe[1]);
    free_models(&s);
    free(s.api_key);
    pthread_cond_destroy(&s.cv);
    pthread_mutex_destroy(&s.mu);
    return 0;
}

/* ====================================================================
 * embed-server entry point
 * ==================================================================== */
#ifndef EMBED_SERVER_TEST

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --model ID=DIR [options]\n"
            "\n"
            "Options:\n"
            "  --model ID=DIR            Model to serve (repeatable)\n"
#ifdef USE_MLX
            "  --mlx                     Use Apple MLX GPU backend\n"
#endif
#ifdef USE_CUDA
            "  --cuda                    Use NVIDIA CUDA GPU backend\n"
#endif
            "  --host HOST               Bind host (default: 127.0.0.1)\n"
            "  --port N                  Bind port (default: 8000)\n"
            "  --api-key K               Require Authorization: Bearer K\n"
            "  -b, --batch-size N        Max texts or documents per inference batch\n"
            "                            (default: 32)\n"
            "  --max-batch-tokens N      Max tokens per inference batch (default: 16384)\n"
            "  --batch-wait-us N         First-arrival micro-batch deadline in us\n"
#ifdef USE_CUDA
            "                            (default: CUDA 1000; CPU 0)\n"
#else
            "                            (default: 0)\n"
#endif
#ifdef USE_MLX
            "  --memory-utilization F    Fraction of physical memory the MLX model-set\n"
            "                            preflight may plan for (default: 0.90; values\n"
            "                            above 1.0 overcommit)\n"
            "  --mlx-quant-bits N        Quantize MLX linear weights to 8 bits at load\n"
            "  --mlx-quant-group-size N  MLX quantization group size (default: 64)\n"
#endif
#ifdef USE_CUDA
            "  --cuda-gemm-mode MODE     CUDA GEMM compute: f32, tf32, bf16, or 16f\n"
            "                            (default: f32, exact)\n"
            "  --cuda-weight-dtype DTYPE CUDA weight storage: f32 or bf16 (default:\n"
            "                            f32); bf16 halves weight memory\n"
#endif
            "  -t, --threads N           CPU threads (default: available cores, cgroup-aware)\n"
            "  -v, --verbose             Verbose (-vv for debug)\n"
            "  -V, --version             Print version and exit\n"
            "  --build-info              Print build details and exit\n"
            "  -h, --help                Show this help\n"
            "\n"
            "Examples:\n"
            "  %s --model pplx-embed-v1-0.6b=./model --port 8000\n"
#ifdef USE_MLX
            "  %s --mlx --mlx-quant-bits 8 \\\n"
            "      --model pplx-embed-v1-4b=./model-4b-bf16\n",
            prog, prog, prog);
#else
            ,
            prog, prog);
#endif
}

typedef struct {
    embed_server_model_spec_t *v;
    int n;
    int cap;
} model_specs_t;

static int append_model_spec(model_specs_t *specs, const char *arg) {
    const char *eq = strchr(arg, '=');
    if (!eq || eq == arg || !eq[1]) {
        fprintf(stderr, "--model expects MODEL_ID=DIR\n");
        return -1;
    }
    if (specs->n == specs->cap) {
        int new_cap = specs->cap ? specs->cap * 2 : 4;
        void *p = realloc(specs->v, (size_t)new_cap * sizeof(specs->v[0]));
        if (!p)
            return -1;
        specs->v = p;
        specs->cap = new_cap;
    }
    size_t id_len = (size_t)(eq - arg);
    char *id = (char *)malloc(id_len + 1);
    char *path = strdup(eq + 1);
    if (id) {
        memcpy(id, arg, id_len);
        id[id_len] = '\0';
    }
    specs->v[specs->n].id = id;
    specs->v[specs->n].path = path;
    if (!specs->v[specs->n].id || !specs->v[specs->n].path)
        return -1;
    specs->n++;
    return 0;
}

static void free_model_specs(model_specs_t *specs) {
    if (!specs)
        return;
    for (int i = 0; i < specs->n; i++) {
        free((char *)specs->v[i].id);
        free((char *)specs->v[i].path);
    }
    free(specs->v);
}

int main(int argc, char *argv[]) {
    int n_threads = 0;
    int verbose = 0;
    int batch_size = 0;
    int max_batch_tokens = 0;
    int batch_wait_us = -1;
#ifdef USE_MLX
    int use_mlx = 0;
    int mlx_quantize_bits = 0;
    int mlx_quantize_group_size = 64;
    double memory_utilization = 0.0;
#endif
#ifdef USE_CUDA
    int use_cuda = 0;
    const char *cuda_fast_gemm = NULL;
    const char *cuda_weights = NULL;
#endif
    const char *host = "127.0.0.1";
    const char *api_key = NULL;
    int port = 8000;
    model_specs_t model_specs = {0};

    const char *prog = embed_prog_name(argv[0]);
    int arg = 1;
    while (arg < argc && argv[arg][0] == '-') {
        const char *f = argv[arg];
        if (!strcmp(f, "-V") || !strcmp(f, "--version")) {
            embed_print_version(prog);
            free_model_specs(&model_specs);
            return 0;
        } else if (!strcmp(f, "--build-info")) {
            embed_print_build_info(prog);
            free_model_specs(&model_specs);
            return 0;
        } else if (!strcmp(f, "--model")) {
            if (append_model_spec(&model_specs, argv[++arg]) != 0) {
                free_model_specs(&model_specs);
                return 1;
            }
        }
#ifdef USE_MLX
        else if (!strcmp(f, "--mlx")) {
            use_mlx = 1;
        } else if (!strcmp(f, "--mlx-quant-bits")) {
            mlx_quantize_bits = atoi(argv[++arg]);
            if (mlx_quantize_bits != 0 && mlx_quantize_bits != 8) {
                fprintf(stderr, "--mlx-quant-bits must be 0 or 8\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--mlx-quant-group-size")) {
            mlx_quantize_group_size = atoi(argv[++arg]);
            if (mlx_quantize_group_size <= 0) {
                fprintf(stderr, "--mlx-quant-group-size must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--memory-utilization")) {
            memory_utilization = atof(argv[++arg]);
            if (memory_utilization <= 0.0) {
                fprintf(stderr, "--memory-utilization must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
#endif
#ifdef USE_CUDA
        else if (!strcmp(f, "--cuda")) {
            use_cuda = 1;
        } else if (!strcmp(f, "--cuda-gemm-mode")) {
            cuda_fast_gemm = argv[++arg];
        } else if (!strcmp(f, "--cuda-weight-dtype")) {
            cuda_weights = argv[++arg];
        }
#endif
        else if (!strcmp(f, "--host")) {
            host = argv[++arg];
        } else if (!strcmp(f, "--port")) {
            port = atoi(argv[++arg]);
        } else if (!strcmp(f, "--api-key")) {
            api_key = argv[++arg];
        } else if (!strcmp(f, "-b") || !strcmp(f, "--batch-size")) {
            batch_size = atoi(argv[++arg]);
        } else if (!strcmp(f, "--max-batch-tokens")) {
            max_batch_tokens = atoi(argv[++arg]);
            if (max_batch_tokens <= 0) {
                fprintf(stderr, "--max-batch-tokens must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--batch-wait-us")) {
            batch_wait_us = atoi(argv[++arg]);
            if (batch_wait_us < 0) {
                fprintf(stderr, "--batch-wait-us must be >= 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "-t") || !strcmp(f, "--threads")) {
            n_threads = atoi(argv[++arg]);
        } else if (!strcmp(f, "-v") || !strcmp(f, "--verbose")) {
            verbose++;
        } else if (!strcmp(f, "-vv")) {
            verbose = 2;
        } else if (!strcmp(f, "-h") || !strcmp(f, "--help")) {
            print_usage(prog);
            free_model_specs(&model_specs);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", f);
            print_usage(prog);
            free_model_specs(&model_specs);
            return 1;
        }
        arg++;
    }

    embed_verbose = verbose;
    qwen_verbose = verbose;

    if (arg < argc) {
        fprintf(stderr, "unexpected argument: %s\n", argv[arg]);
        print_usage(prog);
        free_model_specs(&model_specs);
        return 1;
    }
    if (model_specs.n == 0) {
        fprintf(stderr, "at least one --model MODEL_ID=DIR is required\n");
        print_usage(prog);
        free_model_specs(&model_specs);
        return 1;
    }
#ifdef USE_MLX
    if (mlx_quantize_bits && !use_mlx) {
        fprintf(stderr, "--mlx-quant-bits requires --mlx\n");
        free_model_specs(&model_specs);
        return 1;
    }
#endif
#ifdef USE_CUDA
    if (cuda_fast_gemm) {
        if (!use_cuda) {
            fprintf(stderr, "--cuda-gemm-mode requires --cuda\n");
            free_model_specs(&model_specs);
            return 1;
        }
        if (embed_cuda_set_fast_gemm(cuda_fast_gemm) != 0) {
            fprintf(stderr, "--cuda-gemm-mode must be f32, tf32, bf16, or 16f\n");
            free_model_specs(&model_specs);
            return 1;
        }
    }
    if (cuda_weights) {
        if (!use_cuda) {
            fprintf(stderr, "--cuda-weight-dtype requires --cuda\n");
            free_model_specs(&model_specs);
            return 1;
        }
        if (!strcmp(cuda_weights, "bf16")) {
            embed_cuda_set_weights_bf16(1);
        } else if (!strcmp(cuda_weights, "f32")) {
            embed_cuda_set_weights_bf16(0);
        } else {
            fprintf(stderr, "--cuda-weight-dtype must be f32 or bf16\n");
            free_model_specs(&model_specs);
            return 1;
        }
    }
#endif

#ifdef USE_MLX
    if (use_mlx) {
        if (verbose >= 1)
            fprintf(stderr, "Using MLX GPU backend\n");
    } else
#endif
#ifdef USE_CUDA
        if (use_cuda) {
        if (verbose >= 1)
            fprintf(stderr, "Using CUDA GPU backend\n");
    } else
#endif
    {
        if (n_threads <= 0)
            n_threads = qwen_get_num_cpus();
        qwen_set_threads(n_threads);
        if (verbose >= 1)
            fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
    }

    embed_server_config_t scfg = {
        .models = model_specs.v,
        .n_models = model_specs.n,
        .host = host,
        .port = port,
        .batch_size = batch_size > 0 ? batch_size : EMBED_SERVER_DEFAULT_BATCH_SIZE,
        .max_batch_tokens = max_batch_tokens,
        .batch_wait_us = batch_wait_us,
#ifdef USE_MLX
        .use_mlx = use_mlx,
        .mlx_quantize_bits = mlx_quantize_bits,
        .mlx_quantize_group_size = mlx_quantize_group_size,
        .memory_utilization = memory_utilization,
#endif
#ifdef USE_CUDA
        .use_cuda = use_cuda,
#endif
        .api_key = api_key,
    };
    int rc = embed_run_server(&scfg);
    free_model_specs(&model_specs);
    return rc;
}

#endif /* EMBED_SERVER_TEST */
