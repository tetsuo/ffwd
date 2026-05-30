#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "pplx_server.h"
#include "pplx_embed.h"
#include "qwen_asr_tokenizer.h"

#ifdef USE_MLX
#include "pplx_embed_mlx.h"
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

#include <cjson/cJSON.h>

#define PPLX_HTTP_IO_BUF 8192
#define PPLX_HTTP_MAX_HEADER (64u * 1024u)
#define PPLX_HTTP_MAX_BODY (64u * 1024u * 1024u)
#define PPLX_HTTP_MAX_PATH 255u
#define PPLX_HTTP_BACKLOG 128
#define PPLX_HTTP_SETSIZE 10240
#define PPLX_HTTP_CLIENT_TIMEOUT_MS 30000
#define PPLX_HTTP_SWEEP_MS 1000

#define PPLX_API_MAX_STANDARD_INPUTS 512
#define PPLX_API_MAX_CONTEXT_DOCS 512
#define PPLX_API_MAX_CONTEXT_CHUNKS 16000
#define PPLX_API_MAX_ITEM_TOKENS 32768
#define PPLX_API_MAX_TOTAL_TOKENS 120000
#define PPLX_SERVER_DEFAULT_BATCH_SIZE 8
#define PPLX_SERVER_BATCH_WAIT_US 1000
#define PPLX_SERVER_MICROBATCH_MAX_JOBS 128

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} sbuf;

static void die_oom(void) {
    fprintf(stderr, "pplx-serve: out of memory\n");
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die_oom();
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die_oom();
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die_oom();
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
    if (add > SIZE_MAX - b->len - 1) die_oom();
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) die_oom();
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

static void sbuf_printf(sbuf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die_oom();
    sbuf_reserve(b, (size_t)n);
    int n2 = vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    if (n2 < 0 || n2 != n) die_oom();
    b->len += (size_t)n;
}

static void sbuf_free(sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

typedef enum {
    MODEL_STD_06,
    MODEL_STD_4,
    MODEL_CTX_06,
    MODEL_CTX_4,
    MODEL_UNKNOWN
} model_slot;

typedef struct {
    const char *id;
    int is_contextual;
    int dim;
    double price_per_mtok;
} model_info;

static const model_info k_models[] = {
    {"pplx-embed-v1-0.6b", 0, 1024, 0.004},
    {"pplx-embed-v1-4b", 0, 2560, 0.030},
    {"pplx-embed-context-v1-0.6b", 1, 1024, 0.008},
    {"pplx-embed-context-v1-4b", 1, 2560, 0.050},
};

static model_slot model_slot_for_id(const char *id) {
    if (!id) return MODEL_UNKNOWN;
    for (int i = 0; i < 4; i++) {
        if (!strcmp(id, k_models[i].id)) return (model_slot)i;
    }
    return MODEL_UNKNOWN;
}

typedef struct {
    const model_info *info;
    char *path;
    qwen_tokenizer_t *tok;
    int newline_n;
    int *newline_ids;
    pplx_model_t *cpu_model;
    pplx_workspace_t *cpu_ws;
#ifdef USE_MLX
    pplx_mlx_ctx_t *mlx_ctx;
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
    int batch_wait_us;
    int enable_cors;
    char *api_key;
    loaded_model models[4];
    int use_mlx;

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
    char path[PPLX_HTTP_MAX_PATH + 1];
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
    http_request req;
    client *prev;
    client *next;
};

struct job {
    http_server *srv;
    client *c;
    char method[8];
    char path[PPLX_HTTP_MAX_PATH + 1];
    char *body;
    size_t body_len;
    char *auth;
    int status;
    char *content_type;
    char *response;
    size_t response_len;
    job *next;
};

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_signal_wfd = -1;

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    int saved = errno;
    g_stop_requested = 1;
    int fd = (int)g_signal_wfd;
    if (fd >= 0) {
        const unsigned char byte = 1;
        (void)write(fd, &byte, 1);
    }
    errno = saved;
}

static int set_nonblock(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static void http_request_free(http_request *r) {
    free(r->body);
    free(r->auth);
    memset(r, 0, sizeof(*r));
}

static void client_free(client *c) {
    if (!c) return;
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
    if (--c->refcount == 0) free_now = 1;
    pthread_mutex_unlock(&c->srv->mu);
    if (free_now) client_free(c);
}

static void client_unlink(client *c) {
    http_server *s = c->srv;
    if (!c->linked) return;
    if (c->prev) c->prev->next = c->next;
    else s->clients = c->next;
    if (c->next) c->next->prev = c->prev;
    c->prev = c->next = NULL;
    c->linked = 0;
    if (s->n_clients > 0) s->n_clients--;
}

static void close_client(client *c) {
    if (!c || c->cancelled) return;
    c->cancelled = 1;
    if (c->fd >= 0) {
        aeDeleteFileEvent(c->srv->loop, c->fd, AE_READABLE);
        aeDeleteFileEvent(c->srv->loop, c->fd, AE_WRITABLE);
        close(c->fd);
        c->fd = -1;
    }
    client_unlink(c);
    client_decref(c);
}

static void append_http_response(client *c, int status, const char *ctype,
                                 const char *body, size_t body_len) {
    const char *reason = "OK";
    if (status == 204) reason = "No Content";
    else if (status == 400) reason = "Bad Request";
    else if (status == 401) reason = "Unauthorized";
    else if (status == 404) reason = "Not Found";
    else if (status == 405) reason = "Method Not Allowed";
    else if (status == 422) reason = "Unprocessable Entity";
    else if (status == 503) reason = "Service Unavailable";
    else if (status >= 500) reason = "Internal Server Error";

    sbuf_printf(&c->out,
                "HTTP/1.1 %d %s\r\n"
                "Content-Length: %zu\r\n"
                "Connection: close\r\n",
                status, reason, body_len);
    if (ctype)
        sbuf_printf(&c->out, "Content-Type: %s\r\n", ctype);
    if (c->srv->enable_cors) {
        sbuf_puts(&c->out, "Access-Control-Allow-Origin: *\r\n");
        sbuf_puts(&c->out, "Access-Control-Allow-Headers: authorization, content-type\r\n");
        sbuf_puts(&c->out, "Access-Control-Allow-Methods: POST, OPTIONS\r\n");
    }
    sbuf_puts(&c->out, "\r\n");
    if (body_len) sbuf_append(&c->out, body, body_len);
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
    if (!s->api_key || !s->api_key[0]) return true;
    if (!auth) return false;
    const char prefix[] = "Bearer ";
    return !strncmp(auth, prefix, sizeof(prefix) - 1) &&
           !strcmp(auth + sizeof(prefix) - 1, s->api_key);
}

static ssize_t find_header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' &&
            p[i - 1] == '\r' && p[i] == '\n')
            return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

static char *header_value_dup(const char *h, size_t n, const char *name) {
    size_t name_len = strlen(name);
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len > name_len && !strncasecmp(line, name, name_len) &&
            line[name_len] == ':') {
            const char *v = line + name_len + 1;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            const char *e = line + len;
            while (e > v && isspace((unsigned char)e[-1])) e--;
            return xstrndup(v, (size_t)(e - v));
        }
        if (p < end) p++;
    }
    return NULL;
}

static int parse_request_headers(client *c) {
    char line[512];
    size_t i = 0;
    while (i < c->header_len && c->in.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = c->in.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", c->req.method, c->req.path) != 2)
        return -1;
    char *q = strchr(c->req.path, '?');
    if (q) *q = '\0';

    char *cl = header_value_dup(c->in.ptr, c->header_len, "Content-Length");
    if (cl) {
        char *end = NULL;
        errno = 0;
        unsigned long long v = strtoull(cl, &end, 10);
        if (errno || !end || *end || v > PPLX_HTTP_MAX_BODY) {
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
            if (c->in.len > PPLX_HTTP_MAX_HEADER) return -1;
            return 0;
        }
        c->header_done = 1;
        c->header_len = (size_t)hend;
        if (parse_request_headers(c) != 0) return -1;
    }
    if (c->content_length > PPLX_HTTP_MAX_BODY) return -1;
    if (c->in.len < c->header_len + c->content_length) return 0;
    c->req.body_len = c->content_length;
    c->req.body = xmalloc(c->content_length + 1);
    memcpy(c->req.body, c->in.ptr + c->header_len, c->content_length);
    c->req.body[c->content_length] = '\0';
    return 1;
}

static void write_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop; (void)mask;
    client *c = clientData;
    while (c->sent < c->out.len) {
        ssize_t n = write(fd, c->out.ptr + c->sent, c->out.len - c->sent);
        if (n > 0) {
            c->sent += (size_t)n;
            c->last_active_ms = mstime();
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
        close_client(c);
        return;
    }
    close_client(c);
}

static int queue_write(client *c) {
    if (!c || c->cancelled || c->fd < 0) return -1;
    return aeCreateFileEvent(c->srv->loop, c->fd, AE_WRITABLE, write_cb, c);
}

static void enqueue_done(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->done_tail) s->done_tail->next = j;
    else s->done_head = j;
    s->done_tail = j;
    pthread_mutex_unlock(&s->mu);
    const unsigned char byte = 1;
    (void)write(s->completion_pipe[1], &byte, 1);
}

static job *dequeue_job(http_server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->job_head && !s->stopping)
        pthread_cond_wait(&s->cv, &s->mu);
    job *j = s->job_head;
    if (j) {
        s->job_head = j->next;
        if (!s->job_head) s->job_tail = NULL;
        j->next = NULL;
    }
    pthread_mutex_unlock(&s->mu);
    return j;
}

static int drain_ready_jobs(http_server *s, job **jobs, int max_jobs) {
    int n = 0;
    pthread_mutex_lock(&s->mu);
    while (n < max_jobs && s->job_head) {
        job *j = s->job_head;
        s->job_head = j->next;
        if (!s->job_head) s->job_tail = NULL;
        j->next = NULL;
        jobs[n++] = j;
    }
    pthread_mutex_unlock(&s->mu);
    return n;
}

static void enqueue_job(job *j) {
    http_server *s = j->srv;
    pthread_mutex_lock(&s->mu);
    if (s->job_tail) s->job_tail->next = j;
    else s->job_head = j;
    s->job_tail = j;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
}

static job *pop_done(http_server *s) {
    pthread_mutex_lock(&s->mu);
    job *j = s->done_head;
    if (j) {
        s->done_head = j->next;
        if (!s->done_head) s->done_tail = NULL;
        j->next = NULL;
    }
    pthread_mutex_unlock(&s->mu);
    return j;
}

static void job_free(job *j) {
    if (!j) return;
    free(j->body);
    free(j->auth);
    free(j->response);
    free(j);
}

static void completion_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop; (void)mask;
    http_server *s = clientData;
    unsigned char tmp[128];
    while (read(fd, tmp, sizeof(tmp)) > 0) {}

    job *j;
    while ((j = pop_done(s)) != NULL) {
        client *c = j->c;
        if (!c->cancelled) {
            append_http_response(c, j->status, j->content_type,
                                 j->response ? j->response : "",
                                 j->response ? j->response_len : 0);
            if (queue_write(c) != AE_OK)
                close_client(c);
        }
        client_decref(c);
        job_free(j);
    }
}

static bool route_is_embedding(const char *method, const char *path) {
    return !strcmp(method, "POST") &&
           (!strcmp(path, "/v1/embeddings") ||
            !strcmp(path, "/v1/contextualizedembeddings"));
}

static void dispatch_request(client *c) {
    aeDeleteFileEvent(c->srv->loop, c->fd, AE_READABLE);

    if (!strcmp(c->req.method, "OPTIONS")) {
        append_http_response(c, 204, NULL, "", 0);
        if (queue_write(c) != AE_OK) close_client(c);
        return;
    }

    if (!route_is_embedding(c->req.method, c->req.path)) {
        size_t len;
        char *body = json_error_body("unknown endpoint", "invalid_request_error", &len);
        append_http_response(c, 404, "application/json", body, len);
        free(body);
        if (queue_write(c) != AE_OK) close_client(c);
        return;
    }

    if (!auth_ok(c->srv, c->req.auth)) {
        size_t len;
        char *body = json_error_body("missing or invalid bearer token",
                                     "authentication_error", &len);
        append_http_response(c, 401, "application/json", body, len);
        free(body);
        if (queue_write(c) != AE_OK) close_client(c);
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
    client_incref(c);
    enqueue_job(j);
}

static void read_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop; (void)mask;
    client *c = clientData;
    char tmp[PPLX_HTTP_IO_BUF];
    for (;;) {
        ssize_t n = read(fd, tmp, sizeof(tmp));
        if (n > 0) {
            sbuf_append(&c->in, tmp, (size_t)n);
            c->last_active_ms = mstime();
            int done = complete_request(c);
            if (done < 0) {
                size_t len;
                char *body = json_error_body("bad HTTP request",
                                             "invalid_request_error", &len);
                append_http_response(c, 400, "application/json", body, len);
                free(body);
                aeDeleteFileEvent(c->srv->loop, c->fd, AE_READABLE);
                if (queue_write(c) != AE_OK) close_client(c);
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
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        close_client(c);
        return;
    }
}

static void accept_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)loop; (void)mask;
    http_server *s = clientData;
    char cip[64];
    int cport = 0;
    for (;;) {
        int cfd = anetTcpAccept(NULL, fd, cip, sizeof(cip), &cport);
        if (cfd == ANET_ERR) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return;
            if (errno == EINTR) continue;
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
        if (s->clients) s->clients->prev = c;
        s->clients = c;
        s->n_clients++;
        if (aeCreateFileEvent(s->loop, cfd, AE_READABLE, read_cb, c) != AE_OK)
            close_client(c);
    }
}

static int timeout_cb(aeEventLoop *loop, long long id, void *clientData) {
    (void)loop; (void)id;
    http_server *s = clientData;
    uint64_t now = mstime();
    client *c = s->clients;
    while (c) {
        client *next = c->next;
        if (!c->cancelled &&
            now - c->last_active_ms > PPLX_HTTP_CLIENT_TIMEOUT_MS)
            close_client(c);
        c = next;
    }
    return PPLX_HTTP_SWEEP_MS;
}

static void signal_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
    (void)fd; (void)mask;
    http_server *s = clientData;
    unsigned char tmp[128];
    while (read(s->signal_pipe[0], tmp, sizeof(tmp)) > 0) {}
    aeStop(loop);
}

static bool ve_add(cJSON *detail, const char *loc_json,
                   const char *msg, const char *type) {
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
    while (end && end < body_end && isspace((unsigned char)*end)) end++;
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
    if (!cJSON_IsNumber(item)) return false;
    double d = item->valuedouble;
    int i = item->valueint;
    return d >= (double)INT_MIN && d <= (double)INT_MAX && d == (double)i;
}

static const char *encoding_from_root(cJSON *root, cJSON *detail) {
    cJSON *encoding = cJSON_GetObjectItemCaseSensitive(root, "encoding_format");
    if (!encoding) return "base64_int8";
    if (!cJSON_IsString(encoding) || !encoding->valuestring) {
        ve_add(detail, "[\"body\",\"encoding_format\"]",
               "encoding_format must be a string", "type_error.string");
        return "base64_int8";
    }
    if (strcmp(encoding->valuestring, "base64_int8") &&
        strcmp(encoding->valuestring, "base64_binary") &&
        strcmp(encoding->valuestring, "float")) {
        ve_add(detail, "[\"body\",\"encoding_format\"]",
               "value is not a valid enum member; permitted: 'base64_int8', 'base64_binary', 'float'",
               "enum");
        return "base64_int8";
    }
    return encoding->valuestring;
}

static int dimensions_from_root(cJSON *root, cJSON *detail, int max_dim,
                                const char *encoding) {
    int dims = max_dim;
    cJSON *dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    if (dimensions) {
        if (!cjson_is_integer(dimensions)) {
            ve_add(detail, "[\"body\",\"dimensions\"]", "dimensions must be an integer",
                   "type_error.integer");
        } else {
            dims = dimensions->valueint;
            if (dims < 128 || dims > max_dim) {
                char msg[96];
                snprintf(msg, sizeof(msg), "dimensions must be between 128 and %d", max_dim);
                ve_add(detail, "[\"body\",\"dimensions\"]", msg, "value_error.range");
            }
        }
    }
    if (!strcmp(encoding, "base64_binary") && dims % 8 != 0)
        ve_add(detail, "[\"body\",\"dimensions\"]",
               "dimensions must be divisible by 8 for base64_binary",
               "value_error.divisible");
    return dims;
}

static char *base64_encode(const unsigned char *src, size_t n) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t out_len = ((n + 2) / 3) * 4;
    char *out = xmalloc(out_len + 1);
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = (unsigned)src[i] << 16;
        int remain = (int)(n - i);
        if (remain > 1) v |= (unsigned)src[i + 1] << 8;
        if (remain > 2) v |= (unsigned)src[i + 2];
        out[o++] = tbl[(v >> 18) & 63];
        out[o++] = tbl[(v >> 12) & 63];
        out[o++] = remain > 1 ? tbl[(v >> 6) & 63] : '=';
        out[o++] = remain > 2 ? tbl[v & 63] : '=';
    }
    out[o] = '\0';
    return out;
}

static char *encode_embedding(const float *emb, int dims, const char *encoding) {
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
    float max_abs = 0.0f;
    for (int i = 0; i < dims; i++) {
        float a = fabsf(emb[i]);
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs > 1e-12f ? 127.0f / max_abs : 0.0f;
    for (int i = 0; i < dims; i++) {
        int v = (int)lrintf(emb[i] * scale);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (signed char)v;
    }
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
    pplx_input_t *inputs;
    int total_tokens;
    int ready;
} embedding_request;

static void free_token_bufs(token_buf *t, int n) {
    if (!t) return;
    for (int i = 0; i < n; i++) free(t[i].ids);
}

static void embedding_request_free(embedding_request *r) {
    if (!r) return;
    free_token_bufs(r->tokens, r->n_inputs);
    free(r->tokens);
    free(r->inputs);
    if (r->root) cJSON_Delete(r->root);
    memset(r, 0, sizeof(*r));
}

static int tokenize_one(loaded_model *m, const char *text, token_buf *out) {
    memset(out, 0, sizeof(*out));
    out->ids = qwen_tokenizer_encode(m->tok, text, &out->n_tokens);
    if (!out->ids || out->n_tokens <= 0) {
        free(out->ids);
        memset(out, 0, sizeof(*out));
        return -1;
    }
    return 0;
}

static int model_embed_batch(loaded_model *m, const pplx_input_t *inputs,
                             int batch, float *out) {
#ifdef USE_MLX
    if (m->mlx_ctx)
        return pplx_mlx_embed_batch(m->mlx_ctx, inputs, batch, out);
#endif
    return pplx_model_embed_batch(m->cpu_model, m->cpu_ws, inputs, batch, out);
}

static int model_embed_spans(loaded_model *m, const int *ids, int n_tokens,
                             const pplx_span_t *spans, int n_spans, float *out) {
#ifdef USE_MLX
    if (m->mlx_ctx)
        return pplx_mlx_embed_spans(m->mlx_ctx, ids, n_tokens, spans, n_spans, out);
#endif
    return pplx_model_embed_spans(m->cpu_model, m->cpu_ws, ids, n_tokens,
                                  spans, n_spans, out);
}

static void append_embedding_object(sbuf *b, int index, const char *embedding) {
    sbuf_printf(b, "{\"object\":\"embedding\",\"index\":%d,\"embedding\":\"", index);
    sbuf_puts(b, embedding);
    sbuf_puts(b, "\"}");
}

static void append_float_embedding_object(sbuf *b, int index,
                                          const float *emb, int dims) {
    sbuf_printf(b, "{\"object\":\"embedding\",\"index\":%d,\"embedding\":[", index);
    for (int i = 0; i < dims; i++) {
        if (i) sbuf_putc(b, ',');
        sbuf_printf(b, "%.9g", (double)emb[i]);
    }
    sbuf_puts(b, "]}");
}

static void set_response_from_buf(job *j, sbuf *b) {
    j->status = 200;
    j->content_type = "application/json";
    j->response = b->ptr;
    j->response_len = b->len;
    memset(b, 0, sizeof(*b));
}

static void render_embedding_response(embedding_request *r, const float *embs) {
    char **encoded = NULL;
    if (strcmp(r->encoding, "float")) {
        encoded = xcalloc((size_t)r->n_inputs, sizeof(*encoded));
        for (int i = 0; i < r->n_inputs; i++)
            encoded[i] = encode_embedding(embs + (size_t)i * r->model->info->dim,
                                          r->dims, r->encoding);
    }

    sbuf b = {0};
    sbuf_puts(&b, "{\"object\":\"list\",\"data\":[");
    for (int i = 0; i < r->n_inputs; i++) {
        if (i) sbuf_putc(&b, ',');
        if (encoded)
            append_embedding_object(&b, i, encoded[i]);
        else
            append_float_embedding_object(&b, i,
                embs + (size_t)i * r->model->info->dim, r->dims);
    }
    double cost = ((double)r->total_tokens / 1000000.0) *
                  r->model->info->price_per_mtok;
    sbuf_printf(&b,
        "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%d,"
        "\"total_tokens\":%d,\"cost\":{\"input_cost\":%.10g,"
        "\"total_cost\":%.10g,\"currency\":\"USD\"}}}",
        r->model->info->id, r->total_tokens, r->total_tokens, cost, cost);
    set_response_from_buf(r->j, &b);

    for (int i = 0; encoded && i < r->n_inputs; i++) free(encoded[i]);
    free(encoded);
}

static int embedding_request_compatible(const embedding_request *a,
                                        const embedding_request *b) {
    return a->ready && b->ready &&
           a->model == b->model &&
           a->dims == b->dims &&
           !strcmp(a->encoding, b->encoding);
}

static void execute_embedding_request_list(embedding_request **reqs, int n_reqs) {
    if (n_reqs <= 0) return;
    loaded_model *m = reqs[0]->model;
    int dim = m->info->dim;
    int total_inputs = 0;
    for (int i = 0; i < n_reqs; i++)
        total_inputs += reqs[i]->n_inputs;

    pplx_input_t *inputs = xmalloc((size_t)total_inputs * sizeof(*inputs));
    float *embs = xmalloc((size_t)total_inputs * dim * sizeof(float));

    int pos = 0;
    for (int i = 0; i < n_reqs; i++) {
        memcpy(inputs + pos, reqs[i]->inputs,
               (size_t)reqs[i]->n_inputs * sizeof(*inputs));
        pos += reqs[i]->n_inputs;
    }

    int max_batch = reqs[0]->j->srv->batch_size > 0
        ? reqs[0]->j->srv->batch_size : total_inputs;
    int failed = 0;
    for (int start = 0; start < total_inputs; start += max_batch) {
        int cur = total_inputs - start;
        if (cur > max_batch) cur = max_batch;
        if (model_embed_batch(m, inputs + start, cur,
                              embs + (size_t)start * dim) != 0) {
            failed = 1;
            break;
        }
    }

    pos = 0;
    for (int i = 0; i < n_reqs; i++) {
        if (failed) {
            job_set_error(reqs[i]->j, 500, "embedding failed", "server_error");
        } else {
            render_embedding_response(reqs[i], embs + (size_t)pos * dim);
        }
        pos += reqs[i]->n_inputs;
    }

    free(embs);
    free(inputs);
}

static void execute_embedding_requests(embedding_request *reqs, int n_reqs) {
    if (n_reqs <= 0) return;
    embedding_request **list = xmalloc((size_t)n_reqs * sizeof(*list));
    for (int i = 0; i < n_reqs; i++)
        list[i] = &reqs[i];
    execute_embedding_request_list(list, n_reqs);
    free(list);
}

static void handle_context_job(job *j, cJSON *root, loaded_model *m,
                               int dims, const char *encoding) {
    cJSON *input = cJSON_GetObjectItemCaseSensitive(root, "input");
    int n_docs = cJSON_GetArraySize(input);
    int *chunks = xcalloc((size_t)n_docs, sizeof(*chunks));
    char ***encoded = strcmp(encoding, "float")
        ? xcalloc((size_t)n_docs, sizeof(*encoded)) : NULL;
    float **float_embs = !strcmp(encoding, "float")
        ? xcalloc((size_t)n_docs, sizeof(*float_embs)) : NULL;
    int total_tokens = 0;

    cJSON *doc_arr;
    int di = 0;
    cJSON_ArrayForEach(doc_arr, input) {
        chunks[di] = cJSON_GetArraySize(doc_arr);
        if (encoded) encoded[di] = xcalloc((size_t)chunks[di], sizeof(char *));
        token_buf *chunk_tokens = xcalloc((size_t)chunks[di], sizeof(*chunk_tokens));
        pplx_span_t *spans = xcalloc((size_t)chunks[di], sizeof(*spans));
        int doc_tokens = 0;
        cJSON *chunk;
        int ci = 0;
        cJSON_ArrayForEach(chunk, doc_arr) {
            if (tokenize_one(m, chunk->valuestring, &chunk_tokens[ci]) != 0) {
                job_set_error(j, 500, "tokenization failed", "server_error");
                free_token_bufs(chunk_tokens, chunks[di]);
                free(chunk_tokens); free(spans);
                goto done;
            }
            doc_tokens += chunk_tokens[ci].n_tokens;
            if (ci + 1 < chunks[di]) doc_tokens += m->newline_n;
            ci++;
        }
        int *doc_ids = xmalloc((size_t)doc_tokens * sizeof(int));
        int pos = 0;
        for (ci = 0; ci < chunks[di]; ci++) {
            spans[ci].start = pos;
            spans[ci].n_tokens = chunk_tokens[ci].n_tokens;
            memcpy(doc_ids + pos, chunk_tokens[ci].ids,
                   (size_t)chunk_tokens[ci].n_tokens * sizeof(int));
            pos += chunk_tokens[ci].n_tokens;
            if (ci + 1 < chunks[di] && m->newline_n > 0) {
                memcpy(doc_ids + pos, m->newline_ids,
                       (size_t)m->newline_n * sizeof(int));
                pos += m->newline_n;
            }
        }
        total_tokens += doc_tokens;
        float *embs = xmalloc((size_t)chunks[di] * m->info->dim * sizeof(float));
        if (model_embed_spans(m, doc_ids, doc_tokens, spans, chunks[di], embs) != 0) {
            free(embs); free(doc_ids);
            job_set_error(j, 500, "contextual embedding failed", "server_error");
            free_token_bufs(chunk_tokens, chunks[di]);
            free(chunk_tokens); free(spans);
            goto done;
        }
        if (encoded) {
            for (ci = 0; ci < chunks[di]; ci++)
                encoded[di][ci] = encode_embedding(embs + (size_t)ci * m->info->dim,
                                                   dims, encoding);
            free(embs);
        } else {
            float_embs[di] = embs;
        }
        free(doc_ids);
        free_token_bufs(chunk_tokens, chunks[di]);
        free(chunk_tokens);
        free(spans);
        di++;
    }

    sbuf b = {0};
    sbuf_puts(&b, "{\"object\":\"list\",\"data\":[");
    for (di = 0; di < n_docs; di++) {
        if (di) sbuf_putc(&b, ',');
        sbuf_printf(&b, "{\"object\":\"list\",\"index\":%d,\"data\":[", di);
        for (int ci = 0; ci < chunks[di]; ci++) {
            if (ci) sbuf_putc(&b, ',');
            if (encoded)
                append_embedding_object(&b, ci, encoded[di][ci]);
            else
                append_float_embedding_object(&b, ci,
                    float_embs[di] + (size_t)ci * m->info->dim, dims);
        }
        sbuf_puts(&b, "]}");
    }
    sbuf_printf(&b,
        "],\"model\":\"%s\",\"usage\":{\"prompt_tokens\":%d,\"total_tokens\":%d}}",
        m->info->id, total_tokens, total_tokens);
    set_response_from_buf(j, &b);

done:
    for (di = 0; di < n_docs; di++) {
        if (encoded) {
            if (!encoded[di]) continue;
            for (int ci = 0; ci < chunks[di]; ci++) free(encoded[di][ci]);
            free(encoded[di]);
        } else if (float_embs) {
            free(float_embs[di]);
        }
    }
    free(encoded);
    free(float_embs);
    free(chunks);
}

static int validate_common(cJSON *root, cJSON *detail, int expect_context,
                           loaded_model **out_model, int *out_dims,
                           const char **out_encoding, http_server *s) {
    cJSON *model_item = cJSON_GetObjectItemCaseSensitive(root, "model");
    const char *model_id = NULL;
    if (!model_item) {
        ve_add(detail, "[\"body\",\"model\"]", "field required", "missing");
    } else if (!cJSON_IsString(model_item) || !model_item->valuestring) {
        ve_add(detail, "[\"body\",\"model\"]", "model must be a string", "type_error.string");
    } else {
        model_id = model_item->valuestring;
        model_slot slot = model_slot_for_id(model_id);
        if (slot == MODEL_UNKNOWN || k_models[slot].is_contextual != expect_context) {
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
    const char *enc = encoding_from_root(root, detail);
    int max_dim = model_id && model_slot_for_id(model_id) != MODEL_UNKNOWN
        ? k_models[model_slot_for_id(model_id)].dim : 2560;
    int dims = dimensions_from_root(root, detail, max_dim, enc);
    *out_encoding = enc;
    *out_dims = dims;
    return model_id && model_slot_for_id(model_id) != MODEL_UNKNOWN &&
           !s->models[model_slot_for_id(model_id)].info ? 503 : 0;
}

static void prepare_embedding_request(job *j, cJSON *root, http_server *s,
                                      embedding_request *out) {
    memset(out, 0, sizeof(*out));
    out->j = j;
    out->root = root;

    cJSON *detail = cJSON_CreateArray();
    loaded_model *m = NULL;
    int dims = 0;
    const char *encoding = NULL;
    int unloaded = validate_common(root, detail, 0, &m, &dims, &encoding, s);
    out->model = m;
    out->dims = dims;
    out->encoding = encoding;

    cJSON *input = cJSON_GetObjectItemCaseSensitive(root, "input");
    int n_inputs = 0;
    if (!input) {
        ve_add(detail, "[\"body\",\"input\"]", "field required", "missing");
    } else if (cJSON_IsString(input)) {
        if (!input->valuestring || input->valuestring[0] == '\0')
            ve_add(detail, "[\"body\",\"input\"]", "input must not be empty",
                   "value_error.empty");
        n_inputs = 1;
    } else if (cJSON_IsArray(input)) {
        n_inputs = cJSON_GetArraySize(input);
        if (n_inputs < 1)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at least 1 item",
                   "value_error.list.min_items");
        else if (n_inputs > PPLX_API_MAX_STANDARD_INPUTS)
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
        ve_add(detail, "[\"body\",\"input\"]",
               "input must be a string or an array of strings", "type_error");
    }

    if (cJSON_GetArraySize(detail) == 0 && unloaded) {
        job_set_error(j, 503, "requested model is valid but not loaded",
                      "model_not_loaded");
        cJSON_Delete(detail);
        return;
    }

    if (cJSON_GetArraySize(detail) == 0 && m) {
        out->n_inputs = n_inputs;
        out->tokens = xcalloc((size_t)n_inputs, sizeof(*out->tokens));
        out->inputs = xmalloc((size_t)n_inputs * sizeof(*out->inputs));

        int idx = 0;
        if (cJSON_IsString(input)) {
            if (tokenize_one(m, input->valuestring, &out->tokens[0]) != 0) {
                ve_add(detail, "[\"body\",\"input\"]", "tokenization failed",
                       "value_error.tokenization");
            } else {
                out->inputs[0].ids = out->tokens[0].ids;
                out->inputs[0].n_tokens = out->tokens[0].n_tokens;
                out->total_tokens = out->tokens[0].n_tokens;
                if (out->tokens[0].n_tokens > PPLX_API_MAX_ITEM_TOKENS)
                    ve_add(detail, "[\"body\",\"input\"]",
                           "input exceeds 32768 token limit",
                           "value_error.context_length");
            }
        } else {
            cJSON *item;
            cJSON_ArrayForEach(item, input) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", idx);
                if (tokenize_one(m, item->valuestring, &out->tokens[idx]) != 0) {
                    ve_add(detail, loc, "tokenization failed",
                           "value_error.tokenization");
                } else {
                    out->inputs[idx].ids = out->tokens[idx].ids;
                    out->inputs[idx].n_tokens = out->tokens[idx].n_tokens;
                    out->total_tokens += out->tokens[idx].n_tokens;
                    if (out->tokens[idx].n_tokens > PPLX_API_MAX_ITEM_TOKENS)
                        ve_add(detail, loc, "input exceeds 32768 token limit",
                               "value_error.context_length");
                }
                idx++;
            }
        }

        if (out->total_tokens > PPLX_API_MAX_TOTAL_TOKENS)
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

static void process_embeddings(job *j, cJSON *root, http_server *s) {
    embedding_request r;
    prepare_embedding_request(j, root, s, &r);
    if (r.ready)
        execute_embedding_requests(&r, 1);
    embedding_request_free(&r);
}

static void process_contextual(job *j, cJSON *root, http_server *s) {
    cJSON *detail = cJSON_CreateArray();
    loaded_model *m = NULL;
    int dims = 0;
    const char *encoding = NULL;
    int unloaded = validate_common(root, detail, 1, &m, &dims, &encoding, s);

    cJSON *input = cJSON_GetObjectItemCaseSensitive(root, "input");
    int n_docs = 0, total_chunks = 0, total_tokens = 0;
    if (!input) {
        ve_add(detail, "[\"body\",\"input\"]", "field required", "missing");
    } else if (!cJSON_IsArray(input)) {
        ve_add(detail, "[\"body\",\"input\"]",
               "input must be an array of document chunk arrays", "type_error.array");
    } else {
        n_docs = cJSON_GetArraySize(input);
        if (n_docs < 1)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at least 1 document",
                   "value_error.list.min_items");
        else if (n_docs > PPLX_API_MAX_CONTEXT_DOCS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 512 documents",
                   "value_error.list.max_items");
        cJSON *doc_arr;
        int di = 0;
        cJSON_ArrayForEach(doc_arr, input) {
            if (!cJSON_IsArray(doc_arr)) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", di);
                ve_add(detail, loc, "document must be an array of strings",
                       "type_error.array");
                di++;
                continue;
            }
            int doc_chunks = cJSON_GetArraySize(doc_arr);
            total_chunks += doc_chunks;
            if (doc_chunks < 1) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", di);
                ve_add(detail, loc, "document must contain at least 1 chunk",
                       "value_error.list.min_items");
            }
            int doc_tokens = 0;
            cJSON *chunk;
            int ci = 0;
            cJSON_ArrayForEach(chunk, doc_arr) {
                char loc[80];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d,%d]", di, ci);
                if (!cJSON_IsString(chunk)) {
                    ve_add(detail, loc, "chunk must be a string", "type_error.string");
                } else if (!chunk->valuestring || chunk->valuestring[0] == '\0') {
                    ve_add(detail, loc, "chunk must not be empty", "value_error.empty");
                } else if (m) {
                    token_buf t;
                    if (tokenize_one(m, chunk->valuestring, &t) != 0) {
                        ve_add(detail, loc, "tokenization failed", "value_error.tokenization");
                    } else {
                        doc_tokens += t.n_tokens;
                        total_tokens += t.n_tokens;
                        free(t.ids);
                    }
                }
                ci++;
            }
            if (m && doc_chunks > 1) {
                doc_tokens += (doc_chunks - 1) * m->newline_n;
                total_tokens += (doc_chunks - 1) * m->newline_n;
            }
            if (doc_tokens > PPLX_API_MAX_ITEM_TOKENS) {
                char loc[64];
                snprintf(loc, sizeof(loc), "[\"body\",\"input\",%d]", di);
                ve_add(detail, loc, "document exceeds 32768 token limit",
                       "value_error.context_length");
            }
            di++;
        }
        if (total_chunks > PPLX_API_MAX_CONTEXT_CHUNKS)
            ve_add(detail, "[\"body\",\"input\"]", "input must contain at most 16000 chunks",
                   "value_error.list.max_items");
        if (total_tokens > PPLX_API_MAX_TOTAL_TOKENS)
            ve_add(detail, "[\"body\",\"input\"]", "request exceeds 120000 token limit",
                   "value_error.context_length");
    }

    if (cJSON_GetArraySize(detail) > 0) {
        job_set_422(j, detail);
    } else if (unloaded) {
        job_set_error(j, 503, "requested model is valid but not loaded",
                      "model_not_loaded");
    } else {
        handle_context_job(j, root, m, dims, encoding);
    }
    cJSON_Delete(detail);
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

static void process_job(job *j) {
    cJSON *root = NULL;
    if (parse_job_root(j, &root) != 0)
        return;
    if (!strcmp(j->path, "/v1/embeddings")) {
        process_embeddings(j, root, j->srv);
        return;
    } else if (!strcmp(j->path, "/v1/contextualizedembeddings"))
        process_contextual(j, root, j->srv);
    else
        job_set_error(j, 404, "unknown endpoint", "invalid_request_error");
    cJSON_Delete(root);
}

static void process_job_group(http_server *s, job **jobs, int n_jobs) {
    embedding_request *reqs = xcalloc((size_t)n_jobs, sizeof(*reqs));
    int *is_standard = xcalloc((size_t)n_jobs, sizeof(*is_standard));
    int *done = xcalloc((size_t)n_jobs, sizeof(*done));
    embedding_request **group = xmalloc((size_t)n_jobs * sizeof(*group));

    for (int i = 0; i < n_jobs; i++) {
        if (strcmp(jobs[i]->path, "/v1/embeddings"))
            continue;
        cJSON *root = NULL;
        is_standard[i] = 1;
        if (parse_job_root(jobs[i], &root) == 0)
            prepare_embedding_request(jobs[i], root, s, &reqs[i]);
    }

    for (int i = 0; i < n_jobs; i++) {
        if (done[i])
            continue;

        if (!is_standard[i]) {
            process_job(jobs[i]);
            enqueue_done(jobs[i]);
            done[i] = 1;
            continue;
        }

        if (!reqs[i].ready) {
            enqueue_done(jobs[i]);
            done[i] = 1;
            continue;
        }

        int group_n = 1;
        group[0] = &reqs[i];
        for (int k = i + 1; k < n_jobs; k++) {
            if (!done[k] && is_standard[k] && reqs[k].ready &&
                embedding_request_compatible(&reqs[i], &reqs[k]))
                group[group_n++] = &reqs[k];
        }

        execute_embedding_request_list(group, group_n);
        for (int k = 0; k < group_n; k++) {
            int idx = (int)(group[k] - reqs);
            enqueue_done(jobs[idx]);
            done[idx] = 1;
        }
    }

    for (int i = 0; i < n_jobs; i++)
        embedding_request_free(&reqs[i]);
    free(group);
    free(done);
    free(is_standard);
    free(reqs);
}

static void *worker_main(void *arg) {
    http_server *s = arg;
#ifdef USE_MLX
    if (s->use_mlx) {
        int rc = 0;
        for (int i = 0; i < 4; i++) {
            loaded_model *m = &s->models[i];
            if (!m->info) continue;
            m->mlx_ctx = pplx_mlx_load(m->path);
            if (!m->mlx_ctx) {
                server_log("pplx-serve: failed to load MLX model on worker: %s",
                           m->path);
                rc = 1;
                break;
            }
            if (pplx_mlx_config(m->mlx_ctx)->hidden_size != m->info->dim) {
                server_log("pplx-serve: model %s has unexpected dimension %d",
                           m->info->id,
                           pplx_mlx_config(m->mlx_ctx)->hidden_size);
                rc = 1;
                break;
            }
        }
        pthread_mutex_lock(&s->mu);
        s->worker_init_rc = rc;
        s->worker_ready = 1;
        if (rc) s->stopping = 1;
        pthread_cond_broadcast(&s->cv);
        pthread_mutex_unlock(&s->mu);
        if (rc) {
            for (int i = 0; i < 4; i++) {
                loaded_model *m = &s->models[i];
                if (m->mlx_ctx) {
                    pplx_mlx_free(m->mlx_ctx);
                    m->mlx_ctx = NULL;
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
        job *j = dequeue_job(s);
        if (!j) break;
        job *jobs[PPLX_SERVER_MICROBATCH_MAX_JOBS];
        int n_jobs = 1;
        jobs[0] = j;

        if (!strcmp(j->path, "/v1/embeddings") && s->batch_size > 1) {
            if (s->batch_wait_us > 0)
                usleep((useconds_t)s->batch_wait_us);
            n_jobs += drain_ready_jobs(s, jobs + 1,
                                       PPLX_SERVER_MICROBATCH_MAX_JOBS - 1);
        }

        process_job_group(s, jobs, n_jobs);
    }
#ifdef USE_MLX
    if (s->use_mlx) {
        for (int i = 0; i < 4; i++) {
            loaded_model *m = &s->models[i];
            if (m->mlx_ctx) {
                pplx_mlx_free(m->mlx_ctx);
                m->mlx_ctx = NULL;
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
    int fd = anetTcpServer(err, port, (char *)host, PPLX_HTTP_BACKLOG);
    (void)portbuf;
    if (fd == ANET_ERR) {
        server_log("pplx-serve: listen failed on %s:%d: %s", host, port, err);
        return -1;
    }
    set_nonblock(fd);
    return fd;
}

static int load_one_model(http_server *s, model_slot slot, const char *path) {
    loaded_model *m = &s->models[slot];
    m->info = &k_models[slot];
    m->path = xstrdup(path);
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", path);
    m->tok = qwen_tokenizer_load(vocab_path);
    if (!m->tok) {
        server_log("pplx-serve: failed to load tokenizer: %s", vocab_path);
        return -1;
    }
    m->newline_ids = qwen_tokenizer_encode(m->tok, "\n", &m->newline_n);
    if (!m->newline_ids || m->newline_n <= 0) {
        server_log("pplx-serve: failed to tokenize newline separator");
        return -1;
    }
#ifdef USE_MLX
    if (s->use_mlx) {
        /* MLX streams are thread-local. The inference worker loads the MLX
         * model so all MLX arrays/streams are created, used, and freed by the
         * same thread. */
        return 0;
    }
#endif
    m->cpu_model = pplx_model_load(path);
    if (!m->cpu_model) {
        server_log("pplx-serve: failed to load model: %s", path);
        return -1;
    }
    m->cpu_ws = pplx_workspace_new(m->cpu_model);
    if (!m->cpu_ws) {
        server_log("pplx-serve: failed to allocate workspace: %s", path);
        return -1;
    }
    if (pplx_model_config(m->cpu_model)->hidden_size != m->info->dim) {
        server_log("pplx-serve: model %s has unexpected dimension %d",
                   m->info->id, pplx_model_config(m->cpu_model)->hidden_size);
        return -1;
    }
    return 0;
}

static void free_models(http_server *s) {
    for (int i = 0; i < 4; i++) {
        loaded_model *m = &s->models[i];
        free(m->path);
        free(m->newline_ids);
        if (m->tok) qwen_tokenizer_free(m->tok);
        if (m->cpu_ws) pplx_workspace_free(m->cpu_ws);
        if (m->cpu_model) pplx_model_free(m->cpu_model);
#ifdef USE_MLX
        if (m->mlx_ctx) pplx_mlx_free(m->mlx_ctx);
#endif
    }
}

int pplx_run_server(const pplx_server_config_t *cfg) {
    if (!cfg || !cfg->models || cfg->n_models <= 0) {
        fprintf(stderr, "pplx-serve: at least one --model MODEL_ID=PATH is required\n");
        return 1;
    }

    http_server s;
    memset(&s, 0, sizeof(s));
    s.host = cfg->host && cfg->host[0] ? cfg->host : "127.0.0.1";
    s.port = cfg->port > 0 ? cfg->port : 8000;
    s.batch_size = cfg->batch_size > 0
        ? cfg->batch_size : PPLX_SERVER_DEFAULT_BATCH_SIZE;
    s.batch_wait_us = cfg->batch_wait_us >= 0
        ? cfg->batch_wait_us : PPLX_SERVER_BATCH_WAIT_US;
    s.enable_cors = cfg->enable_cors;
    s.use_mlx = cfg->use_mlx;
    if (cfg->api_key && cfg->api_key[0])
        s.api_key = xstrdup(cfg->api_key);
    else {
        const char *env_key = getenv("PPLX_API_KEY");
        if (env_key && env_key[0]) s.api_key = xstrdup(env_key);
    }

    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);

    for (int i = 0; i < cfg->n_models; i++) {
        model_slot slot = model_slot_for_id(cfg->models[i].id);
        if (slot == MODEL_UNKNOWN) {
            fprintf(stderr, "pplx-serve: unknown model id: %s\n", cfg->models[i].id);
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

    s.loop = aeCreateEventLoop(PPLX_HTTP_SETSIZE);
    if (!s.loop) {
        fprintf(stderr, "pplx-serve: failed to create event loop\n");
        free_models(&s);
        return 1;
    }

    if (pthread_create(&s.worker, NULL, worker_main, &s) != 0) {
        fprintf(stderr, "pplx-serve: failed to start worker\n");
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
    aeCreateTimeEvent(s.loop, PPLX_HTTP_SWEEP_MS, timeout_cb, &s, NULL);

    server_log("pplx-serve: listening on http://%s:%d", s.host, s.port);
    aeMain(s.loop);

    server_log("pplx-serve: shutdown requested");
    if (s.listen_fd >= 0) close(s.listen_fd);
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
