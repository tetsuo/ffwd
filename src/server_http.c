#include "server_internal.h"
#include "server_util.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <unistd.h>

/* ========================================================================
 * HTTP/1.1 connection lifecycle, header parsing, and response framing
 *
 * One client per connection, refcounted: the event loop holds a reference
 * while the socket is open and the worker thread holds one while a job derived
 * from the connection is in flight, so a request that completes after the peer
 * disconnects frees the client safely. read_cb accumulates bytes until a full
 * request is framed, then hands off to dispatch_request (server.c); write_cb
 * drains the response and either closes or arms the next pipelined request.
 * ======================================================================== */

int set_nonblock(int fd) {
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

void client_incref(client *c) {
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

void client_decref_n(client *c, int n) {
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

int close_client_unlink(client *c) {
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

void close_client(client *c) {
    if (!close_client_unlink(c))
        return;
    client_decref(c);
}

void append_http_response_ex(client *c,
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

void append_http_response(
    client *c, int status, const char *ctype, const char *body, size_t body_len) {
    append_http_response_ex(c, status, ctype, NULL, body, body_len);
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

int queue_write(client *c) {
    if (!c || c->cancelled || c->fd < 0)
        return -1;
    return aeCreateFileEvent(c->srv->loop, c->fd, AE_WRITABLE, write_cb, c);
}

bool route_is_inference(const char *method, const char *path) {
    return !strcmp(method, "POST") &&
           (!strcmp(path, "/v1/embeddings") || !strcmp(path, "/v1/contextualizedembeddings") ||
            !strcmp(path, "/v1/rerank"));
}

void read_cb(aeEventLoop *loop, int fd, void *clientData, int mask) {
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
