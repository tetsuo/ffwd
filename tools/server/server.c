#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include "server.h"
#include "server_internal.h"
#include "util.h"
#include "sbuf.h"
#include "ffwd.h"
#include "ae.h"
#include "anet.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
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
#    include <sys/sysctl.h>
#endif

#include <cjson/cJSON.h>

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

static bool auth_ok(http_server *s, const char *auth) {
    if (!s->api_key || !s->api_key[0])
        return true;
    if (!auth)
        return false;
    const char prefix[] = "Bearer ";
    return !strncmp(auth, prefix, sizeof(prefix) - 1) &&
           !strcmp(auth + sizeof(prefix) - 1, s->api_key);
}

void dispatch_request(client *c) {
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
        char *body = json_error_body("missing or invalid bearer token", "authentication_error", &len);
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
    /* Tokenize off the worker when the pipeline is already busy so it overlaps
     * in-flight work; tokenize inline on the worker when idle to avoid a
     * queue hand-off that single-stream requests gain nothing from. */
    if (server_has_backlog(c->srv))
        enqueue_raw_job(j);
    else
        enqueue_job(j);
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
            now - c->last_active_ms > FFWD_HTTP_CLIENT_TIMEOUT_MS)
            close_client(c);
        c = next;
    }
    return FFWD_HTTP_SWEEP_MS;
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

static void *worker_main(void *arg) {
    http_server *s = arg;
    /* Create each model's context on this thread (a GPU build defers the load
     * here so the context lives on the thread that uses it) and finalize the
     * config a GPU build could not finalize at open time. CPU is loaded and
     * finalized already, marked by info->dim being set. */
    int rc = 0;
    char load_err[256];
    for (int i = 0; i < s->n_models; i++) {
        loaded_model *m = &s->models[i];
        if (ffwd_activate(m->backend, load_err, sizeof(load_err)) != 0) {
            server_log("ffwd-server: failed to load model on worker: %s: %s", m->path, load_err);
            rc = 1;
            break;
        }
        if (m->info->dim == 0 && finalize_loaded_model(m) != 0) {
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
        for (int i = 0; i < s->n_models; i++)
            ffwd_worker_free(s->models[i].backend);
        return NULL;
    }

    for (;;) {
        job *jobs[FFWD_SERVER_MICROBATCH_MAX_JOBS];
        int n_jobs = collect_job_batch(s, jobs, FFWD_SERVER_MICROBATCH_MAX_JOBS);
        if (n_jobs == 0)
            break;
        process_job_group(s, jobs, n_jobs);
    }
    for (int i = 0; i < s->n_models; i++)
        ffwd_worker_free(s->models[i].backend);
    return NULL;
}

/* Tokenizer stage: parse + tokenize raw jobs off the worker, then feed the
 * worker's job queue. Waits for the worker to finish model init first, because
 * a GPU build finalizes the model config (dims, pooling) on the worker and
 * request validation reads it. Idle while no request is routed through it. */
static void *tokenizer_main(void *arg) {
    http_server *s = arg;
    pthread_mutex_lock(&s->mu);
    while (!s->worker_ready)
        pthread_cond_wait(&s->cv, &s->mu);
    int rc = s->worker_init_rc;
    pthread_mutex_unlock(&s->mu);
    if (rc)
        return NULL;

    for (;;) {
        job *j = dequeue_raw_job(s);
        if (!j)
            break;
        tokenize_job(s, j);
        enqueue_job(j);
    }
    return NULL;
}

static void *renderer_main(void *arg) {
    http_server *s = arg;
    for (;;) {
        job *j = dequeue_render_job(s);
        if (!j)
            break;

        uint64_t t0 = nstime();
        if (render_job_response(j) != 0)
            job_set_error(j, 500, "embedding normalization failed", "server_error");
        j->encode_ns += nstime() - t0;
        job_render_free(j);
        finish_job(j);
    }
    return NULL;
}

static int listen_on(const char *host, int port) {
    char portbuf[32];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    char err[ANET_ERR_LEN];
    int fd = anetTcpServer(err, port, (char *)host, FFWD_HTTP_BACKLOG);
    (void)portbuf;
    if (fd == ANET_ERR) {
        server_log("ffwd-server: listen failed on %s:%d: %s", host, port, err);
        return -1;
    }
    set_nonblock(fd);
    return fd;
}

static int validate_model_specs(const ffwd_server_config_t *cfg) {
    for (int i = 0; i < cfg->n_models; i++) {
        const ffwd_server_model_spec_t *spec = &cfg->models[i];
        if (!spec->id || !spec->id[0]) {
            fprintf(stderr, "ffwd-server: model label must not be empty\n");
            return -1;
        }
        if (!spec->path || !spec->path[0]) {
            fprintf(stderr, "ffwd-server: model %s has an empty path\n", spec->id);
            return -1;
        }
        if (spec->kind < FFWD_SERVER_MODEL_STANDARD || spec->kind > FFWD_SERVER_MODEL_LATE) {
            fprintf(stderr, "ffwd-server: model %s has an invalid kind\n", spec->id);
            return -1;
        }
        if (spec->api < FFWD_SERVER_API_DEFAULT || spec->api > FFWD_SERVER_API_PERPLEXITY) {
            fprintf(stderr, "ffwd-server: model %s has an invalid api\n", spec->id);
            return -1;
        }
        if (spec->min_dim < 0) {
            fprintf(stderr, "ffwd-server: model %s has an invalid min_dim\n", spec->id);
            return -1;
        }
        for (int j = 0; j < i; j++) {
            if (spec->id && cfg->models[j].id && !strcmp(cfg->models[j].id, spec->id)) {
                fprintf(stderr, "ffwd-server: duplicate model label: %s\n", spec->id);
                return -1;
            }
        }
    }
    return 0;
}

/* Hold the model paths in one array so the backend preflight can size the set
 * against host memory in a single pass. */
static int run_memory_preflight(const ffwd_server_config_t *cfg, const ffwd_options_t *opts) {
    const char **paths = (const char **)malloc((size_t)cfg->n_models * sizeof(*paths));
    if (!paths)
        return -1;
    for (int i = 0; i < cfg->n_models; i++)
        paths[i] = cfg->models[i].path;
    char err[256] = {0};
    int rc = ffwd_preflight(paths, cfg->n_models, opts, err, sizeof(err));
    free(paths);
    if (rc != 0 && err[0])
        server_log("ffwd-server: %s", err);
    return rc;
}

int ffwd_run_server(const ffwd_server_config_t *cfg) {
    if (!cfg || !cfg->models || cfg->n_models <= 0) {
        fprintf(stderr, "ffwd-server: at least one model label and path is required\n");
        return 1;
    }
    ffwd_options_t backend_opts = {
        .gpu_quant_bits = cfg->gpu_quantize_bits,
        .gpu_quant_group_size = cfg->gpu_quantize_group_size > 0 ? cfg->gpu_quantize_group_size : 64,
        .memory_utilization = cfg->memory_utilization,
    };
    if (validate_model_specs(cfg) != 0 || run_memory_preflight(cfg, &backend_opts) != 0)
        return 1;

    http_server s;
    memset(&s, 0, sizeof(s));
    s.host = cfg->host && cfg->host[0] ? cfg->host : "127.0.0.1";
    s.port = cfg->port > 0 ? cfg->port : 8000;
    s.batch_size = cfg->batch_size > 0 ? cfg->batch_size : FFWD_SERVER_DEFAULT_BATCH_SIZE;
    s.max_batch_tokens =
        cfg->max_batch_tokens > 0 ? cfg->max_batch_tokens : FFWD_SERVER_DEFAULT_MAX_BATCH_TOKENS;
    s.batch_wait_us = cfg->batch_wait_us >= 0 ? cfg->batch_wait_us : ffwd_default_batch_wait_us();
    s.backend_opts = backend_opts;
    if (cfg->api_key && cfg->api_key[0])
        s.api_key = xstrdup(cfg->api_key);
    else {
        const char *env_key = getenv("FFWD_API_KEY");
        if (env_key && env_key[0])
            s.api_key = xstrdup(env_key);
    }

    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.raw_cv, NULL);
    pthread_cond_init(&s.render_cv, NULL);

    for (int i = 0; i < cfg->n_models; i++) {
        if (load_one_model(&s, &cfg->models[i]) != 0) {
            free_models(&s);
            free(s.api_key);
            pthread_cond_destroy(&s.render_cv);
            pthread_cond_destroy(&s.raw_cv);
            pthread_cond_destroy(&s.cv);
            pthread_mutex_destroy(&s.mu);
            return 1;
        }
    }

    if (pipe(s.completion_pipe) != 0 || pipe(s.signal_pipe) != 0) {
        perror("pipe");
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.render_cv);
        pthread_cond_destroy(&s.raw_cv);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
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

    s.loop = aeCreateEventLoop(FFWD_HTTP_SETSIZE);
    if (!s.loop) {
        fprintf(stderr, "ffwd-server: failed to create event loop\n");
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.render_cv);
        pthread_cond_destroy(&s.raw_cv);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
        return 1;
    }

    if (pthread_create(&s.renderer, NULL, renderer_main, &s) != 0) {
        fprintf(stderr, "ffwd-server: failed to start renderer\n");
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.render_cv);
        pthread_cond_destroy(&s.raw_cv);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
        return 1;
    }

    if (pthread_create(&s.worker, NULL, worker_main, &s) != 0) {
        fprintf(stderr, "ffwd-server: failed to start worker\n");
        pthread_mutex_lock(&s.mu);
        s.render_stopping = 1;
        pthread_cond_signal(&s.render_cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.renderer, NULL);
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.render_cv);
        pthread_cond_destroy(&s.raw_cv);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
        return 1;
    }
    pthread_mutex_lock(&s.mu);
    while (!s.worker_ready)
        pthread_cond_wait(&s.cv, &s.mu);
    int worker_init_rc = s.worker_init_rc;
    pthread_mutex_unlock(&s.mu);
    if (worker_init_rc) {
        pthread_mutex_lock(&s.mu);
        s.stopping = 1;
        pthread_cond_signal(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.worker, NULL);
        pthread_mutex_lock(&s.mu);
        s.render_stopping = 1;
        pthread_cond_signal(&s.render_cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.renderer, NULL);
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.render_cv);
        pthread_cond_destroy(&s.raw_cv);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
        return 1;
    }

    /* Started after worker init so the tokenizer sees a finalized model config
     * (a GPU build sets dims/pooling on the worker, and request validation reads
     * them). Joined before the worker at shutdown so it drains raw jobs into the
     * job queue first. */
    if (pthread_create(&s.tokenizer, NULL, tokenizer_main, &s) != 0) {
        fprintf(stderr, "ffwd-server: failed to start tokenizer\n");
        pthread_mutex_lock(&s.mu);
        s.stopping = 1;
        pthread_cond_signal(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.worker, NULL);
        pthread_mutex_lock(&s.mu);
        s.render_stopping = 1;
        pthread_cond_signal(&s.render_cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.renderer, NULL);
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.render_cv);
        pthread_cond_destroy(&s.raw_cv);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
        return 1;
    }

    s.listen_fd = listen_on(s.host, s.port);
    if (s.listen_fd < 0) {
        pthread_mutex_lock(&s.mu);
        s.raw_stopping = 1;
        pthread_cond_signal(&s.raw_cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.tokenizer, NULL);
        pthread_mutex_lock(&s.mu);
        s.stopping = 1;
        pthread_cond_signal(&s.cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.worker, NULL);
        pthread_mutex_lock(&s.mu);
        s.render_stopping = 1;
        pthread_cond_signal(&s.render_cv);
        pthread_mutex_unlock(&s.mu);
        pthread_join(s.renderer, NULL);
        aeDeleteEventLoop(s.loop);
        free_models(&s);
        free(s.api_key);
        pthread_cond_destroy(&s.render_cv);
        pthread_cond_destroy(&s.raw_cv);
        pthread_cond_destroy(&s.cv);
        pthread_mutex_destroy(&s.mu);
        return 1;
    }

    aeCreateFileEvent(s.loop, s.listen_fd, AE_READABLE, accept_cb, &s);
    aeCreateFileEvent(s.loop, s.completion_pipe[0], AE_READABLE, completion_cb, &s);
    aeCreateFileEvent(s.loop, s.signal_pipe[0], AE_READABLE, signal_cb, &s);
    aeCreateTimeEvent(s.loop, FFWD_HTTP_SWEEP_MS, timeout_cb, &s, NULL);

    server_log("ffwd-server: listening on http://%s:%d", s.host, s.port);
    aeMain(s.loop);

    server_log("ffwd-server: shutdown requested");
    if (s.listen_fd >= 0)
        close(s.listen_fd);
    client *c = s.clients;
    while (c) {
        client *next = c->next;
        close_client(c);
        c = next;
    }

    /* Drain stages front to back: the tokenizer flushes raw jobs into the job
     * queue, then the worker drains that, then the renderer drains finished
     * jobs, so nothing is stranded mid-pipeline. */
    pthread_mutex_lock(&s.mu);
    s.raw_stopping = 1;
    pthread_cond_signal(&s.raw_cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(s.tokenizer, NULL);
    pthread_mutex_lock(&s.mu);
    s.stopping = 1;
    pthread_cond_signal(&s.cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(s.worker, NULL);
    pthread_mutex_lock(&s.mu);
    s.render_stopping = 1;
    pthread_cond_signal(&s.render_cv);
    pthread_mutex_unlock(&s.mu);
    pthread_join(s.renderer, NULL);

    aeDeleteEventLoop(s.loop);
    close(s.completion_pipe[0]);
    close(s.completion_pipe[1]);
    close(s.signal_pipe[0]);
    close(s.signal_pipe[1]);
    free_models(&s);
    free(s.api_key);
    pthread_cond_destroy(&s.render_cv);
    pthread_cond_destroy(&s.raw_cv);
    pthread_cond_destroy(&s.cv);
    pthread_mutex_destroy(&s.mu);
    return 0;
}
