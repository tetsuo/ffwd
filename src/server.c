#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "build.h"
#include "server.h"
#include "server_internal.h"
#include "server_util.h"
#include "sbuf.h"
#include "embed.h"
#include "safetensors.h"
#include "kernels.h"
#include "tokenizer_bpe.h"
#include "tokenizer_wordpiece.h"

#ifdef USE_MLX
#include "mlx.h"
#endif
#ifdef USE_CUDA
#include "cuda.h"
#endif

#include "deps/ae/ae.h"
#include "deps/ae/anet.h"

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
#include <sys/sysctl.h>
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
            n_threads = embed_get_num_cpus();
        embed_set_threads(n_threads);
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
