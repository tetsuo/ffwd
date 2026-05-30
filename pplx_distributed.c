/*
 * pplx_distributed.c - Stateless two-node layer-sharded inference.
 *
 * This first transport is deliberately synchronous: one coordinator sends a
 * packed fp32 hidden-state batch to one final worker and receives pooled
 * embeddings. The protocol is bounded and versioned so later route discovery,
 * retries, and fp16 transport can extend it without changing model math.
 */

#include "pplx_distributed.h"

#ifdef USE_MLX
#include "pplx_embed_mlx.h"
#endif

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define PPLX_DIST_MAGIC UINT32_C(0x504c5844) /* "PLXD" */
#define PPLX_DIST_VERSION 1u

#define PPLX_DIST_MSG_HELLO  1u
#define PPLX_DIST_MSG_WORK   2u
#define PPLX_DIST_MSG_RESULT 3u

#define PPLX_DIST_MAX_FRAME_BYTES (512u * 1024u * 1024u)
#define PPLX_DIST_MAX_BATCH 512u
#define PPLX_DIST_MAX_TOKENS 120000u
#define PPLX_DIST_MAX_ERROR_BYTES 1024u
#define PPLX_DIST_SOCKET_TIMEOUT_SEC 60

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t type;
    uint32_t body_bytes;
} dist_frame_header;

typedef struct {
    uint32_t signature_hi;
    uint32_t signature_lo;
    uint32_t hidden_size;
    uint32_t n_layers;
    uint32_t vocab_size;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t layer_start;
    uint32_t layer_end;
} dist_hello;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t batch;
    uint32_t total_tokens;
    uint32_t hidden_size;
    uint32_t lengths_bytes;
    uint32_t states_bytes;
} dist_work;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t status;
    uint32_t batch;
    uint32_t dimensions;
    uint32_t payload_bytes;
} dist_result;

typedef struct {
    pplx_model_t *model;
    pplx_workspace_t *workspace;
#ifdef USE_MLX
    pplx_mlx_ctx_t *mlx;
#endif
    const pplx_config_t *cfg;
} dist_stage;

struct pplx_dist_client {
    dist_stage local;
    int fd;
    int local_layer_end;
    uint64_t next_request_id;
};

static int host_is_little_endian(void)
{
    const uint16_t one = 1;
    return *(const unsigned char *)&one == 1;
}

static uint64_t fnv_u32(uint64_t h, uint32_t v)
{
    for (int i = 0; i < 4; i++) {
        h ^= (unsigned char)(v >> (i * 8));
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t config_signature(const pplx_config_t *cfg)
{
    uint64_t h = UINT64_C(1469598103934665603);
    uint32_t eps = 0, theta = 0;
    memcpy(&eps, &cfg->rms_norm_eps, sizeof(eps));
    memcpy(&theta, &cfg->rope_theta, sizeof(theta));
    h = fnv_u32(h, (uint32_t)cfg->hidden_size);
    h = fnv_u32(h, (uint32_t)cfg->n_layers);
    h = fnv_u32(h, (uint32_t)cfg->n_heads);
    h = fnv_u32(h, (uint32_t)cfg->n_kv_heads);
    h = fnv_u32(h, (uint32_t)cfg->head_dim);
    h = fnv_u32(h, (uint32_t)cfg->intermediate_size);
    h = fnv_u32(h, (uint32_t)cfg->vocab_size);
    h = fnv_u32(h, eps);
    h = fnv_u32(h, theta);
    return h;
}

static void u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo)
{
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static uint64_t u64_from_halves(uint32_t hi, uint32_t lo)
{
    return ((uint64_t)hi << 32) | lo;
}

static int mul_u32(uint32_t a, uint32_t b, uint32_t *out)
{
    if (a != 0 && b > UINT32_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static void frame_to_wire(dist_frame_header *h)
{
    h->magic = htonl(h->magic);
    h->version = htonl(h->version);
    h->type = htonl(h->type);
    h->body_bytes = htonl(h->body_bytes);
}

static void hello_to_wire(dist_hello *h)
{
    uint32_t *v = (uint32_t *)h;
    for (size_t i = 0; i < sizeof(*h) / sizeof(*v); i++) v[i] = htonl(v[i]);
}

static void hello_from_wire(dist_hello *h)
{
    uint32_t *v = (uint32_t *)h;
    for (size_t i = 0; i < sizeof(*h) / sizeof(*v); i++) v[i] = ntohl(v[i]);
}

static void work_to_wire(dist_work *w)
{
    uint32_t *v = (uint32_t *)w;
    for (size_t i = 0; i < sizeof(*w) / sizeof(*v); i++) v[i] = htonl(v[i]);
}

static void work_from_wire(dist_work *w)
{
    uint32_t *v = (uint32_t *)w;
    for (size_t i = 0; i < sizeof(*w) / sizeof(*v); i++) v[i] = ntohl(v[i]);
}

static void result_to_wire(dist_result *r)
{
    uint32_t *v = (uint32_t *)r;
    for (size_t i = 0; i < sizeof(*r) / sizeof(*v); i++) v[i] = htonl(v[i]);
}

static void result_from_wire(dist_result *r)
{
    uint32_t *v = (uint32_t *)r;
    for (size_t i = 0; i < sizeof(*r) / sizeof(*v); i++) v[i] = ntohl(v[i]);
}

static int write_full(int fd, const void *buf, size_t len)
{
    const unsigned char *p = buf;
    while (len) {
#ifdef MSG_NOSIGNAL
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t n = send(fd, p, len, 0);
#endif
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t len)
{
    unsigned char *p = buf;
    while (len) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int write_frame_header(int fd, uint32_t type, uint32_t body_bytes)
{
    dist_frame_header h = {PPLX_DIST_MAGIC, PPLX_DIST_VERSION, type, body_bytes};
    frame_to_wire(&h);
    return write_full(fd, &h, sizeof(h));
}

static int read_frame_header(int fd, uint32_t *type, uint32_t *body_bytes)
{
    dist_frame_header h;
    if (read_full(fd, &h, sizeof(h)) != 0) return -1;
    h.magic = ntohl(h.magic);
    h.version = ntohl(h.version);
    h.type = ntohl(h.type);
    h.body_bytes = ntohl(h.body_bytes);
    if (h.magic != PPLX_DIST_MAGIC || h.version != PPLX_DIST_VERSION ||
        h.body_bytes > PPLX_DIST_MAX_FRAME_BYTES)
        return -1;
    *type = h.type;
    *body_bytes = h.body_bytes;
    return 0;
}

static void set_socket_options(int fd)
{
    int yes = 1;
    struct timeval tv = {PPLX_DIST_SOCKET_TIMEOUT_SEC, 0};
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
}

static int open_listener(const char *host, int port)
{
    char service[16];
    snprintf(service, sizeof(service), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo *list = NULL;
    if (getaddrinfo(host, service, &hints, &list) != 0) return -1;

    int listen_fd = -1;
    for (struct addrinfo *ai = list; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int yes = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 &&
            listen(fd, 16) == 0) {
            listen_fd = fd;
            break;
        }
        close(fd);
    }
    freeaddrinfo(list);
    return listen_fd;
}

static int connect_endpoint(const char *host, int port)
{
    char service[16];
    snprintf(service, sizeof(service), "%d", port);
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *list = NULL;
    if (getaddrinfo(host, service, &hints, &list) != 0) return -1;

    int fd = -1;
    for (struct addrinfo *ai = list; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        set_socket_options(fd);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(list);
    return fd;
}

static int stage_load(dist_stage *stage, const char *model_dir,
                      int layer_start, int layer_end, int use_mlx)
{
    memset(stage, 0, sizeof(*stage));
#ifdef USE_MLX
    if (use_mlx) {
        stage->mlx = pplx_mlx_load_slice(model_dir, layer_start, layer_end);
        if (!stage->mlx) return -1;
        stage->cfg = pplx_mlx_config(stage->mlx);
        return 0;
    }
#else
    if (use_mlx) return -1;
#endif
    stage->model = pplx_model_load_slice(model_dir, layer_start, layer_end);
    if (!stage->model) return -1;
    stage->workspace = pplx_workspace_new(stage->model);
    if (!stage->workspace) {
        pplx_model_free(stage->model);
        stage->model = NULL;
        return -1;
    }
    stage->cfg = pplx_model_config(stage->model);
    return 0;
}

static void stage_free(dist_stage *stage)
{
    if (!stage) return;
    if (stage->workspace) pplx_workspace_free(stage->workspace);
    if (stage->model) pplx_model_free(stage->model);
#ifdef USE_MLX
    if (stage->mlx) pplx_mlx_free(stage->mlx);
#endif
    memset(stage, 0, sizeof(*stage));
}

static int stage_forward(dist_stage *stage, const pplx_input_t *inputs,
                         int batch, const float *input_states,
                         int layer_start, int layer_end, int apply_final_norm,
                         float *out_states)
{
#ifdef USE_MLX
    if (stage->mlx)
        return pplx_mlx_forward_slice_batch(stage->mlx, inputs, batch,
                                            input_states, layer_start,
                                            layer_end, apply_final_norm,
                                            out_states);
#endif
    return pplx_model_forward_slice_batch(stage->model, stage->workspace,
                                          inputs, batch, input_states,
                                          layer_start, layer_end,
                                          apply_final_norm, out_states);
}

static int send_hello(int fd, const pplx_config_t *cfg,
                      int layer_start, int layer_end)
{
    uint64_t sig = config_signature(cfg);
    dist_hello h;
    u64_to_halves(sig, &h.signature_hi, &h.signature_lo);
    h.hidden_size = (uint32_t)cfg->hidden_size;
    h.n_layers = (uint32_t)cfg->n_layers;
    h.vocab_size = (uint32_t)cfg->vocab_size;
    h.n_heads = (uint32_t)cfg->n_heads;
    h.n_kv_heads = (uint32_t)cfg->n_kv_heads;
    h.head_dim = (uint32_t)cfg->head_dim;
    h.intermediate_size = (uint32_t)cfg->intermediate_size;
    h.layer_start = (uint32_t)layer_start;
    h.layer_end = (uint32_t)layer_end;
    hello_to_wire(&h);
    if (write_frame_header(fd, PPLX_DIST_MSG_HELLO, sizeof(h)) != 0)
        return -1;
    return write_full(fd, &h, sizeof(h));
}

static int recv_hello(int fd, const pplx_config_t *cfg, int expected_start)
{
    uint32_t type = 0, bytes = 0;
    dist_hello h;
    if (read_frame_header(fd, &type, &bytes) != 0 ||
        type != PPLX_DIST_MSG_HELLO || bytes != sizeof(h) ||
        read_full(fd, &h, sizeof(h)) != 0)
        return -1;
    hello_from_wire(&h);
    if (u64_from_halves(h.signature_hi, h.signature_lo) !=
            config_signature(cfg) ||
        h.hidden_size != (uint32_t)cfg->hidden_size ||
        h.n_layers != (uint32_t)cfg->n_layers ||
        h.layer_start != (uint32_t)expected_start ||
        h.layer_end != (uint32_t)cfg->n_layers)
        return -1;
    return 0;
}

static int send_error_result(int fd, uint64_t request_id, const char *msg)
{
    size_t len = msg ? strlen(msg) : 0;
    if (len > PPLX_DIST_MAX_ERROR_BYTES) len = PPLX_DIST_MAX_ERROR_BYTES;
    dist_result result;
    memset(&result, 0, sizeof(result));
    u64_to_halves(request_id, &result.request_hi, &result.request_lo);
    result.status = 1;
    result.payload_bytes = (uint32_t)len;
    dist_result wire = result;
    result_to_wire(&wire);
    if (write_frame_header(fd, PPLX_DIST_MSG_RESULT,
                           (uint32_t)sizeof(wire) + (uint32_t)len) != 0 ||
        write_full(fd, &wire, sizeof(wire)) != 0)
        return -1;
    return len ? write_full(fd, msg, len) : 0;
}

static int worker_process_work(int fd, dist_stage *stage,
                               int layer_start, int layer_end,
                               const void *body, uint32_t body_bytes)
{
    uint64_t request_id = 0;
    if (body_bytes < sizeof(dist_work))
        return send_error_result(fd, request_id, "truncated WORK frame");

    dist_work work;
    memcpy(&work, body, sizeof(work));
    work_from_wire(&work);
    request_id = u64_from_halves(work.request_hi, work.request_lo);

    uint32_t expected_lengths = 0, expected_states = 0;
    if (work.batch == 0 || work.batch > PPLX_DIST_MAX_BATCH ||
        work.total_tokens == 0 || work.total_tokens > PPLX_DIST_MAX_TOKENS ||
        work.hidden_size != (uint32_t)stage->cfg->hidden_size ||
        mul_u32(work.batch, sizeof(uint32_t), &expected_lengths) != 0 ||
        mul_u32(work.total_tokens, work.hidden_size, &expected_states) != 0 ||
        mul_u32(expected_states, sizeof(float), &expected_states) != 0 ||
        work.lengths_bytes != expected_lengths ||
        work.states_bytes != expected_states ||
        body_bytes != sizeof(work) + expected_lengths + expected_states)
        return send_error_result(fd, request_id, "invalid WORK frame");

    const unsigned char *p = (const unsigned char *)body + sizeof(work);
    int *lengths = malloc((size_t)work.batch * sizeof(*lengths));
    pplx_input_t *inputs = calloc(work.batch, sizeof(*inputs));
    float *states = malloc(expected_states);
    float *out_states = malloc(expected_states);
    uint32_t emb_bytes = 0;
    if (mul_u32(work.batch, work.hidden_size, &emb_bytes) != 0 ||
        mul_u32(emb_bytes, sizeof(float), &emb_bytes) != 0) {
        free(lengths); free(inputs); free(states); free(out_states);
        return send_error_result(fd, request_id, "embedding result is too large");
    }
    float *embeddings = malloc(emb_bytes);
    if (!lengths || !inputs || !states || !out_states || !embeddings) {
        free(lengths); free(inputs); free(states); free(out_states);
        free(embeddings);
        return send_error_result(fd, request_id, "out of memory");
    }

    uint32_t total = 0;
    for (uint32_t i = 0; i < work.batch; i++) {
        uint32_t n = 0;
        memcpy(&n, p + (size_t)i * sizeof(n), sizeof(n));
        n = ntohl(n);
        if (n == 0 || n > PPLX_DIST_MAX_TOKENS - total) {
            free(lengths); free(inputs); free(states); free(out_states);
            free(embeddings);
            return send_error_result(fd, request_id, "invalid sequence length");
        }
        lengths[i] = (int)n;
        inputs[i].n_tokens = (int)n;
        total += n;
    }
    if (total != work.total_tokens) {
        free(lengths); free(inputs); free(states); free(out_states);
        free(embeddings);
        return send_error_result(fd, request_id, "sequence lengths do not match token total");
    }
    memcpy(states, p + expected_lengths, expected_states);

    int rc = stage_forward(stage, inputs, (int)work.batch, states,
                           layer_start, layer_end, 1, out_states);
    if (rc == 0)
        rc = pplx_pool_batch(stage->cfg, out_states, lengths,
                             (int)work.batch, embeddings);
    free(lengths); free(inputs); free(states); free(out_states);
    if (rc != 0) {
        free(embeddings);
        return send_error_result(fd, request_id, "worker inference failed");
    }

    dist_result result;
    memset(&result, 0, sizeof(result));
    u64_to_halves(request_id, &result.request_hi, &result.request_lo);
    result.batch = work.batch;
    result.dimensions = work.hidden_size;
    result.payload_bytes = emb_bytes;
    dist_result wire = result;
    result_to_wire(&wire);
    rc = write_frame_header(fd, PPLX_DIST_MSG_RESULT,
                            (uint32_t)sizeof(wire) + emb_bytes);
    if (rc == 0) rc = write_full(fd, &wire, sizeof(wire));
    if (rc == 0) rc = write_full(fd, embeddings, emb_bytes);
    free(embeddings);
    return rc;
}

static int worker_serve_connection(int fd, dist_stage *stage,
                                   int layer_start, int layer_end)
{
    if (send_hello(fd, stage->cfg, layer_start, layer_end) != 0) return -1;
    for (;;) {
        uint32_t type = 0, bytes = 0;
        if (read_frame_header(fd, &type, &bytes) != 0) return -1;
        if (type != PPLX_DIST_MSG_WORK || bytes == 0) return -1;
        void *body = malloc(bytes);
        if (!body) return -1;
        int rc = read_full(fd, body, bytes);
        if (rc == 0)
            rc = worker_process_work(fd, stage, layer_start, layer_end,
                                     body, bytes);
        free(body);
        if (rc != 0) return -1;
    }
}

int pplx_dist_run_worker(const char *model_dir, const char *host, int port,
                         int layer_start, int layer_end, int use_mlx)
{
    if (!model_dir || !host || port <= 0 || port > 65535 ||
        !host_is_little_endian()) {
        fprintf(stderr, "pplx-dist: invalid worker configuration\n");
        return 1;
    }
    dist_stage stage;
    if (stage_load(&stage, model_dir, layer_start, layer_end, use_mlx) != 0) {
        fprintf(stderr, "pplx-dist: failed to load worker shard\n");
        return 1;
    }
    if (layer_end != stage.cfg->n_layers) {
        fprintf(stderr, "pplx-dist: initial worker must own final layer %d\n",
                stage.cfg->n_layers);
        stage_free(&stage);
        return 1;
    }
    int listen_fd = open_listener(host, port);
    if (listen_fd < 0) {
        fprintf(stderr, "pplx-dist: listen failed on %s:%d: %s\n",
                host, port, strerror(errno));
        stage_free(&stage);
        return 1;
    }
    fprintf(stderr, "pplx-dist: worker listening on %s:%d layers=[%d,%d)\n",
            host, port, layer_start, layer_end);
    for (;;) {
        int fd = accept(listen_fd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "pplx-dist: accept failed: %s\n", strerror(errno));
            break;
        }
        set_socket_options(fd);
        if (pplx_verbose >= 1) fprintf(stderr, "pplx-dist: coordinator connected\n");
        worker_serve_connection(fd, &stage, layer_start, layer_end);
        close(fd);
        if (pplx_verbose >= 1) fprintf(stderr, "pplx-dist: coordinator disconnected\n");
    }
    close(listen_fd);
    stage_free(&stage);
    return 1;
}

pplx_dist_client_t *pplx_dist_client_open(const char *model_dir,
                                          const char *host, int port,
                                          int local_layer_end, int use_mlx)
{
    if (!model_dir || !host || port <= 0 || port > 65535 ||
        local_layer_end <= 0 || !host_is_little_endian())
        return NULL;

    pplx_dist_client_t *client = calloc(1, sizeof(*client));
    if (!client) return NULL;
    client->fd = -1;
    if (stage_load(&client->local, model_dir, 0, local_layer_end, use_mlx) != 0)
        goto fail;
    client->fd = connect_endpoint(host, port);
    if (client->fd < 0 ||
        recv_hello(client->fd, client->local.cfg, local_layer_end) != 0)
        goto fail;
    client->local_layer_end = local_layer_end;
    client->next_request_id = 1;
    return client;

fail:
    pplx_dist_client_free(client);
    return NULL;
}

void pplx_dist_client_free(pplx_dist_client_t *client)
{
    if (!client) return;
    if (client->fd >= 0) close(client->fd);
    stage_free(&client->local);
    free(client);
}

const pplx_config_t *pplx_dist_client_config(const pplx_dist_client_t *client)
{
    return client ? client->local.cfg : NULL;
}

static int client_transport_error(pplx_dist_client_t *client)
{
    if (client->fd >= 0) {
        close(client->fd);
        client->fd = -1;
    }
    return -1;
}

int pplx_dist_client_embed_batch(pplx_dist_client_t *client,
                                 const pplx_input_t *inputs, int batch,
                                 float *out_embeddings)
{
    if (!client || client->fd < 0 || !inputs || batch <= 0 ||
        batch > (int)PPLX_DIST_MAX_BATCH || !out_embeddings)
        return -1;

    uint32_t total_tokens = 0;
    for (int i = 0; i < batch; i++) {
        if (!inputs[i].ids || inputs[i].n_tokens <= 0 ||
            (uint32_t)inputs[i].n_tokens > PPLX_DIST_MAX_TOKENS - total_tokens)
            return -1;
        total_tokens += (uint32_t)inputs[i].n_tokens;
    }

    uint32_t state_bytes = 0, lengths_bytes = 0;
    uint32_t hidden = (uint32_t)client->local.cfg->hidden_size;
    if (mul_u32(total_tokens, hidden, &state_bytes) != 0 ||
        mul_u32(state_bytes, sizeof(float), &state_bytes) != 0 ||
        mul_u32((uint32_t)batch, sizeof(uint32_t), &lengths_bytes) != 0)
        return -1;
    uint32_t body_bytes = (uint32_t)sizeof(dist_work) + lengths_bytes +
                          state_bytes;
    if (body_bytes > PPLX_DIST_MAX_FRAME_BYTES)
        return -1;
    float *states = malloc(state_bytes);
    if (!states) return -1;
    int rc = stage_forward(&client->local, inputs, batch, NULL, 0,
                           client->local_layer_end, 0, states);
    if (rc != 0) {
        free(states);
        return -1;
    }

    uint64_t request_id = client->next_request_id++;
    dist_work work;
    u64_to_halves(request_id, &work.request_hi, &work.request_lo);
    work.batch = (uint32_t)batch;
    work.total_tokens = total_tokens;
    work.hidden_size = hidden;
    work.lengths_bytes = lengths_bytes;
    work.states_bytes = state_bytes;
    dist_work wire = work;
    work_to_wire(&wire);
    rc = write_frame_header(client->fd, PPLX_DIST_MSG_WORK, body_bytes);
    if (rc == 0) rc = write_full(client->fd, &wire, sizeof(wire));
    for (int i = 0; rc == 0 && i < batch; i++) {
        uint32_t n = htonl((uint32_t)inputs[i].n_tokens);
        rc = write_full(client->fd, &n, sizeof(n));
    }
    if (rc == 0) rc = write_full(client->fd, states, state_bytes);
    free(states);
    if (rc != 0) return client_transport_error(client);

    uint32_t type = 0, bytes = 0;
    dist_result result;
    if (read_frame_header(client->fd, &type, &bytes) != 0 ||
        type != PPLX_DIST_MSG_RESULT || bytes < sizeof(result) ||
        read_full(client->fd, &result, sizeof(result)) != 0)
        return client_transport_error(client);
    result_from_wire(&result);
    uint32_t payload_bytes = bytes - (uint32_t)sizeof(result);
    if (result.payload_bytes != payload_bytes ||
        u64_from_halves(result.request_hi, result.request_lo) != request_id)
        return client_transport_error(client);
    if (result.status != 0) {
        char error[PPLX_DIST_MAX_ERROR_BYTES + 1];
        if (payload_bytes > PPLX_DIST_MAX_ERROR_BYTES ||
            read_full(client->fd, error, payload_bytes) != 0)
            return client_transport_error(client);
        error[payload_bytes] = '\0';
        fprintf(stderr, "pplx-dist: worker error: %s\n", error);
        return client_transport_error(client);
    }
    uint32_t expected = 0;
    if (result.batch != (uint32_t)batch || result.dimensions != hidden ||
        mul_u32((uint32_t)batch, hidden, &expected) != 0 ||
        mul_u32(expected, sizeof(float), &expected) != 0 ||
        payload_bytes != expected ||
        read_full(client->fd, out_embeddings, payload_bytes) != 0)
        return client_transport_error(client);
    return 0;
}
