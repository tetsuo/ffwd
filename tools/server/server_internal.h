#ifndef FFWD_SERVER_INTERNAL_H
#define FFWD_SERVER_INTERNAL_H

/* Declarations shared across the ffwd-server translation units (HTTP framing,
 * JSON validation, output encoding, scheduling, model lifecycle, endpoint
 * handlers). The server is split by concern but compiles as several objects
 * linked into one binary; this header is what they share. Not a public API -
 * the CLI and library do not include it. */

#include "ffwd.h"
#include "server.h"
#include "sbuf.h"

#include "ae.h"

#include <cjson/cJSON.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FFWD_HTTP_IO_BUF            8192
#define FFWD_HTTP_MAX_HEADER        (64u * 1024u)
#define FFWD_HTTP_MAX_BODY          (64u * 1024u * 1024u)
#define FFWD_HTTP_MAX_PATH          255u
#define FFWD_HTTP_BACKLOG           128
#define FFWD_HTTP_SETSIZE           10240
#define FFWD_HTTP_CLIENT_TIMEOUT_MS 30000
#define FFWD_HTTP_SWEEP_MS          1000

#define FFWD_API_MAX_STANDARD_INPUTS  512
#define FFWD_API_MAX_CONTEXT_DOCS     512
#define FFWD_API_MAX_CONTEXT_CHUNKS   16000
#define FFWD_API_MAX_RERANK_DOCUMENTS 1000
#define FFWD_API_MAX_ITEM_TOKENS      32768
#define FFWD_API_MAX_TOTAL_TOKENS     120000
/* FFWD_LATE_* and FFWD_CONTEXT_SEPARATOR_TOKEN_ID now live in ffwd.h, owned
 * by the tokenizer (ffwd_tok). */
/* Chosen from the L4 scheduler sweep: -b 32 gained ~24% concurrent
 * short-request throughput over 8 with no long-document penalty, while 128
 * was no faster and inflated long-document tail latency. */
#define FFWD_SERVER_DEFAULT_BATCH_SIZE       32
#define FFWD_SERVER_DEFAULT_MAX_BATCH_TOKENS 16384
/* Microseconds the worker waits for more requests before dispatching a batch.
 * CUDA uses a 1 ms window so arrivals group into one launch and keep the GPU
 * busy; MLX and CPU dispatch immediately, where waiting only adds latency.
 * Override with --batch-wait-us. */
#define FFWD_SERVER_BATCH_WAIT_US       0
#define FFWD_SERVER_CUDA_BATCH_WAIT_US  1000
#define FFWD_SERVER_MICROBATCH_MAX_JOBS 128
#define FFWD_MLX_MEMORY_BUDGET_PERCENT  90
#define FFWD_MLX_RESIDENT_MULTIPLIER    2

typedef enum { MODEL_KIND_STANDARD, MODEL_KIND_CONTEXTUAL, MODEL_KIND_LATE } model_kind;

typedef enum {
    FFWD_API_OPENAI = 0,
    FFWD_API_PERPLEXITY = 1,
} embedding_api_t;

typedef struct {
    char *id;
    model_kind kind;
    int dim;
    int min_dim;
    int token_dim;
    ffwd_attention_mode_t attention_mode;
    ffwd_pooling_mode_t pooling_mode;
    int normalize_embeddings;
    embedding_api_t api;
    char *query_instruct;
} model_info;

typedef struct {
    model_info *info;
    char *path;
    ffwd_tok_t *tok;           /* tokenizer: text -> ids + the model's special tokens */
    ffwd_t *backend;           /* compute handle */
    int renormalize_truncated; /* serving policy: re-normalize a Matryoshka cut */
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
    loaded_model *models;
    int n_models;
    ffwd_options_t backend_opts;

    pthread_t worker;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int stopping;
    int worker_ready;
    int worker_init_rc;
    job *job_head;
    job *job_tail;
    /* Tokenizer stage: dispatch enqueues every raw job here, and a single
     * dedicated thread parses + tokenizes it off the worker so tokenization
     * overlaps the GPU. One thread, so the shared model tokenizer is never
     * touched concurrently. */
    pthread_t tokenizer;
    pthread_cond_t raw_cv;
    int raw_stopping;
    job *raw_head;
    job *raw_tail;
    pthread_t renderer;
    pthread_cond_t render_cv;
    int render_stopping;
    job *render_head;
    job *render_tail;
    job *done_head;
    job *done_tail;

    client *clients;
    int n_clients;
} http_server;

typedef struct {
    char method[8];
    char path[FFWD_HTTP_MAX_PATH + 1];
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
    char path[FFWD_HTTP_MAX_PATH + 1];
    char *body;
    size_t body_len;
    char *auth;
    int status;
    char *content_type;
    char *extra_headers;
    char *response;
    size_t response_len;
    /* Filled by tokenize_job (on the tokenizer thread, or on the worker for a
     * request dispatched inline): prep points at a heap request of type
     * prep_kind (1 embedding, 2 contextual, 3 rerank), which the worker moves
     * into its request array. tokenized guards against tokenizing twice. */
    int tokenized;
    int prep_kind;
    void *prep;
    uint64_t created_ns;
    uint64_t started_ns;
    uint64_t parse_ns;
    uint64_t tokenize_ns;
    uint64_t infer_ns;
    uint64_t encode_ns;
    int render_kind;
    struct {
        loaded_model *model;
        char *encoding;
        float *embs;
        int dims;
        int n_inputs;
        int total_tokens;
    } embedding_render;
    job *next;
};

/* Parsed request state, shared by the prepare (handlers), execute (schedule),
 * and render (encode) stages. Each carries its source cJSON root (freed when
 * the request is freed), the resolved model, and the tokenized inputs. */
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
    ffwd_input_t *inputs;
    int total_tokens;
    int ready;
} embedding_request;

typedef struct {
    int *ids;
    int n_tokens;
    ffwd_span_t *spans;
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

/* The server's late_tokens is exactly what ffwd_tokenize_late fills. */
typedef ffwd_late_tokens_t late_tokens;

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

/* ---- server_json.c: request validation and error bodies ---- */
void append_json_string(sbuf *b, const char *s);
char *json_error_body(const char *message, const char *type, size_t *len);
void job_set_error(job *j, int status, const char *message, const char *type);
bool ve_add(cJSON *detail, const char *loc_json, const char *msg, const char *type);
void job_set_422(job *j, cJSON *detail);
cJSON *parse_json_body(job *j, cJSON *detail);
bool cjson_is_integer(cJSON *item);
const char *encoding_from_root(cJSON *root, cJSON *detail, embedding_api_t api);
int text_type_is_query(cJSON *root, cJSON *detail, const char *query_instruct);
int dimensions_from_root(cJSON *root, cJSON *detail, int min_dim, int max_dim, const char *encoding);

/* ---- server_http.c: connection lifecycle, header parse, response framing ---- */
int set_nonblock(int fd);
void client_incref(client *c);
void client_decref_n(client *c, int n);
int close_client_unlink(client *c);
void close_client(client *c);
void append_http_response_ex(client *c,
                             int status,
                             const char *ctype,
                             const char *extra_headers,
                             const char *body,
                             size_t body_len);
void append_http_response(
    client *c, int status, const char *ctype, const char *body, size_t body_len);
int queue_write(client *c);
bool route_is_inference(const char *method, const char *path);
void read_cb(aeEventLoop *loop, int fd, void *clientData, int mask);
/* Defined in server.c (request routing): framed requests are handed here. */
void dispatch_request(client *c);

/* ---- server_encode.c: embedding output encoding ---- */
signed char quantize_int8_tanh(float x);
char *encode_embedding(const float *emb, int dims, const char *encoding);
void append_embedding_value(
    sbuf *b, int index, const float *emb, int dims, const char *encoding, embedding_api_t api);
void set_response_from_buf(job *j, sbuf *b);
int render_embedding_response(embedding_request *r, const float *embs);
void render_contextual_response(contextual_request *r, const float *embs);
void job_render_free(job *j);
void job_set_embedding_render(job *j, const embedding_request *r, const float *embs);
int render_job_response(job *j);

/* ---- server_models.c: model load/lifecycle, tokenize/inference dispatch ---- */
loaded_model *loaded_model_for_label(http_server *s, const char *label);
void free_token_bufs(token_buf *t, int n);
void embedding_request_free(embedding_request *r);
void contextual_request_free(contextual_request *r);
void late_tokens_free(late_tokens *t);
void rerank_request_free(rerank_request *r);
/* Thin timing wrappers over libffwd's ffwd_tokenize / ffwd_tokenize_late. */
int tokenize_one(loaded_model *m, job *j, const char *text, token_buf *out);
int tokenize_input(
    loaded_model *m, job *j, const char *text, const char *query_instruct, token_buf *out);
int tokenize_late_text(loaded_model *m, job *j, const char *text, int is_query, late_tokens *out);
int model_ffwd_batch(loaded_model *m, const ffwd_input_t *inputs, int batch, float *out);
int model_ffwd_spans_batch(loaded_model *m,
                           const ffwd_context_input_t *inputs,
                           int batch,
                           float *out);
int inference_batch_accepts_input(
    const loaded_model *m, int batch, int packed_tokens, int next_tokens, int max_batch_tokens);
int configure_loaded_model(loaded_model *m, const ffwd_config_t *config, int token_dim);
int finalize_loaded_model(loaded_model *m);
int load_one_model(http_server *s, const ffwd_server_model_spec_t *spec);
void free_models(http_server *s);

/* ---- server_handlers.c: per-endpoint prepare / batch-group / execute ---- */
void prepare_embedding_request(job *j, cJSON *root, http_server *s, embedding_request *out);
int embedding_request_compatible(const embedding_request *a, const embedding_request *b);
void execute_embedding_request_list(embedding_request **reqs, int n_reqs);
void prepare_contextual_request(job *j, cJSON *root, http_server *s, contextual_request *out);
int contextual_request_compatible(const contextual_request *a, const contextual_request *b);
void execute_contextual_request_list(contextual_request **reqs, int n_reqs);
void prepare_rerank_request(job *j, cJSON *root, http_server *s, rerank_request *out);
void execute_rerank_request(rerank_request *r);

/* ---- server_schedule.c: job queue, micro-batching, completion ---- */
void enqueue_job(job *j);
int worker_has_pending_jobs(http_server *s);
void enqueue_raw_job(job *j);
job *dequeue_raw_job(http_server *s);
void tokenize_job(http_server *s, job *j);
void enqueue_render_job(job *j);
job *dequeue_render_job(http_server *s);
void finish_job(job *j);
int collect_job_batch(http_server *s, job **jobs, int max_jobs);
void process_job_group(http_server *s, job **jobs, int n_jobs);
void completion_cb(aeEventLoop *loop, int fd, void *clientData, int mask);

#endif /* FFWD_SERVER_INTERNAL_H */
