#ifndef EMBED_SERVER_INTERNAL_H
#define EMBED_SERVER_INTERNAL_H

/* Declarations shared across the embed-server translation units (HTTP framing,
 * JSON validation, output encoding, scheduling, model lifecycle, endpoint
 * handlers). The server is split by concern but compiles as several objects
 * linked into one binary; this header is what they share. Not a public API -
 * the CLI and library do not include it. */

#include "embed.h"
#include "sbuf.h"
#include "tokenizer_bpe.h"
#include "tokenizer_wordpiece.h"

#ifdef USE_MLX
#include "mlx.h"
#endif
#ifdef USE_CUDA
#include "cuda.h"
#endif

#include "deps/ae/ae.h"

#include <cjson/cJSON.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

typedef struct {
    const model_info *info;
    char *path;
    embed_tokenizer_t *tok;
    embed_tokenizer_workspace_t *tok_ws;
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
int dimensions_from_root(
    cJSON *root, cJSON *detail, int min_dim, int max_dim, const char *encoding);

#endif /* EMBED_SERVER_INTERNAL_H */
