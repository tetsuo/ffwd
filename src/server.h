#ifndef EMBED_SERVER_H
#define EMBED_SERVER_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying EMBED_API are exported
 * from the shared library; everything else stays internal. */
#ifndef EMBED_API
#if defined(__GNUC__)
#define EMBED_API __attribute__((visibility("default")))
#else
#define EMBED_API
#endif
#endif

#include <stdbool.h>

typedef enum {
    EMBED_SERVER_MODEL_STANDARD = 0,
    EMBED_SERVER_MODEL_CONTEXTUAL = 1,
    EMBED_SERVER_MODEL_LATE = 2,
} embed_server_model_kind_t;

typedef enum {
    EMBED_SERVER_API_DEFAULT = 0, /* currently OpenAI-compatible float output */
    EMBED_SERVER_API_OPENAI = 1,
    EMBED_SERVER_API_PERPLEXITY = 2,
} embed_server_embedding_api_t;

typedef struct {
    const char *id; /* served label; requests match this string exactly */
    const char *path;
    embed_server_model_kind_t kind;
    embed_server_embedding_api_t api;
    int min_dim;                /* 0 means the loaded model's full hidden size */
    const char *query_instruct; /* NULL means use config_sentence_transformers.json if present */
} embed_server_model_spec_t;

typedef struct {
    const embed_server_model_spec_t *models;
    int n_models;
    const char *host;
    int port;
    int batch_size;
    int max_batch_tokens;
    int batch_wait_us;
    int use_mlx;
    int use_cuda;
    int mlx_quantize_bits;
    int mlx_quantize_group_size;
    double memory_utilization; /* fraction of physical memory the MLX
                                   preflight may plan for; 0 = default 0.90,
                                   values above 1.0 allow overcommit */
    const char *api_key;
} embed_server_config_t;

EMBED_API int embed_run_server(const embed_server_config_t *cfg);

#endif
