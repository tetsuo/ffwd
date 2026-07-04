#ifndef FFWD_SERVER_H
#define FFWD_SERVER_H

#ifndef FFWD_API
#    if defined(__GNUC__)
#        define FFWD_API __attribute__((visibility("default")))
#    else
#        define FFWD_API
#    endif
#endif

#include <stdbool.h>

typedef enum {
    FFWD_SERVER_MODEL_STANDARD = 0,
    FFWD_SERVER_MODEL_CONTEXTUAL = 1,
    FFWD_SERVER_MODEL_LATE = 2,
} ffwd_server_model_kind_t;

typedef enum {
    FFWD_SERVER_API_DEFAULT = 0, /* currently OpenAI-compatible float output */
    FFWD_SERVER_API_OPENAI = 1,
    FFWD_SERVER_API_PERPLEXITY = 2,
} ffwd_server_embedding_api_t;

typedef struct {
    const char *id; /* served label; requests match this string exactly */
    const char *path;
    ffwd_server_model_kind_t kind;
    ffwd_server_embedding_api_t api;
    int min_dim;                /* 0 means the loaded model's full hidden size */
    const char *query_instruct; /* NULL means use config_sentence_transformers.json if present */
} ffwd_server_model_spec_t;

typedef struct {
    const ffwd_server_model_spec_t *models;
    int n_models;
    const char *host;
    int port;
    int batch_size;
    int max_batch_tokens;
    int max_request_tokens;      /* per-request total-token cap; 0 = default 120000 */
    int max_concurrent_requests; /* in-flight request cap before 429; 0 = default 512 */
    int auto_truncate;           /* truncate over-long inputs instead of rejecting them */
    int batch_wait_us;
    /* GPU build tuning; ignored by backends that do not use it. */
    int gpu_quantize_bits;
    int gpu_quantize_group_size;
    double memory_utilization; /* fraction of physical memory the GPU preflight
                                  may plan for; 0 = default 0.90, above 1.0
                                  allows overcommit */
    const char *api_key;
} ffwd_server_config_t;

FFWD_API int ffwd_run_server(const ffwd_server_config_t *cfg);

#endif
