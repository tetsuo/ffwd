#ifndef PPLX_SERVER_H
#define PPLX_SERVER_H

#include <stdbool.h>

typedef struct {
    const char *id;
    const char *path;
} pplx_server_model_spec_t;

typedef struct {
    const pplx_server_model_spec_t *models;
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
    int allow_memory_overcommit;
    int enable_cors;
    const char *api_key;
} pplx_server_config_t;

int pplx_run_server(const pplx_server_config_t *cfg);

#endif
