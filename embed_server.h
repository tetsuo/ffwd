#ifndef PPLX_SERVER_H
#define PPLX_SERVER_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying PPLX_API are exported
 * from the shared library; everything else stays internal. */
#ifndef PPLX_API
#if defined(__GNUC__)
#define PPLX_API __attribute__((visibility("default")))
#else
#define PPLX_API
#endif
#endif

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
    double memory_utilization;  /* fraction of physical memory the MLX
                                   preflight may plan for; 0 = default 0.90,
                                   values above 1.0 allow overcommit */
    int enable_cors;
    const char *api_key;
} pplx_server_config_t;

PPLX_API int pplx_run_server(const pplx_server_config_t *cfg);

#endif
