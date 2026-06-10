/*
 * embed_distributed.h - Stateless two-node layer-sharded inference.
 */

#ifndef PPLX_DISTRIBUTED_H
#define PPLX_DISTRIBUTED_H

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

#include "embed.h"

typedef struct pplx_dist_client pplx_dist_client_t;

typedef struct {
    int mlx_quantize_bits;
    int mlx_quantize_group_size;
    int activation_bits; /* 0 defaults to fp32; supported: 32, 16 */
} pplx_dist_options_t;

/*
 * Open a coordinator-side connection. The coordinator owns layers
 * [0, local_layer_end); the remote worker must own [local_layer_end, N).
 */
PPLX_API pplx_dist_client_t *pplx_dist_client_open(const char *model_dir,
                                          const char *host, int port,
                                          int local_layer_end, int use_mlx);
PPLX_API pplx_dist_client_t *pplx_dist_client_open_with_options(
    const char *model_dir, const char *host, int port,
    int local_layer_end, int use_mlx, const pplx_dist_options_t *opts);
PPLX_API void pplx_dist_client_free(pplx_dist_client_t *client);

PPLX_API const pplx_config_t *pplx_dist_client_config(const pplx_dist_client_t *client);

PPLX_API int pplx_dist_client_embed_batch(pplx_dist_client_t *client,
                                 const pplx_input_t *inputs, int batch,
                                 float *out_embeddings);

/*
 * Run a final-stage worker until the process is terminated. The worker owns
 * [layer_start, layer_end), and layer_end must be the model's final layer.
 */
PPLX_API int pplx_dist_run_worker(const char *model_dir, const char *host, int port,
                         int layer_start, int layer_end, int use_mlx);
PPLX_API int pplx_dist_run_worker_with_options(
    const char *model_dir, const char *host, int port,
    int layer_start, int layer_end, int use_mlx,
    const pplx_dist_options_t *opts);

#endif /* PPLX_DISTRIBUTED_H */
