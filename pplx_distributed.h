/*
 * pplx_distributed.h - Stateless two-node layer-sharded inference.
 */

#ifndef PPLX_DISTRIBUTED_H
#define PPLX_DISTRIBUTED_H

#include "pplx_embed.h"

typedef struct pplx_dist_client pplx_dist_client_t;

/*
 * Open a coordinator-side connection. The coordinator owns layers
 * [0, local_layer_end); the remote worker must own [local_layer_end, N).
 */
pplx_dist_client_t *pplx_dist_client_open(const char *model_dir,
                                          const char *host, int port,
                                          int local_layer_end, int use_mlx);
void pplx_dist_client_free(pplx_dist_client_t *client);

const pplx_config_t *pplx_dist_client_config(const pplx_dist_client_t *client);

int pplx_dist_client_embed_batch(pplx_dist_client_t *client,
                                 const pplx_input_t *inputs, int batch,
                                 float *out_embeddings);

/*
 * Run a final-stage worker until the process is terminated. The worker owns
 * [layer_start, layer_end), and layer_end must be the model's final layer.
 */
int pplx_dist_run_worker(const char *model_dir, const char *host, int port,
                         int layer_start, int layer_end, int use_mlx);

#endif /* PPLX_DISTRIBUTED_H */
