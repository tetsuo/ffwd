/*
 * pplx_embed_mlx.h - MLX GPU backend for pplx-embed inference
 *
 * Uses Apple's mlx-c to run the transformer on Metal.
 */

#ifndef PPLX_EMBED_MLX_H
#define PPLX_EMBED_MLX_H

#include <stddef.h>
#include <stdint.h>
#include "pplx_embed.h"

/* Opaque MLX model context */
typedef struct pplx_mlx_ctx pplx_mlx_ctx_t;

/*
 * Load model into MLX arrays from safetensors + config.json.
 * Returns NULL on error.
 */
pplx_mlx_ctx_t *pplx_mlx_load(const char *model_dir);

void pplx_mlx_free(pplx_mlx_ctx_t *ctx);

/*
 * Compute embedding for token_ids[0..n_tokens-1].
 * Returns malloc'd float[hidden_size] (caller frees). NULL on error.
 */
float *pplx_mlx_embed(pplx_mlx_ctx_t *ctx, const int *token_ids, int n_tokens);

/*
 * Compute one embedding into caller-provided out[hidden_size].
 * Returns 0 on success, -1 on error.
 */
int pplx_mlx_embed_into(pplx_mlx_ctx_t *ctx, const int *token_ids,
                        int n_tokens, float *out_embedding);

/*
 * Compute a true padded dense batch on MLX.
 *
 * out_embeddings is caller-provided [batch, hidden_size].
 * Padded tokens are masked out of attention keys and mean pooling.
 * Returns 0 on success, -1 on error.
 */
int pplx_mlx_embed_batch(pplx_mlx_ctx_t *ctx, const pplx_input_t *inputs,
                         int batch, float *out_embeddings);

/* Get the config (so main.c can read hidden_size etc.) */
const pplx_config_t *pplx_mlx_config(const pplx_mlx_ctx_t *ctx);

#endif /* PPLX_EMBED_MLX_H */
