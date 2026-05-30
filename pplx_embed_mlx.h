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

/*
 * Load one contiguous transformer layer range for distributed execution.
 * Uses the same half-open range and endpoint ownership rules as
 * pplx_model_load_slice().
 */
pplx_mlx_ctx_t *pplx_mlx_load_slice(const char *model_dir,
                                    int layer_start, int layer_end);

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

/*
 * MLX equivalent of pplx_model_forward_slice_batch().
 *
 * Hidden states cross the API boundary as packed float32 rows even though MLX
 * executes a padded dense batch internally.
 */
int pplx_mlx_forward_slice_batch(pplx_mlx_ctx_t *ctx,
                                 const pplx_input_t *inputs, int batch,
                                 const float *input_states,
                                 int layer_start, int layer_end,
                                 int apply_final_norm,
                                 float *out_states);

/*
 * Run one contextual sequence and pool selected token spans.
 * Returns 0 on success, -1 on error.
 */
int pplx_mlx_embed_spans(pplx_mlx_ctx_t *ctx, const int *token_ids,
                         int n_tokens, const pplx_span_t *spans,
                         int n_spans, float *out_embeddings);

/*
 * Run a padded dense contextual document batch and pool every selected span.
 * out_embeddings contains spans in document order.
 */
int pplx_mlx_embed_spans_batch(pplx_mlx_ctx_t *ctx,
                               const pplx_context_input_t *inputs, int batch,
                               float *out_embeddings);

/* Get the config (so main.c can read hidden_size etc.) */
const pplx_config_t *pplx_mlx_config(const pplx_mlx_ctx_t *ctx);

#endif /* PPLX_EMBED_MLX_H */
