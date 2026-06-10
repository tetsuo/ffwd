/*
 * embed_mlx.h - MLX GPU backend for pplx-embed inference
 *
 * Uses Apple's mlx-c to run the transformer on Metal.
 */

#ifndef PPLX_EMBED_MLX_H
#define PPLX_EMBED_MLX_H

#include <stddef.h>
#include <stdint.h>
#include "embed.h"

/* Opaque MLX model context */
typedef struct pplx_mlx_ctx pplx_mlx_ctx_t;
typedef struct pplx_mlx_late_ctx pplx_mlx_late_ctx_t;
typedef struct pplx_mlx_late_vectors pplx_mlx_late_vectors_t;

typedef struct {
    int quantize_bits;       /* 0 disables; currently 8 */
    int quantize_group_size; /* default 64 when quantize_bits is set */
} pplx_mlx_options_t;

/*
 * Load model into MLX arrays from safetensors + config.json.
 * Returns NULL on error.
 */
pplx_mlx_ctx_t *pplx_mlx_load(const char *model_dir);
pplx_mlx_ctx_t *pplx_mlx_load_with_options(const char *model_dir,
                                           const pplx_mlx_options_t *opts);

/*
 * Load one contiguous transformer layer range for distributed execution.
 * Uses the same half-open range and endpoint ownership rules as
 * pplx_model_load_slice().
 */
pplx_mlx_ctx_t *pplx_mlx_load_slice(const char *model_dir,
                                    int layer_start, int layer_end);
pplx_mlx_ctx_t *pplx_mlx_load_slice_with_options(
    const char *model_dir, int layer_start, int layer_end,
    const pplx_mlx_options_t *opts);

void pplx_mlx_free(pplx_mlx_ctx_t *ctx);

/*
 * Late-interaction MLX path. This is separate from pooled embeddings because
 * late snapshots produce one projected token vector per input token for MaxSim
 * scoring, not one pooled document vector.
 */
pplx_mlx_late_ctx_t *pplx_mlx_late_load(const char *model_dir);
pplx_mlx_late_ctx_t *pplx_mlx_late_load_with_options(
    const char *model_dir, const pplx_mlx_options_t *opts);
void pplx_mlx_late_free(pplx_mlx_late_ctx_t *ctx);

const pplx_config_t *pplx_mlx_late_config(const pplx_mlx_late_ctx_t *ctx);
int pplx_mlx_late_token_dim(const pplx_mlx_late_ctx_t *ctx);

/*
 * Encode token_ids into out_vectors[n_tokens, token_dim].
 *
 * If normalize is non-zero, every token vector is L2-normalized on device
 * before copying back to CPU memory.
 */
int pplx_mlx_late_encode_tokens(pplx_mlx_late_ctx_t *ctx,
                                const int *token_ids, int n_tokens,
                                int normalize, float *out_vectors);

/*
 * Device-resident late-interaction token vectors.
 *
 * The model context owns the MLX stream; the returned vector handle must not
 * outlive ctx. Copying is optional: MaxSim can consume these handles directly
 * and copy back only final candidate scores.
 */
pplx_mlx_late_vectors_t *pplx_mlx_late_encode_tokens_device(
    pplx_mlx_late_ctx_t *ctx, const int *token_ids, int n_tokens,
    int normalize);
void pplx_mlx_late_vectors_free(pplx_mlx_late_vectors_t *vecs);
int pplx_mlx_late_vectors_token_count(const pplx_mlx_late_vectors_t *vecs);
int pplx_mlx_late_vectors_dim(const pplx_mlx_late_vectors_t *vecs);
int pplx_mlx_late_vectors_copy(const pplx_mlx_late_vectors_t *vecs,
                               float *out_vectors);
pplx_mlx_late_vectors_t *pplx_mlx_late_vectors_concat(
    pplx_mlx_late_ctx_t *ctx,
    const pplx_mlx_late_vectors_t *const *items, int count);
pplx_mlx_late_vectors_t *pplx_mlx_late_vectors_select(
    pplx_mlx_late_ctx_t *ctx, const pplx_mlx_late_vectors_t *vecs,
    const int *token_indices, int count);
int pplx_mlx_late_maxsim_batch_device(
    pplx_mlx_late_ctx_t *ctx, const pplx_mlx_late_vectors_t *query,
    const pplx_mlx_late_vectors_t *docs, const int *doc_offsets,
    int docs_count, float *scores);

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

/* Get the config (so embed_cli.c can read hidden_size etc.) */
const pplx_config_t *pplx_mlx_config(const pplx_mlx_ctx_t *ctx);

#endif /* PPLX_EMBED_MLX_H */
