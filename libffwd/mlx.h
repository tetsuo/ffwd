/*
 * ffwd_mlx.h - MLX GPU backend for Apple Silicon
 *
 * Uses Apple's mlx-c to run the transformer on Metal.
 */

#ifndef FFWD_MLX_H
#define FFWD_MLX_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying FFWD_API are exported
 * from the shared library; everything else stays internal. */
#ifndef FFWD_API
#    if defined(__GNUC__)
#        define FFWD_API __attribute__((visibility("default")))
#    else
#        define FFWD_API
#    endif
#endif

#include <stddef.h>
#include <stdint.h>
#include "ffwd.h"

/* Opaque MLX model context */
typedef struct ffwd_mlx_ctx ffwd_mlx_ctx_t;
typedef struct ffwd_mlx_late_ctx ffwd_mlx_late_ctx_t;
typedef struct ffwd_mlx_late_vectors ffwd_mlx_late_vectors_t;

typedef struct {
    int quantize_bits;       /* 0 disables; currently 8 */
    int quantize_group_size; /* default 64 when quantize_bits is set */
} ffwd_mlx_options_t;

/*
 * Load model into MLX arrays from safetensors + config.json.
 * Returns NULL on error.
 */
FFWD_API ffwd_mlx_ctx_t *ffwd_mlx_load(const char *model_dir);
FFWD_API ffwd_mlx_ctx_t *ffwd_mlx_load_with_options(const char *model_dir,
                                                    const ffwd_mlx_options_t *opts);

FFWD_API void ffwd_mlx_free(ffwd_mlx_ctx_t *ctx);

/*
 * Late-interaction MLX path. This is separate from pooled embeddings because
 * late snapshots produce one projected token vector per input token for MaxSim
 * scoring, not one pooled document vector.
 */
FFWD_API ffwd_mlx_late_ctx_t *ffwd_mlx_late_load(const char *model_dir);
FFWD_API ffwd_mlx_late_ctx_t *ffwd_mlx_late_load_with_options(const char *model_dir,
                                                              const ffwd_mlx_options_t *opts);
FFWD_API void ffwd_mlx_late_free(ffwd_mlx_late_ctx_t *ctx);

FFWD_API const ffwd_config_t *ffwd_mlx_late_config(const ffwd_mlx_late_ctx_t *ctx);
FFWD_API int ffwd_mlx_late_token_dim(const ffwd_mlx_late_ctx_t *ctx);

/*
 * Encode token_ids into out_vectors[n_tokens, token_dim].
 *
 * If normalize is non-zero, every token vector is L2-normalized on device
 * before copying back to CPU memory.
 */
FFWD_API int ffwd_mlx_late_encode_tokens(
    ffwd_mlx_late_ctx_t *ctx, const int *token_ids, int n_tokens, int normalize, float *out_vectors);

/*
 * Device-resident late-interaction token vectors.
 *
 * The model context owns the MLX stream; the returned vector handle must not
 * outlive ctx. Copying is optional: MaxSim can consume these handles directly
 * and copy back only final candidate scores.
 */
FFWD_API ffwd_mlx_late_vectors_t *ffwd_mlx_late_encode_tokens_device(ffwd_mlx_late_ctx_t *ctx,
                                                                     const int *token_ids,
                                                                     int n_tokens,
                                                                     int normalize);
FFWD_API void ffwd_mlx_late_vectors_free(ffwd_mlx_late_vectors_t *vecs);
FFWD_API int ffwd_mlx_late_vectors_token_count(const ffwd_mlx_late_vectors_t *vecs);
FFWD_API int ffwd_mlx_late_vectors_dim(const ffwd_mlx_late_vectors_t *vecs);
FFWD_API int ffwd_mlx_late_vectors_copy(const ffwd_mlx_late_vectors_t *vecs, float *out_vectors);
/*
 * Encode n_docs documents in one padded transformer pass and pack each
 * document's kept token vectors back-to-back into [total_keep, token_dim] in
 * document order. out_offsets[n_docs + 1] receives the prefix sum of kept
 * counts, so the result feeds ffwd_mlx_late_maxsim_batch_device directly.
 * Returns NULL on error; the handle must not outlive ctx.
 */
FFWD_API ffwd_mlx_late_vectors_t *ffwd_mlx_late_encode_docs_device(ffwd_mlx_late_ctx_t *ctx,
                                                                   const int *const *doc_ids,
                                                                   const int *n_tokens,
                                                                   const int *const *keep,
                                                                   const int *n_keep,
                                                                   int n_docs,
                                                                   int normalize,
                                                                   int *out_offsets);
/* Device-resident select/concat primitives, retained for the late-interaction
 * verification harness (tests/integration/check_late_interaction.py); the server rerank
 * path uses ffwd_mlx_late_encode_docs_device. */
FFWD_API ffwd_mlx_late_vectors_t *ffwd_mlx_late_vectors_concat(
    ffwd_mlx_late_ctx_t *ctx, const ffwd_mlx_late_vectors_t *const *items, int count);
FFWD_API ffwd_mlx_late_vectors_t *ffwd_mlx_late_vectors_select(ffwd_mlx_late_ctx_t *ctx,
                                                               const ffwd_mlx_late_vectors_t *vecs,
                                                               const int *token_indices,
                                                               int count);
FFWD_API int ffwd_mlx_late_maxsim_batch_device(ffwd_mlx_late_ctx_t *ctx,
                                               const ffwd_mlx_late_vectors_t *query,
                                               const ffwd_mlx_late_vectors_t *docs,
                                               const int *doc_offsets,
                                               int docs_count,
                                               float *scores);

/*
 * Compute embedding for token_ids[0..n_tokens-1].
 * Returns malloc'd float[hidden_size] (caller frees). NULL on error.
 */
FFWD_API float *ffwd_mlx_encode(ffwd_mlx_ctx_t *ctx, const int *token_ids, int n_tokens);

/*
 * Compute one embedding into caller-provided out[hidden_size].
 * Returns 0 on success, -1 on error.
 */
FFWD_API int
ffwd_mlx_encode_into(ffwd_mlx_ctx_t *ctx, const int *token_ids, int n_tokens, float *out_embedding);

/*
 * Compute a true padded dense batch on MLX.
 *
 * out_embeddings is caller-provided [batch, hidden_size].
 * Pooling and attention masking follow the loaded model configuration.
 * Returns 0 on success, -1 on error.
 */
FFWD_API int ffwd_mlx_encode_batch(ffwd_mlx_ctx_t *ctx,
                                   const ffwd_input_t *inputs,
                                   int batch,
                                   float *out_embeddings);

/*
 * Run one contextual sequence and pool selected token spans.
 * Returns 0 on success, -1 on error.
 */
FFWD_API int ffwd_mlx_encode_spans(ffwd_mlx_ctx_t *ctx,
                                   const int *token_ids,
                                   int n_tokens,
                                   const ffwd_span_t *spans,
                                   int n_spans,
                                   float *out_embeddings);

/*
 * Run a padded dense contextual document batch and pool every selected span.
 * out_embeddings contains spans in document order.
 */
FFWD_API int ffwd_mlx_encode_spans_batch(ffwd_mlx_ctx_t *ctx,
                                         const ffwd_context_input_t *inputs,
                                         int batch,
                                         float *out_embeddings);

/* Get the config (so ffwd_cli.c can read hidden_size etc.) */
FFWD_API const ffwd_config_t *ffwd_mlx_config(const ffwd_mlx_ctx_t *ctx);

#endif /* FFWD_MLX_H */
