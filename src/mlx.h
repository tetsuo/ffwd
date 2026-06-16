/*
 * embed_mlx.h - MLX GPU backend for Apple Silicon
 *
 * Uses Apple's mlx-c to run the transformer on Metal.
 */

#ifndef EMBED_MLX_H
#define EMBED_MLX_H

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

#include <stddef.h>
#include <stdint.h>
#include "embed.h"

/* Opaque MLX model context */
typedef struct embed_mlx_ctx embed_mlx_ctx_t;
typedef struct embed_mlx_late_ctx embed_mlx_late_ctx_t;
typedef struct embed_mlx_late_vectors embed_mlx_late_vectors_t;

typedef struct {
    int quantize_bits;       /* 0 disables; currently 8 */
    int quantize_group_size; /* default 64 when quantize_bits is set */
} embed_mlx_options_t;

/*
 * Load model into MLX arrays from safetensors + config.json.
 * Returns NULL on error.
 */
EMBED_API embed_mlx_ctx_t *embed_mlx_load(const char *model_dir);
EMBED_API embed_mlx_ctx_t *embed_mlx_load_with_options(const char *model_dir,
                                                       const embed_mlx_options_t *opts);

EMBED_API void embed_mlx_free(embed_mlx_ctx_t *ctx);

/*
 * Late-interaction MLX path. This is separate from pooled embeddings because
 * late snapshots produce one projected token vector per input token for MaxSim
 * scoring, not one pooled document vector.
 */
EMBED_API embed_mlx_late_ctx_t *embed_mlx_late_load(const char *model_dir);
EMBED_API embed_mlx_late_ctx_t *embed_mlx_late_load_with_options(const char *model_dir,
                                                                 const embed_mlx_options_t *opts);
EMBED_API void embed_mlx_late_free(embed_mlx_late_ctx_t *ctx);

EMBED_API const embed_config_t *embed_mlx_late_config(const embed_mlx_late_ctx_t *ctx);
EMBED_API int embed_mlx_late_token_dim(const embed_mlx_late_ctx_t *ctx);

/*
 * Encode token_ids into out_vectors[n_tokens, token_dim].
 *
 * If normalize is non-zero, every token vector is L2-normalized on device
 * before copying back to CPU memory.
 */
EMBED_API int embed_mlx_late_encode_tokens(embed_mlx_late_ctx_t *ctx,
                                           const int *token_ids,
                                           int n_tokens,
                                           int normalize,
                                           float *out_vectors);

/*
 * Device-resident late-interaction token vectors.
 *
 * The model context owns the MLX stream; the returned vector handle must not
 * outlive ctx. Copying is optional: MaxSim can consume these handles directly
 * and copy back only final candidate scores.
 */
EMBED_API embed_mlx_late_vectors_t *embed_mlx_late_encode_tokens_device(embed_mlx_late_ctx_t *ctx,
                                                                        const int *token_ids,
                                                                        int n_tokens,
                                                                        int normalize);
EMBED_API void embed_mlx_late_vectors_free(embed_mlx_late_vectors_t *vecs);
EMBED_API int embed_mlx_late_vectors_token_count(const embed_mlx_late_vectors_t *vecs);
EMBED_API int embed_mlx_late_vectors_dim(const embed_mlx_late_vectors_t *vecs);
EMBED_API int embed_mlx_late_vectors_copy(const embed_mlx_late_vectors_t *vecs, float *out_vectors);
/*
 * Encode n_docs documents in one padded transformer pass and pack each
 * document's kept token vectors back-to-back into [total_keep, token_dim] in
 * document order. out_offsets[n_docs + 1] receives the prefix sum of kept
 * counts, so the result feeds embed_mlx_late_maxsim_batch_device directly.
 * Returns NULL on error; the handle must not outlive ctx.
 */
EMBED_API embed_mlx_late_vectors_t *embed_mlx_late_encode_docs_device(embed_mlx_late_ctx_t *ctx,
                                                                      const int *const *doc_ids,
                                                                      const int *n_tokens,
                                                                      const int *const *keep,
                                                                      const int *n_keep,
                                                                      int n_docs,
                                                                      int normalize,
                                                                      int *out_offsets);
/* Device-resident select/concat primitives, retained for the late-interaction
 * verification harness (scripts/check_late_interaction.py); the server rerank
 * path uses embed_mlx_late_encode_docs_device. */
EMBED_API embed_mlx_late_vectors_t *embed_mlx_late_vectors_concat(
    embed_mlx_late_ctx_t *ctx, const embed_mlx_late_vectors_t *const *items, int count);
EMBED_API embed_mlx_late_vectors_t *
embed_mlx_late_vectors_select(embed_mlx_late_ctx_t *ctx,
                              const embed_mlx_late_vectors_t *vecs,
                              const int *token_indices,
                              int count);
EMBED_API int embed_mlx_late_maxsim_batch_device(embed_mlx_late_ctx_t *ctx,
                                                 const embed_mlx_late_vectors_t *query,
                                                 const embed_mlx_late_vectors_t *docs,
                                                 const int *doc_offsets,
                                                 int docs_count,
                                                 float *scores);

/*
 * Compute embedding for token_ids[0..n_tokens-1].
 * Returns malloc'd float[hidden_size] (caller frees). NULL on error.
 */
EMBED_API float *embed_mlx_encode(embed_mlx_ctx_t *ctx, const int *token_ids, int n_tokens);

/*
 * Compute one embedding into caller-provided out[hidden_size].
 * Returns 0 on success, -1 on error.
 */
EMBED_API int embed_mlx_encode_into(embed_mlx_ctx_t *ctx,
                                    const int *token_ids,
                                    int n_tokens,
                                    float *out_embedding);

/*
 * Compute a true padded dense batch on MLX.
 *
 * out_embeddings is caller-provided [batch, hidden_size].
 * Pooling and attention masking follow the loaded model configuration.
 * Returns 0 on success, -1 on error.
 */
EMBED_API int embed_mlx_encode_batch(embed_mlx_ctx_t *ctx,
                                     const embed_input_t *inputs,
                                     int batch,
                                     float *out_embeddings);

/*
 * Run one contextual sequence and pool selected token spans.
 * Returns 0 on success, -1 on error.
 */
EMBED_API int embed_mlx_encode_spans(embed_mlx_ctx_t *ctx,
                                     const int *token_ids,
                                     int n_tokens,
                                     const embed_span_t *spans,
                                     int n_spans,
                                     float *out_embeddings);

/*
 * Run a padded dense contextual document batch and pool every selected span.
 * out_embeddings contains spans in document order.
 */
EMBED_API int embed_mlx_encode_spans_batch(embed_mlx_ctx_t *ctx,
                                           const embed_context_input_t *inputs,
                                           int batch,
                                           float *out_embeddings);

/* Get the config (so embed_cli.c can read hidden_size etc.) */
EMBED_API const embed_config_t *embed_mlx_config(const embed_mlx_ctx_t *ctx);

#endif /* EMBED_MLX_H */
