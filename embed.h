/*
 * embed.h - pplx-embed API
 */

#ifndef PPLX_EMBED_H
#define PPLX_EMBED_H

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

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Shared constants (identical across all models)
 * ======================================================================== */

#define PPLX_VOCAB_SIZE     151936
#define PPLX_HEAD_DIM       128
#define PPLX_MAX_LAYERS     64      /* upper bound for stack arrays */
#define PPLX_CONTEXT_SEPARATOR_TOKEN_ID 151643 /* <|endoftext|> */

/* ========================================================================
 * Model Configuration (populated from config.json at load time)
 * ======================================================================== */

typedef struct {
    int hidden_size;        /* 0.6B: 1024   4B: 2560 */
    int n_layers;           /* 0.6B: 28     4B: 36   */
    int n_heads;            /* 0.6B: 16     4B: 32   */
    int n_kv_heads;         /* 0.6B: 8      4B: 8    */
    int head_dim;           /* always 128 */
    int q_dim;              /* n_heads    * head_dim */
    int kv_dim;             /* n_kv_heads * head_dim */
    int intermediate_size;  /* 0.6B: 3072   4B: 9728 */
    int vocab_size;         /* 151936 */
    float rms_norm_eps;     /* 1e-6 */
    float rope_theta;       /* 1e6  */
} pplx_config_t;

/* ========================================================================
 * Model + workspace ownership
 * ======================================================================== */

typedef struct pplx_model pplx_model_t;
typedef struct pplx_workspace pplx_workspace_t;
typedef struct pplx_late_model pplx_late_model_t;
typedef struct pplx_late_workspace pplx_late_workspace_t;

typedef struct {
    const int *ids;
    int n_tokens;
} pplx_input_t;

typedef struct {
    int start;
    int n_tokens;
} pplx_span_t;

typedef struct {
    pplx_input_t input;
    const pplx_span_t *spans;
    int n_spans;
} pplx_context_input_t;

/* ========================================================================
 * API
 * ======================================================================== */

/*
 * Load immutable model data from a directory containing:
 *   config.json                          - model hyperparameters
 *   model.safetensors (or sharded)       - F32 or BF16 weights
 *   vocab.json + merges.txt              - BPE tokenizer
 *
 * Reads config.json to determine model size (0.6B vs 4B).
 * Returns NULL on error.
 */
PPLX_API pplx_model_t *pplx_model_load(const char *model_dir);

PPLX_API void pplx_model_free(pplx_model_t *model);

/*
 * Allocate/free mutable scratch buffers for a model. The model must outlive
 * workspaces created from it.
 */
PPLX_API pplx_workspace_t *pplx_workspace_new(const pplx_model_t *model);
PPLX_API void pplx_workspace_free(pplx_workspace_t *ws);

/* Config accessor for opaque model ownership. */
PPLX_API const pplx_config_t *pplx_model_config(const pplx_model_t *model);

/*
 * Current mutable workspace memory owned by this workspace, in bytes.
 *
 * This includes the workspace struct, scratch buffers, and RoPE cache capacity.
 * It does not include immutable mmap'd model weights.
 */
PPLX_API size_t pplx_workspace_nbytes(const pplx_workspace_t *ws);

/*
 * Compute embedding for a token sequence.
 *
 *   1. Token embedding lookup
 *   2. N transformer layers (bidirectional GQA + RoPE + SwiGLU)
 *   3. Final RMSNorm
 *   4. Mean pooling over all positions
 *
 * pplx-embed-v1 embeddings are intentionally not L2-normalized. Use cosine
 * similarity for comparisons, or normalize explicitly before storing in vector
 * databases that only support inner product.
 *
 * Returns malloc'd float[hidden_size] (caller frees). NULL on error.
 */
PPLX_API float *pplx_model_embed(const pplx_model_t *model, pplx_workspace_t *ws,
                        const int *token_ids, int n_tokens);

/*
 * Compute one embedding into caller-provided out[hidden_size].
 * Returns 0 on success, -1 on error.
 */
PPLX_API int pplx_model_embed_into(const pplx_model_t *model, pplx_workspace_t *ws,
                          const int *token_ids, int n_tokens,
                          float *out_embedding);

/*
 * Compute embeddings for a true packed/ragged batch.
 *
 * out_embeddings is caller-provided [batch, hidden_size].
 * Each sequence attends only to tokens from the same input, and RoPE
 * positions restart at zero for every input.
 *
 * Returns 0 on success, -1 on error.
 */
PPLX_API int pplx_model_embed_batch(const pplx_model_t *model, pplx_workspace_t *ws,
                           const pplx_input_t *inputs, int batch,
                           float *out_embeddings);

/*
 * Mean-pool packed final hidden states for a ragged batch.
 *
 * states is [sum(n_tokens), hidden_size]. seq_lengths contains one positive
 * token count per sequence. The states must already include final RMSNorm.
 */
PPLX_API int pplx_pool_batch(const pplx_config_t *cfg, const float *states,
                    const int *seq_lengths, int batch,
                    float *out_embeddings);

/*
 * Run the transformer forward pass WITHOUT pooling.
 * Returns the full [n_tokens * hidden_size] output after final RMSNorm.
 * Caller frees the returned buffer.  NULL on error.
 *
 * Useful for contextual (late-chunking) models where you need
 * per-token embeddings before splitting by separator positions.
 */
PPLX_API float *pplx_model_forward(const pplx_model_t *model, pplx_workspace_t *ws,
                          const int *token_ids, int n_tokens);

/*
 * Run the transformer forward pass into caller-provided
 * out_states[n_tokens * hidden_size]. Returns 0 on success, -1 on error.
 */
PPLX_API int pplx_model_forward_into(const pplx_model_t *model, pplx_workspace_t *ws,
                            const int *token_ids, int n_tokens,
                            float *out_states);

/*
 * Pool embeddings from final hidden states.
 *
 * states is [n_tokens, hidden_size], normally produced by
 * pplx_model_forward_into(). Each span is mean pooled into
 * out_embeddings[n_spans, hidden_size] without L2 normalization.
 */
PPLX_API int pplx_pool_spans(const pplx_config_t *cfg, const float *states, int n_tokens,
                    const pplx_span_t *spans, int n_spans,
                    float *out_embeddings);

/*
 * Run one contextual sequence and pool selected token spans.
 * Returns 0 on success, -1 on error.
 */
PPLX_API int pplx_model_embed_spans(const pplx_model_t *model, pplx_workspace_t *ws,
                           const int *token_ids, int n_tokens,
                           const pplx_span_t *spans, int n_spans,
                           float *out_embeddings);

/*
 * Run a packed/ragged contextual document batch and pool every selected span.
 *
 * Documents attend independently, RoPE positions restart for each document,
 * and out_embeddings contains spans in document order.
 */
PPLX_API int pplx_model_embed_spans_batch(const pplx_model_t *model,
                                 pplx_workspace_t *ws,
                                 const pplx_context_input_t *inputs, int batch,
                                 float *out_embeddings);

/*
 * L2-normalize vec[dim] in place.
 *
 * Returns 0 on success. Returns -1 for invalid arguments or a zero-length
 * vector, which cannot be normalized.
 */
PPLX_API int pplx_l2_normalize(float *vec, int dim);

/*
 * Cosine similarity between two vectors.
 * Returns 0 for invalid arguments or if either vector has zero length.
 */
PPLX_API float pplx_cosine_similarity(const float *a, const float *b, int dim);

/*
 * Late-interaction model API.
 *
 * A late model produces one 128D vector per retained token. The vectors are
 * projected by the snapshot's 1_Dense head and L2-normalized per token by
 * default, matching ColBERT/MaxSim scoring semantics.
 */
PPLX_API pplx_late_model_t *pplx_late_model_load(const char *model_dir);
PPLX_API void pplx_late_model_free(pplx_late_model_t *model);

PPLX_API pplx_late_workspace_t *pplx_late_workspace_new(const pplx_late_model_t *model);
PPLX_API void pplx_late_workspace_free(pplx_late_workspace_t *ws);

PPLX_API const pplx_config_t *pplx_late_model_config(const pplx_late_model_t *model);
PPLX_API int pplx_late_model_token_dim(const pplx_late_model_t *model);

/*
 * Encode token_ids into out_vectors[n_tokens, token_dim].
 *
 * If normalize is non-zero, every non-zero token vector is L2-normalized.
 */
PPLX_API int pplx_late_model_encode_tokens(const pplx_late_model_t *model,
                                  pplx_late_workspace_t *ws,
                                  const int *token_ids, int n_tokens,
                                  int normalize, float *out_vectors);

/*
 * Sum over query tokens of max dot product against document tokens.
 */
PPLX_API float pplx_late_maxsim(const float *query_vectors, int query_tokens,
                       const float *doc_vectors, int doc_tokens,
                       int dim);

/*
 * Score one query against a packed/ragged batch of documents.
 *
 * doc_offsets has docs + 1 entries. Document i is stored in
 * doc_vectors[doc_offsets[i] * dim .. doc_offsets[i + 1] * dim).
 * scores is caller-provided [docs].
 */
PPLX_API int pplx_late_maxsim_batch(const float *query_vectors, int query_tokens,
                           const float *doc_vectors,
                           const int *doc_offsets, int docs,
                           int dim, float *scores);

/* Verbose level: 0=quiet, 1=info, 2=debug */
PPLX_API extern int pplx_verbose;
PPLX_API extern int qwen_verbose;

#endif /* PPLX_EMBED_H */
