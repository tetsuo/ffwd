/*
 * embed.h - embed API
 */

#ifndef EMBED_H
#define EMBED_H

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

/* ========================================================================
 * Shared constants and family defaults
 * ======================================================================== */

#define EMBED_VOCAB_SIZE     151936
#define EMBED_HEAD_DIM       128
#define EMBED_MAX_LAYERS     64      /* upper bound for stack arrays */
#define EMBED_CONTEXT_SEPARATOR_TOKEN_ID 151643 /* <|endoftext|> */

/* ========================================================================
 * Model Configuration (populated from config.json at load time)
 * ======================================================================== */

typedef enum {
    EMBED_ATTENTION_BIDIRECTIONAL = 0,
    EMBED_ATTENTION_CAUSAL = 1,
} embed_attention_mode_t;

typedef enum {
    EMBED_POOL_MEAN = 0,
    EMBED_POOL_LAST_TOKEN = 1,
} embed_pooling_mode_t;

typedef struct {
    int hidden_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int q_dim;              /* n_heads    * head_dim */
    int kv_dim;             /* n_kv_heads * head_dim */
    int intermediate_size;
    int vocab_size;
    float rms_norm_eps;
    float rope_theta;
    embed_attention_mode_t attention_mode;
    embed_pooling_mode_t pooling_mode;
    int normalize_embeddings;
    /* Qwen3-Embedding pools the last token, which is the tokenizer's
     * <|endoftext|> suffix added by its post-processor - not the model's chat
     * eos_token_id (<|im_end|>). Text frontends append it and resolve the id
     * from the tokenizer exactly like the contextual separator
     * (EMBED_CONTEXT_SEPARATOR_TOKEN_ID is the released-model fallback). The
     * model forward consumes token ids exactly and never appends. */
    int append_terminal_token;
} embed_config_t;

/* ========================================================================
 * Model + workspace ownership
 * ======================================================================== */

typedef struct embed_model embed_model_t;
typedef struct embed_workspace embed_workspace_t;
typedef struct embed_late_model embed_late_model_t;
typedef struct embed_late_workspace embed_late_workspace_t;

typedef struct {
    const int *ids;
    int n_tokens;
} embed_input_t;

typedef struct {
    int start;
    int n_tokens;
} embed_span_t;

typedef struct {
    embed_input_t input;
    const embed_span_t *spans;
    int n_spans;
} embed_context_input_t;

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
EMBED_API embed_model_t *embed_model_load(const char *model_dir);

EMBED_API void embed_model_free(embed_model_t *model);

/*
 * Allocate/free mutable scratch buffers for a model. The model must outlive
 * workspaces created from it.
 */
EMBED_API embed_workspace_t *embed_workspace_new(const embed_model_t *model);
EMBED_API void embed_workspace_free(embed_workspace_t *ws);

/* Config accessor for opaque model ownership. */
EMBED_API const embed_config_t *embed_model_config(const embed_model_t *model);

/*
 * Current mutable workspace memory owned by this workspace, in bytes.
 *
 * This includes the workspace struct, scratch buffers, and RoPE cache capacity.
 * It does not include immutable mmap'd model weights.
 */
EMBED_API size_t embed_workspace_nbytes(const embed_workspace_t *ws);

/*
 * Compute embedding for a token sequence.
 *
 *   1. Token embedding lookup
 *   2. N transformer layers (GQA + RoPE + SwiGLU)
 *   3. Final RMSNorm
 *   4. Model-defined pooling and optional L2 normalization
 *
 * pplx-embed-v1 uses bidirectional attention and unnormalized mean pooling.
 * Qwen3-Embedding uses causal attention and normalized last-token pooling.
 * Token IDs are consumed exactly as provided; text frontends append any
 * model-required terminal token before calling this API.
 *
 * Returns malloc'd float[hidden_size] (caller frees). NULL on error.
 */
EMBED_API float *embed_model_encode(const embed_model_t *model, embed_workspace_t *ws,
                        const int *token_ids, int n_tokens);

/*
 * Compute one embedding into caller-provided out[hidden_size].
 * Returns 0 on success, -1 on error.
 */
EMBED_API int embed_model_encode_into(const embed_model_t *model, embed_workspace_t *ws,
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
EMBED_API int embed_model_encode_batch(const embed_model_t *model, embed_workspace_t *ws,
                           const embed_input_t *inputs, int batch,
                           float *out_embeddings);

/*
 * Apply the config's pooling mode to packed final hidden states.
 *
 * states is [sum(n_tokens), hidden_size]. seq_lengths contains one positive
 * token count per sequence. The states must already include final RMSNorm.
 */
EMBED_API int embed_pool_batch(const embed_config_t *cfg, const float *states,
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
EMBED_API float *embed_model_forward(const embed_model_t *model, embed_workspace_t *ws,
                          const int *token_ids, int n_tokens);

/*
 * Run the transformer forward pass into caller-provided
 * out_states[n_tokens * hidden_size]. Returns 0 on success, -1 on error.
 */
EMBED_API int embed_model_forward_into(const embed_model_t *model, embed_workspace_t *ws,
                            const int *token_ids, int n_tokens,
                            float *out_states);

/*
 * Pool embeddings from final hidden states.
 *
 * states is [n_tokens, hidden_size], normally produced by
 * embed_model_forward_into(). Each span is mean pooled into
 * out_embeddings[n_spans, hidden_size] without L2 normalization.
 */
EMBED_API int embed_pool_spans(const embed_config_t *cfg, const float *states, int n_tokens,
                    const embed_span_t *spans, int n_spans,
                    float *out_embeddings);

/*
 * Run one contextual sequence and pool selected token spans.
 * Returns 0 on success, -1 on error.
 */
EMBED_API int embed_model_encode_spans(const embed_model_t *model, embed_workspace_t *ws,
                           const int *token_ids, int n_tokens,
                           const embed_span_t *spans, int n_spans,
                           float *out_embeddings);

/*
 * Run a packed/ragged contextual document batch and pool every selected span.
 *
 * Documents attend independently, RoPE positions restart for each document,
 * and out_embeddings contains spans in document order.
 */
EMBED_API int embed_model_encode_spans_batch(const embed_model_t *model,
                                 embed_workspace_t *ws,
                                 const embed_context_input_t *inputs, int batch,
                                 float *out_embeddings);

/*
 * L2-normalize vec[dim] in place.
 *
 * Returns 0 on success. Returns -1 for invalid arguments or a zero-length
 * vector, which cannot be normalized.
 */
EMBED_API int embed_l2_normalize(float *vec, int dim);

/*
 * Cosine similarity between two vectors.
 * Returns 0 for invalid arguments or if either vector has zero length.
 */
EMBED_API float embed_cosine_similarity(const float *a, const float *b, int dim);

/*
 * Late-interaction model API.
 *
 * A late model produces one 128D vector per retained token. The vectors are
 * projected by the snapshot's 1_Dense head and L2-normalized per token by
 * default, matching ColBERT/MaxSim scoring semantics.
 */
EMBED_API embed_late_model_t *embed_late_model_load(const char *model_dir);
EMBED_API void embed_late_model_free(embed_late_model_t *model);

EMBED_API embed_late_workspace_t *embed_late_workspace_new(const embed_late_model_t *model);
EMBED_API void embed_late_workspace_free(embed_late_workspace_t *ws);

EMBED_API const embed_config_t *embed_late_model_config(const embed_late_model_t *model);
EMBED_API int embed_late_model_token_dim(const embed_late_model_t *model);

/*
 * Encode token_ids into out_vectors[n_tokens, token_dim].
 *
 * If normalize is non-zero, every non-zero token vector is L2-normalized.
 */
EMBED_API int embed_late_model_encode_tokens(const embed_late_model_t *model,
                                  embed_late_workspace_t *ws,
                                  const int *token_ids, int n_tokens,
                                  int normalize, float *out_vectors);

/*
 * Sum over query tokens of max dot product against document tokens.
 */
EMBED_API float embed_late_maxsim(const float *query_vectors, int query_tokens,
                       const float *doc_vectors, int doc_tokens,
                       int dim);

/*
 * Score one query against a packed/ragged batch of documents.
 *
 * doc_offsets has docs + 1 entries. Document i is stored in
 * doc_vectors[doc_offsets[i] * dim .. doc_offsets[i + 1] * dim).
 * scores is caller-provided [docs].
 */
EMBED_API int embed_late_maxsim_batch(const float *query_vectors, int query_tokens,
                           const float *doc_vectors,
                           const int *doc_offsets, int docs,
                           int dim, float *scores);

/* Verbose level: 0=quiet, 1=info, 2=debug */
EMBED_API extern int embed_verbose;
EMBED_API extern int qwen_verbose;

#endif /* EMBED_H */
