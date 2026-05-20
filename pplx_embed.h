/*
 * pplx_embed.h - Pure C inference for Perplexity AI's pplx-embed models.
 *
 * Supports all pplx-embed-v1 and pplx-embed-context-v1 variants (0.6B, 4B).
 */

#ifndef PPLX_EMBED_H
#define PPLX_EMBED_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Shared constants (identical across all models)
 * ======================================================================== */

#define PPLX_VOCAB_SIZE     151936
#define PPLX_HEAD_DIM       128
#define PPLX_MAX_LAYERS     64      /* upper bound for stack arrays */

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
 * Per-Layer Weights  (all F32, direct mmap pointers - read-only)
 * ======================================================================== */

typedef struct {
    const float *wq;            /* [q_dim,   hidden] */
    const float *wk;            /* [kv_dim,  hidden] */
    const float *wv;            /* [kv_dim,  hidden] */
    const float *wo;            /* [hidden,  q_dim]  */
    const float *q_norm;        /* [head_dim] */
    const float *k_norm;        /* [head_dim] */
    const float *input_norm;    /* [hidden] */
    const float *post_attn_norm;/* [hidden] */
    const float *gate_proj;     /* [intermediate, hidden] */
    const float *up_proj;       /* [intermediate, hidden] */
    const float *down_proj;     /* [hidden, intermediate] */
} pplx_layer_t;

/* ========================================================================
 * Full Model Weights
 * ======================================================================== */

typedef struct {
    const float *embed_tokens;  /* [vocab_size, hidden] */
    pplx_layer_t *layers;       /* [n_layers] heap-allocated */
    const float *norm;          /* [hidden] */
} pplx_weights_t;

/* ========================================================================
 * Model + workspace ownership
 * ======================================================================== */

typedef struct pplx_model pplx_model_t;

typedef struct {
    const pplx_model_t *model;  /* immutable model this workspace belongs to */
    /* Working buffers, grown lazily to buf_seq_cap */
    int    buf_seq_cap;
    float *x;                   /* [seq, hidden]       */
    float *x_norm;              /* [seq, hidden]       */
    float *q;                   /* [seq, q_dim]        */
    float *k;                   /* [seq, kv_dim]       */
    float *v;                   /* [seq, kv_dim]       */
    float *attn_out;            /* [seq, q_dim]        */
    float *proj_out;            /* [seq, hidden]       */
    float *ffn_gate;            /* [seq, intermediate] */
    float *ffn_up;              /* [seq, intermediate] */
    float *ffn_out;             /* [seq, hidden]       */

    /* RoPE cosine/sine cache [n_pos, head_dim], grown lazily */
    float *rope_cos;
    float *rope_sin;
    int    rope_cache_cap;
} pplx_workspace_t;

/* Convenience context: owns one model and one workspace. */
typedef struct {
    pplx_model_t     *model;
    pplx_workspace_t *workspace;
    pplx_config_t     config;   /* backward-compatible snapshot */
    pplx_weights_t    weights;  /* backward-compatible read-only aliases */
} pplx_ctx_t;

typedef struct {
    const int *ids;
    int n_tokens;
} pplx_input_t;

/* ========================================================================
 * API
 * ======================================================================== */

/*
 * Load model from directory containing:
 *   config.json                          - model hyperparameters
 *   model.safetensors (or sharded)       - F32 weights
 *   vocab.json + merges.txt              - BPE tokenizer
 *
 * Reads config.json to determine model size (0.6B vs 4B).
 * Returns NULL on error.
 */
pplx_ctx_t *pplx_load(const char *model_dir);

/* Free all resources */
void pplx_free(pplx_ctx_t *ctx);

/*
 * Load/free immutable model data only. A model owns config, mmap'd
 * safetensors, and read-only weight pointers. It can be shared by multiple
 * workspaces, but the model must outlive all workspaces created from it.
 */
pplx_model_t *pplx_model_load(const char *model_dir);
void pplx_model_free(pplx_model_t *model);

/*
 * Allocate/free mutable scratch buffers for a model. A workspace is not
 * thread-safe, but multiple workspaces can use the same immutable model in
 * different threads.
 */
pplx_workspace_t *pplx_workspace_new(const pplx_model_t *model);
void pplx_workspace_free(pplx_workspace_t *ws);

/* Config accessors for opaque-ish model/context ownership. */
const pplx_config_t *pplx_model_config(const pplx_model_t *model);
const pplx_config_t *pplx_config(const pplx_ctx_t *ctx);

/*
 * Compute embedding for a token sequence.
 *
 *   1. Token embedding lookup
 *   2. N transformer layers (bidirectional GQA + RoPE + SwiGLU)
 *   3. Final RMSNorm
 *   4. Mean pooling over all positions
 *   5. L2 normalization
 *
 * Returns malloc'd float[hidden_size] (caller frees). NULL on error.
 */
float *pplx_embed(pplx_ctx_t *ctx, const int *token_ids, int n_tokens);

/*
 * Compute one embedding into caller-provided out[hidden_size].
 * Returns 0 on success, -1 on error.
 */
int pplx_embed_into(pplx_ctx_t *ctx, const int *token_ids, int n_tokens,
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
int pplx_embed_batch(pplx_ctx_t *ctx, const pplx_input_t *inputs, int batch,
                     float *out_embeddings);

int pplx_model_embed_batch(const pplx_model_t *model, pplx_workspace_t *ws,
                           const pplx_input_t *inputs, int batch,
                           float *out_embeddings);
int pplx_model_embed_into(const pplx_model_t *model, pplx_workspace_t *ws,
                          const int *token_ids, int n_tokens,
                          float *out_embedding);
float *pplx_model_embed(const pplx_model_t *model, pplx_workspace_t *ws,
                        const int *token_ids, int n_tokens);

/*
 * Run the transformer forward pass WITHOUT pooling.
 * Returns the full [n_tokens * hidden_size] output after final RMSNorm.
 * Caller frees the returned buffer.  NULL on error.
 *
 * Useful for contextual (late-chunking) models where you need
 * per-token embeddings before splitting by separator positions.
 */
float *pplx_forward(pplx_ctx_t *ctx, const int *token_ids, int n_tokens);

/*
 * Run the transformer forward pass into caller-provided
 * out_states[n_tokens * hidden_size]. Returns 0 on success, -1 on error.
 */
int pplx_forward_into(pplx_ctx_t *ctx, const int *token_ids, int n_tokens,
                      float *out_states);

int pplx_model_forward_into(const pplx_model_t *model, pplx_workspace_t *ws,
                            const int *token_ids, int n_tokens,
                            float *out_states);
float *pplx_model_forward(const pplx_model_t *model, pplx_workspace_t *ws,
                          const int *token_ids, int n_tokens);

/*
 * Cosine similarity between two L2-normalized vectors.
 */
float pplx_cosine_similarity(const float *a, const float *b, int dim);

/* Verbose level: 0=quiet, 1=info, 2=debug */
extern int pplx_verbose;
extern int qwen_verbose;

#endif /* PPLX_EMBED_H */
