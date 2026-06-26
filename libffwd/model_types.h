/* In-memory layout of a loaded model: model/workspace structs, per-layer weight
 * references, and accessors for GPU backends.
 *
 * Shared by CPU model code and the CUDA backend. Not part of the public ABI
 * (ffwd.h).
 *
 * model_internal.h declares the CPU model's private functions.
 */

#ifndef FFWD_MODEL_TYPES_H
#define FFWD_MODEL_TYPES_H

#include "internal.h"
#include "safetensors.h"

typedef struct {
    safetensor_dtype_t dtype;
    const void *data;
    /* 0: data is a borrowed view into the safetensors mmap (F32/BF16, the common
     * case). 1: data is a heap buffer owned by the model and freed on unload.
     * Set when an F16 tensor is widened to F32 at load (no fused F16 kernel). */
    int owned;
} ffwd_weight_ref_t;

typedef struct {
    ffwd_weight_ref_t wq;        /* [q_dim,   hidden] */
    ffwd_weight_ref_t wk;        /* [kv_dim,  hidden] */
    ffwd_weight_ref_t wv;        /* [kv_dim,  hidden] */
    ffwd_weight_ref_t wo;        /* [hidden,  q_dim]  */
    const float *q_bias;         /* [q_dim], optional */
    const float *k_bias;         /* [kv_dim], optional */
    const float *v_bias;         /* [kv_dim], optional */
    const float *q_norm;         /* [head_dim] */
    const float *k_norm;         /* [head_dim] */
    const float *input_norm;     /* [hidden] */
    const float *post_attn_norm; /* [hidden] */
    ffwd_weight_ref_t gate_proj; /* [intermediate, hidden] */
    ffwd_weight_ref_t up_proj;   /* [intermediate, hidden] */
    ffwd_weight_ref_t down_proj; /* [hidden, intermediate] */
    /* BERT family (all NULL for Qwen3). The attention projections reuse
     * wq/wk/wv/wo and q/k/v_bias; the two FFN matrices reuse up_proj
     * (intermediate.dense) and down_proj (output.dense); the two block
     * LayerNorm weights reuse input_norm (post-attention) and post_attn_norm
     * (post-FFN). These fields add the biases those reused slots lack. */
    const float *o_bias;         /* attention output dense bias [hidden] */
    const float *attn_ln_bias;   /* post-attention LayerNorm bias [hidden] */
    const float *ffn_inter_bias; /* intermediate.dense bias [intermediate] */
    const float *ffn_out_bias;   /* output.dense bias [hidden] */
    const float *ffn_ln_bias;    /* post-FFN LayerNorm bias [hidden] */
} ffwd_layer_t;

typedef struct {
    ffwd_weight_ref_t embed_tokens; /* [vocab_size, hidden] */
    ffwd_layer_t *layers;           /* [n_layers] heap-allocated */
    const float *norm;              /* [hidden]; NULL for BERT (no final norm) */
    /* BERT family (NULL for Qwen3): learned absolute position embeddings,
     * token-type (segment) embeddings, and the embedding LayerNorm. */
    const float *position_embeddings;   /* [max_position_embeddings, hidden] */
    const float *token_type_embeddings; /* [type_vocab_size, hidden] */
    const float *ffwd_ln_w;             /* embedding LayerNorm weight [hidden] */
    const float *ffwd_ln_b;             /* embedding LayerNorm bias [hidden] */
} ffwd_weights_t;

typedef void (*ffwd_attention_fn)(float *out,
                                  const float *Q,
                                  const float *K,
                                  const float *V,
                                  const int *offsets,
                                  int batch,
                                  int n_heads,
                                  int n_kv_heads,
                                  int head_dim,
                                  float scale,
                                  float *scratch,
                                  size_t scratch_bytes);

struct ffwd_model {
    ffwd_config_t config;
    ffwd_weights_t weights;
    void *safetensors; /* multi_safetensors_t*, keeps mmap alive */
    int layer_start;   /* loaded transformer range, inclusive */
    int layer_end;     /* loaded transformer range, exclusive */
};

struct ffwd_workspace {
    const ffwd_model_t *model; /* immutable model this workspace belongs to */
    int buf_seq_cap;
    float *x;           /* [seq, hidden]       */
    float *x_norm;      /* [seq, hidden]       */
    float *q;           /* [seq, q_dim]        */
    float *k;           /* [seq, kv_dim]       */
    float *v;           /* [seq, kv_dim]       */
    float *attn_out;    /* [seq, q_dim]        */
    float *proj_out;    /* [seq, hidden]       */
    float *ffn_gate;    /* [seq, intermediate] */
    float *ffn_up;      /* [seq, intermediate] */
    float *attn_scores; /* reusable BLAS attention score tiles */
    size_t attn_scores_bytes;

    int *offsets; /* [batch + 1] packed sequence offsets */
    int offsets_cap;

    /* RoPE cosine/sine cache [n_pos, head_dim], grown lazily. */
    float *rope_cos;
    float *rope_sin;
    int rope_cache_cap;

    /* F32 view of one BF16 weight matrix for SGEMM dispatch, grown to the
     * largest projection. Workspace-owned so concurrent workspaces on one
     * model never share conversion scratch. */
    float *bf16_widen;
    size_t bf16_widen_count;
};

/* Internal accessors for the late-interaction model. GPU backends build a device
 * context from an already-loaded CPU late model because the model dir is a late
 * snapshot that the pooled loader rejects; these expose the base model and the
 * projection weight without making struct ffwd_late_model public. */
ffwd_model_t *ffwd_late_model_base(const ffwd_late_model_t *model);
const ffwd_weight_ref_t *ffwd_late_model_projection(const ffwd_late_model_t *model);

#endif /* FFWD_MODEL_TYPES_H */
