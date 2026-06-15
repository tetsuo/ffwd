#ifndef EMBED_INTERNAL_H
#define EMBED_INTERNAL_H

#include "embed.h"
#include "qwen_safetensors.h"

typedef struct {
    safetensor_dtype_t dtype;
    const void *data;
} embed_weight_ref_t;

typedef struct {
    embed_weight_ref_t wq;        /* [q_dim,   hidden] */
    embed_weight_ref_t wk;        /* [kv_dim,  hidden] */
    embed_weight_ref_t wv;        /* [kv_dim,  hidden] */
    embed_weight_ref_t wo;        /* [hidden,  q_dim]  */
    const float *q_norm;          /* [head_dim] */
    const float *k_norm;          /* [head_dim] */
    const float *input_norm;      /* [hidden] */
    const float *post_attn_norm;  /* [hidden] */
    embed_weight_ref_t gate_proj; /* [intermediate, hidden] */
    embed_weight_ref_t up_proj;   /* [intermediate, hidden] */
    embed_weight_ref_t down_proj; /* [hidden, intermediate] */
} embed_layer_t;

typedef struct {
    embed_weight_ref_t embed_tokens; /* [vocab_size, hidden] */
    embed_layer_t *layers;           /* [n_layers] heap-allocated */
    const float *norm;               /* [hidden] */
} embed_weights_t;

typedef void (*embed_attention_fn)(float *out,
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

struct embed_model {
    embed_config_t config;
    embed_weights_t weights;
    void *safetensors; /* multi_safetensors_t*, keeps mmap alive */
    int layer_start;   /* loaded transformer range, inclusive */
    int layer_end;     /* loaded transformer range, exclusive */
    embed_attention_fn attention;
};

struct embed_workspace {
    const embed_model_t *model; /* immutable model this workspace belongs to */
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

#endif /* EMBED_INTERNAL_H */
