#ifndef PPLX_EMBED_INTERNAL_H
#define PPLX_EMBED_INTERNAL_H

#include "pplx_embed.h"
#include "qwen_safetensors.h"

typedef struct {
    safetensor_dtype_t dtype;
    const void *data;
} pplx_weight_ref_t;

typedef struct {
    pplx_weight_ref_t wq;        /* [q_dim,   hidden] */
    pplx_weight_ref_t wk;        /* [kv_dim,  hidden] */
    pplx_weight_ref_t wv;        /* [kv_dim,  hidden] */
    pplx_weight_ref_t wo;        /* [hidden,  q_dim]  */
    const float *q_norm;         /* [head_dim] */
    const float *k_norm;         /* [head_dim] */
    const float *input_norm;     /* [hidden] */
    const float *post_attn_norm; /* [hidden] */
    pplx_weight_ref_t gate_proj; /* [intermediate, hidden] */
    pplx_weight_ref_t up_proj;   /* [intermediate, hidden] */
    pplx_weight_ref_t down_proj; /* [hidden, intermediate] */
} pplx_layer_t;

typedef struct {
    pplx_weight_ref_t embed_tokens; /* [vocab_size, hidden] */
    pplx_layer_t *layers;           /* [n_layers] heap-allocated */
    const float *norm;              /* [hidden] */
} pplx_weights_t;

struct pplx_model {
    pplx_config_t  config;
    pplx_weights_t weights;
    void *safetensors;          /* multi_safetensors_t*, keeps mmap alive */
    int layer_start;            /* loaded transformer range, inclusive */
    int layer_end;              /* loaded transformer range, exclusive */
};

struct pplx_workspace {
    const pplx_model_t *model;  /* immutable model this workspace belongs to */
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
    float *attn_scores;         /* reusable BLAS attention score tiles */
    size_t attn_scores_bytes;

    int   *offsets;             /* [batch + 1] packed sequence offsets */
    int    offsets_cap;

    /* RoPE cosine/sine cache [n_pos, head_dim], grown lazily. */
    float *rope_cos;
    float *rope_sin;
    int    rope_cache_cap;
};

#endif /* PPLX_EMBED_INTERNAL_H */
