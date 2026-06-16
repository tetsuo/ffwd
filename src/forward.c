#include "embed.h"
#include "model_internal.h"
#include "dtype.h"
#include "kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/* Below these packed-sequence lengths the fused BF16 row kernels beat widening
 * the weight to F32 and dispatching through BLAS; above them, widen once. */
#define EMBED_BF16_QKV_FUSE_MAX_SEQ  16
#define EMBED_BF16_PAIR_FUSE_MAX_SEQ 4

/* ========================================================================
 * Linear projections (weight-dtype dispatch) and workspace scratch growth
 * ======================================================================== */

static void bf16_row_to_f32(float *dst, const uint16_t *src, int n) {
    for (int i = 0; i < n; i++)
        dst[i] = embed_bf16_to_f32(src[i]);
}

static void copy_weight_row(float *dst, const embed_weight_ref_t *w, size_t row, int row_width) {
    if (w->dtype == DTYPE_F32) {
        const float *src = (const float *)w->data + row * (size_t)row_width;
        memcpy(dst, src, (size_t)row_width * sizeof(float));
    } else {
        const uint16_t *src = (const uint16_t *)w->data + row * (size_t)row_width;
        bf16_row_to_f32(dst, src, row_width);
    }
}

static int ensure_bf16_widen(embed_workspace_t *ws, size_t count) {
    if (ws->bf16_widen_count >= count)
        return 0;
    float *p = (float *)realloc(ws->bf16_widen, count * sizeof(float));
    if (!p)
        return -1;
    ws->bf16_widen = p;
    ws->bf16_widen_count = count;
    return 0;
}

void linear_nobias_weight(embed_workspace_t *ws,
                          float *y,
                          const float *x,
                          const embed_weight_ref_t *w,
                          int seq_len,
                          int in_dim,
                          int out_dim) {
    if (w->dtype == DTYPE_F32) {
        embed_linear_nobias(y, x, (const float *)w->data, seq_len, in_dim, out_dim);
        return;
    }
    if (seq_len > 16) {
        /* Widen the weight matrix once into workspace scratch and use the
         * BLAS-backed F32 path; below that the fused BF16 row kernels win. */
        size_t count = (size_t)out_dim * (size_t)in_dim;
        if (ensure_bf16_widen(ws, count) == 0) {
            embed_bf16_to_f32_buf(ws->bf16_widen, (const uint16_t *)w->data, count);
            embed_linear_nobias(y, x, ws->bf16_widen, seq_len, in_dim, out_dim);
            return;
        }
    }
    embed_linear_nobias_bf16(y, x, (const uint16_t *)w->data, seq_len, in_dim, out_dim);
}

static void linear_qkv_weight(embed_workspace_t *ws,
                              float *q,
                              float *k,
                              float *v,
                              const float *x,
                              const embed_weight_ref_t *wq,
                              const embed_weight_ref_t *wk,
                              const embed_weight_ref_t *wv,
                              int seq_len,
                              int in_dim,
                              int q_dim,
                              int kv_dim) {
    if (wq->dtype == DTYPE_BF16 && wk->dtype == DTYPE_BF16 && wv->dtype == DTYPE_BF16 &&
        seq_len <= EMBED_BF16_QKV_FUSE_MAX_SEQ) {
        embed_linear_nobias_bf16_qkv(q, k, v, x, (const uint16_t *)wq->data,
                                     (const uint16_t *)wk->data, (const uint16_t *)wv->data,
                                     seq_len, in_dim, q_dim, kv_dim);
        return;
    }

    linear_nobias_weight(ws, q, x, wq, seq_len, in_dim, q_dim);
    linear_nobias_weight(ws, k, x, wk, seq_len, in_dim, kv_dim);
    linear_nobias_weight(ws, v, x, wv, seq_len, in_dim, kv_dim);
}

static void add_bias_rows(float *x, const float *bias, int rows, int dim) {
    if (!bias)
        return;
    for (int row = 0; row < rows; row++) {
        float *dst = x + (size_t)row * dim;
        for (int d = 0; d < dim; d++)
            dst[d] += bias[d];
    }
}

static void linear_pair_weight(embed_workspace_t *ws,
                               float *a,
                               float *b,
                               const float *x,
                               const embed_weight_ref_t *wa,
                               const embed_weight_ref_t *wb,
                               int seq_len,
                               int in_dim,
                               int a_dim,
                               int b_dim) {
    if (wa->dtype == DTYPE_BF16 && wb->dtype == DTYPE_BF16 &&
        seq_len <= EMBED_BF16_PAIR_FUSE_MAX_SEQ) {
        embed_linear_nobias_bf16_pair(a, b, x, (const uint16_t *)wa->data,
                                      (const uint16_t *)wb->data, seq_len, in_dim, a_dim, b_dim);
        return;
    }

    linear_nobias_weight(ws, a, x, wa, seq_len, in_dim, a_dim);
    linear_nobias_weight(ws, b, x, wb, seq_len, in_dim, b_dim);
}

/* ========================================================================
 * Packed forward pass
 *
 * The workspace-growth helpers these call (ensure_buffers, ensure_rope_cache,
 * ensure_attention_scores, ensure_offsets) live in model.c beside the workspace
 * lifecycle and embed_workspace_nbytes, whose size accounting mirrors them.
 * ======================================================================== */

int build_offsets(const embed_input_t *inputs,
                  int batch,
                  int require_ids,
                  int *offsets,
                  int *total_out,
                  int *max_out) {
    if (!inputs || batch <= 0 || !offsets || !total_out || !max_out)
        return -1;

    int total = 0;
    int max_seq = 0;
    for (int b = 0; b < batch; b++) {
        if ((require_ids && !inputs[b].ids) || inputs[b].n_tokens <= 0)
            return -1;
        if (total > INT32_MAX - inputs[b].n_tokens)
            return -1;
        offsets[b] = total;
        total += inputs[b].n_tokens;
        if (inputs[b].n_tokens > max_seq)
            max_seq = inputs[b].n_tokens;
    }
    offsets[batch] = total;
    *total_out = total;
    *max_out = max_seq;
    return 0;
}

static int forward_packed_slice_inplace(const embed_model_t *model,
                                        embed_workspace_t *ws,
                                        const embed_input_t *inputs,
                                        int batch,
                                        const int *offsets,
                                        int total_seq,
                                        int max_seq,
                                        const float *input_states,
                                        int layer_start,
                                        int layer_end,
                                        int apply_final_norm) {
    if (!model || !ws || ws->model != model || !inputs || batch <= 0 || !offsets ||
        total_seq <= 0 || max_seq <= 0)
        return -1;

    const embed_config_t *cfg = &model->config;
    const embed_weights_t *w = &model->weights;
    if (layer_start < 0 || layer_start > layer_end || layer_end > cfg->n_layers ||
        layer_start < model->layer_start || layer_end > model->layer_end ||
        (layer_start == 0 && input_states) || (layer_start != 0 && !input_states) ||
        (layer_start == 0 && !w->embed_tokens.data) ||
        (apply_final_norm && (layer_end != cfg->n_layers || !w->norm)))
        return -1;

    int hidden = cfg->hidden_size;
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim = cfg->head_dim;
    int q_dim = cfg->q_dim;
    int kv_dim = cfg->kv_dim;
    int inter = cfg->intermediate_size;
    float eps = cfg->rms_norm_eps;

    if (ensure_buffers(ws, cfg, total_seq) != 0)
        return -1;
    if (ensure_rope_cache(ws, cfg, max_seq) != 0)
        return -1;
    ensure_attention_scores(ws, offsets, batch);

    const float *rope_cos = ws->rope_cos;
    const float *rope_sin = ws->rope_sin;

    float *x = ws->x;
    float *x_norm = ws->x_norm;
    float *q_buf = ws->q;
    float *k_buf = ws->k;
    float *v_buf = ws->v;
    float *attn_out = ws->attn_out;
    float *proj_out = ws->proj_out;
    float *ffn_gate = ws->ffn_gate;
    float *ffn_up = ws->ffn_up;

    /* A first stage performs embedding lookup. Later stages resume from a
     * packed hidden-state tensor received from the preceding stage. */
    if (layer_start == 0) {
        for (int b = 0; b < batch; b++) {
            int start = offsets[b];
            for (int i = 0; i < inputs[b].n_tokens; i++) {
                int id = inputs[b].ids[i];
                if (id < 0 || id >= cfg->vocab_size) {
                    fprintf(stderr, "embed: invalid token id %d at input %d position %d\n", id, b,
                            i);
                    return -1;
                }
                copy_weight_row(x + (size_t)(start + i) * hidden, &w->embed_tokens, (size_t)id,
                                hidden);
            }
        }
    } else {
        memcpy(x, input_states, (size_t)total_seq * hidden * sizeof(float));
    }

    float scale = 1.0f / sqrtf((float)head_dim);

    /* 2. Transformer layers. Linear/MLP ops run over all packed tokens;
     * attention is explicitly block-diagonal by sequence offsets. */
    for (int layer = layer_start; layer < layer_end; layer++) {
        const embed_layer_t *l = &w->layers[layer];

        embed_rms_norm(x_norm, x, l->input_norm, total_seq, hidden, eps);

        linear_qkv_weight(ws, q_buf, k_buf, v_buf, x_norm, &l->wq, &l->wk, &l->wv, total_seq,
                          hidden, q_dim, kv_dim);

        add_bias_rows(q_buf, l->q_bias, total_seq, q_dim);
        add_bias_rows(k_buf, l->k_bias, total_seq, kv_dim);
        add_bias_rows(v_buf, l->v_bias, total_seq, kv_dim);
        if (cfg->qk_norm) {
            embed_rms_norm_per_head(q_buf, l->q_norm, total_seq, n_heads, head_dim, eps);
            embed_rms_norm_per_head(k_buf, l->k_norm, total_seq, n_kv_heads, head_dim, eps);
        }

        for (int b = 0; b < batch; b++) {
            int start = offsets[b];
            int len = offsets[b + 1] - start;
            embed_apply_rope_neox(q_buf + (size_t)start * q_dim, rope_cos, rope_sin, len, n_heads,
                                  head_dim);
            embed_apply_rope_neox(k_buf + (size_t)start * kv_dim, rope_cos, rope_sin, len,
                                  n_kv_heads, head_dim);
        }

        model->attention(attn_out, q_buf, k_buf, v_buf, offsets, batch, n_heads, n_kv_heads,
                         head_dim, scale, ws->attn_scores, ws->attn_scores_bytes);

        linear_nobias_weight(ws, proj_out, attn_out, &l->wo, total_seq, q_dim, hidden);
        embed_add_inplace(x, proj_out, total_seq * hidden);

        embed_rms_norm(x_norm, x, l->post_attn_norm, total_seq, hidden, eps);

        linear_pair_weight(ws, ffn_gate, ffn_up, x_norm, &l->gate_proj, &l->up_proj, total_seq,
                           hidden, inter, inter);
        embed_silu_mul_inplace(ffn_gate, ffn_up, total_seq * inter);
        linear_nobias_weight(ws, proj_out, ffn_gate, &l->down_proj, total_seq, inter, hidden);
        embed_add_inplace(x, proj_out, total_seq * hidden);
    }

    if (apply_final_norm)
        embed_rms_norm(x, x, w->norm, total_seq, hidden, eps);
    return 0;
}

int forward_packed_inplace(const embed_model_t *model,
                           embed_workspace_t *ws,
                           const embed_input_t *inputs,
                           int batch,
                           const int *offsets,
                           int total_seq,
                           int max_seq,
                           int apply_final_norm) {
    return forward_packed_slice_inplace(model, ws, inputs, batch, offsets, total_seq, max_seq, NULL,
                                        0, model->config.n_layers, apply_final_norm);
}

/* BERT-family forward: token + learned-position + token-type embeddings and an
 * embedding LayerNorm, then post-norm encoder layers (bias on every projection,
 * GeLU feed-forward, no RoPE, no per-head norm). Bidirectional attention is
 * block-diagonal over the packed batch, so each sequence's positions restart at
 * cfg->position_id_offset (0 for BERT, pad_id + 1 for RoBERTa/XLM-R). Leaves the
 * final hidden states in ws->x; pooling reads them as-is because the last
 * layer's LayerNorm is already the model's final normalization. */
static int forward_packed_bert(const embed_model_t *model,
                               embed_workspace_t *ws,
                               const embed_input_t *inputs,
                               int batch,
                               const int *offsets,
                               int total_seq,
                               int max_seq) {
    if (!model || !ws || ws->model != model || !inputs || batch <= 0 || !offsets ||
        total_seq <= 0 || max_seq <= 0)
        return -1;

    const embed_config_t *cfg = &model->config;
    const embed_weights_t *w = &model->weights;
    int hidden = cfg->hidden_size;
    int n_heads = cfg->n_heads;
    int head_dim = cfg->head_dim;
    int inter = cfg->intermediate_size;
    float eps = cfg->layer_norm_eps;

    if (!w->embed_tokens.data || !w->position_embeddings || !w->token_type_embeddings ||
        !w->embed_ln_w || !w->embed_ln_b)
        return -1;
    if (max_seq > cfg->max_position_embeddings - cfg->position_id_offset) {
        fprintf(stderr,
                "embed: sequence length %d plus position offset %d exceeds max_position_embeddings %d\n",
                max_seq, cfg->position_id_offset, cfg->max_position_embeddings);
        return -1;
    }
    if (ensure_buffers(ws, cfg, total_seq) != 0)
        return -1;
    ensure_attention_scores(ws, offsets, batch);

    float *x = ws->x;
    float *q_buf = ws->q, *k_buf = ws->k, *v_buf = ws->v;
    float *attn_out = ws->attn_out, *proj_out = ws->proj_out, *ffn_up = ws->ffn_up;

    /* Embeddings: token + learned absolute position + token-type[0], then the
     * embedding LayerNorm over every packed token. */
    for (int b = 0; b < batch; b++) {
        int start = offsets[b];
        for (int i = 0; i < inputs[b].n_tokens; i++) {
            int id = inputs[b].ids[i];
            if (id < 0 || id >= cfg->vocab_size) {
                fprintf(stderr, "embed: invalid token id %d at input %d position %d\n", id, b, i);
                return -1;
            }
            float *row = x + (size_t)(start + i) * hidden;
            copy_weight_row(row, &w->embed_tokens, (size_t)id, hidden);
            int pos = cfg->position_id_offset + i;
            embed_add_inplace(row, w->position_embeddings + (size_t)pos * hidden, hidden);
            embed_add_inplace(row, w->token_type_embeddings, hidden);
        }
    }
    embed_layer_norm(x, x, w->embed_ln_w, w->embed_ln_b, total_seq, hidden, eps);

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->n_layers; layer++) {
        const embed_layer_t *l = &w->layers[layer];

        /* Self-attention (post-norm): projections read x directly. */
        linear_qkv_weight(ws, q_buf, k_buf, v_buf, x, &l->wq, &l->wk, &l->wv, total_seq, hidden,
                          hidden, hidden);
        add_bias_rows(q_buf, l->q_bias, total_seq, hidden);
        add_bias_rows(k_buf, l->k_bias, total_seq, hidden);
        add_bias_rows(v_buf, l->v_bias, total_seq, hidden);

        model->attention(attn_out, q_buf, k_buf, v_buf, offsets, batch, n_heads, n_heads, head_dim,
                         scale, ws->attn_scores, ws->attn_scores_bytes);

        linear_nobias_weight(ws, proj_out, attn_out, &l->wo, total_seq, hidden, hidden);
        add_bias_rows(proj_out, l->o_bias, total_seq, hidden);
        embed_add_inplace(x, proj_out, total_seq * hidden);
        embed_layer_norm(x, x, l->input_norm, l->attn_ln_bias, total_seq, hidden, eps);

        /* Feed-forward (post-norm): dense -> GeLU -> dense. */
        linear_nobias_weight(ws, ffn_up, x, &l->up_proj, total_seq, hidden, inter);
        add_bias_rows(ffn_up, l->ffn_inter_bias, total_seq, inter);
        if (cfg->ffn_act == EMBED_ACT_GELU_TANH)
            embed_gelu_tanh_inplace(ffn_up, total_seq * inter);
        else
            embed_gelu_inplace(ffn_up, total_seq * inter);
        linear_nobias_weight(ws, proj_out, ffn_up, &l->down_proj, total_seq, inter, hidden);
        add_bias_rows(proj_out, l->ffn_out_bias, total_seq, hidden);
        embed_add_inplace(x, proj_out, total_seq * hidden);
        embed_layer_norm(x, x, l->post_attn_norm, l->ffn_ln_bias, total_seq, hidden, eps);
    }
    return 0;
}

static int pool_embeddings(const embed_model_t *model,
                           const embed_workspace_t *ws,
                           const int *offsets,
                           int batch,
                           float *out_embeddings) {
    const embed_config_t *cfg = &model->config;
    int hidden = cfg->hidden_size;
    float eps = cfg->rms_norm_eps;
    /* Qwen pools pre-final-norm states and applies the final RMSNorm here; BERT
     * pools already-normed states (no final norm), signalled by a NULL norm. */
    const float *norm_weight = model->weights.norm;
    const float *x = ws->x;
    if (hidden <= 0 || !x)
        return -1;

    for (int b = 0; b < batch; b++) {
        int start = offsets[b];
        int end = offsets[b + 1];
        int len = end - start;
        float *emb = out_embeddings + (size_t)b * hidden;

        if (cfg->pooling_mode == EMBED_POOL_LAST_TOKEN || cfg->pooling_mode == EMBED_POOL_CLS) {
            /* Single-token pooling: CLS takes the first token, last-token the
             * final one. */
            int idx = cfg->pooling_mode == EMBED_POOL_CLS ? start : end - 1;
            const float *row = x + (size_t)idx * hidden;
            if (norm_weight) {
                float sum_sq = 0.0f;
                for (int d = 0; d < hidden; d++)
                    sum_sq += row[d] * row[d];
                float rms_inv = 1.0f / sqrtf(sum_sq / (float)hidden + eps);
                for (int d = 0; d < hidden; d++)
                    emb[d] = row[d] * rms_inv * norm_weight[d];
            } else {
                memcpy(emb, row, (size_t)hidden * sizeof(float));
            }
        } else {
            memset(emb, 0, (size_t)hidden * sizeof(float));
            for (int i = start; i < end; i++) {
                const float *row = x + (size_t)i * hidden;
                if (norm_weight) {
                    float sum_sq = 0.0f;
                    for (int d = 0; d < hidden; d++)
                        sum_sq += row[d] * row[d];
                    float rms_inv = 1.0f / sqrtf(sum_sq / (float)hidden + eps);
                    for (int d = 0; d < hidden; d++)
                        emb[d] += row[d] * rms_inv * norm_weight[d];
                } else {
                    for (int d = 0; d < hidden; d++)
                        emb[d] += row[d];
                }
            }
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < hidden; d++)
                emb[d] *= inv_len;
        }

        if (cfg->normalize_embeddings && embed_l2_normalize(emb, hidden) != 0)
            return -1;
    }
    return 0;
}

/* ========================================================================
 * Public forward/embed APIs
 * ======================================================================== */

int embed_model_forward_into(const embed_model_t *model,
                             embed_workspace_t *ws,
                             const int *token_ids,
                             int n_tokens,
                             float *out_states) {
    if (!model || !ws || !token_ids || n_tokens <= 0 || !out_states)
        return -1;

    size_t count, bytes;
    if (mul_size((size_t)n_tokens, (size_t)model->config.hidden_size, &count) != 0 ||
        mul_size(count, sizeof(float), &bytes) != 0)
        return -1;

    embed_input_t input = {token_ids, n_tokens};
    int offsets[2] = {0, n_tokens};
    if (forward_packed_inplace(model, ws, &input, 1, offsets, n_tokens, n_tokens, 1) != 0)
        return -1;

    memcpy(out_states, ws->x, bytes);
    return 0;
}

int embed_pool_spans(const embed_config_t *cfg,
                     const float *states,
                     int n_tokens,
                     const embed_span_t *spans,
                     int n_spans,
                     float *out_embeddings) {
    if (!cfg || !states || n_tokens <= 0 || !spans || n_spans <= 0 || !out_embeddings)
        return -1;

    int hidden = cfg->hidden_size;
    if (hidden <= 0)
        return -1;

    for (int s = 0; s < n_spans; s++) {
        int start = spans[s].start;
        int len = spans[s].n_tokens;
        if (start < 0 || len <= 0 || start > n_tokens || len > n_tokens - start)
            return -1;

        float *emb = out_embeddings + (size_t)s * hidden;
        memset(emb, 0, (size_t)hidden * sizeof(float));

        for (int t = 0; t < len; t++) {
            const float *row = states + (size_t)(start + t) * hidden;
            for (int d = 0; d < hidden; d++)
                emb[d] += row[d];
        }

        float inv_len = 1.0f / (float)len;
        for (int d = 0; d < hidden; d++) {
            emb[d] *= inv_len;
        }
    }
    return 0;
}

int embed_model_encode_spans_batch(const embed_model_t *model,
                                   embed_workspace_t *ws,
                                   const embed_context_input_t *inputs,
                                   int batch,
                                   float *out_embeddings) {
    if (!model || !ws || !inputs || batch <= 0 || !out_embeddings)
        return -1;

    if (ensure_offsets(ws, batch) != 0)
        return -1;
    if ((size_t)batch > SIZE_MAX / sizeof(embed_input_t))
        return -1;
    embed_input_t *packed_inputs = malloc((size_t)batch * sizeof(*packed_inputs));
    if (!packed_inputs)
        return -1;
    int *offsets = ws->offsets;
    int total_seq = 0;
    int max_seq = 0;
    int total_spans = 0;
    for (int b = 0; b < batch; b++) {
        const embed_context_input_t *input = &inputs[b];
        int n_tokens = input->input.n_tokens;
        if (!input->input.ids || n_tokens <= 0 || !input->spans || input->n_spans <= 0 ||
            total_seq > INT_MAX - n_tokens || total_spans > INT_MAX - input->n_spans) {
            free(packed_inputs);
            return -1;
        }
        packed_inputs[b] = input->input;
        offsets[b] = total_seq;
        total_seq += n_tokens;
        total_spans += input->n_spans;
        if (n_tokens > max_seq)
            max_seq = n_tokens;
        for (int s = 0; s < input->n_spans; s++) {
            int start = input->spans[s].start;
            int len = input->spans[s].n_tokens;
            if (start < 0 || len <= 0 || start > n_tokens || len > n_tokens - start) {
                free(packed_inputs);
                return -1;
            }
        }
    }
    offsets[batch] = total_seq;

    if (forward_packed_inplace(model, ws, packed_inputs, batch, offsets, total_seq, max_seq, 1) !=
        0) {
        free(packed_inputs);
        return -1;
    }
    free(packed_inputs);

    int span_offset = 0;
    int hidden = model->config.hidden_size;
    for (int b = 0; b < batch; b++) {
        const embed_context_input_t *input = &inputs[b];
        if (embed_pool_spans(&model->config, ws->x + (size_t)offsets[b] * hidden,
                             input->input.n_tokens, input->spans, input->n_spans,
                             out_embeddings + (size_t)span_offset * hidden) != 0)
            return -1;
        span_offset += input->n_spans;
    }
    return 0;
}

int embed_model_encode_spans(const embed_model_t *model,
                             embed_workspace_t *ws,
                             const int *token_ids,
                             int n_tokens,
                             const embed_span_t *spans,
                             int n_spans,
                             float *out_embeddings) {
    embed_context_input_t input = {
        .input = {token_ids, n_tokens},
        .spans = spans,
        .n_spans = n_spans,
    };
    return embed_model_encode_spans_batch(model, ws, &input, 1, out_embeddings);
}

float *embed_model_forward(const embed_model_t *model,
                           embed_workspace_t *ws,
                           const int *token_ids,
                           int n_tokens) {
    if (!model || !ws || !token_ids || n_tokens <= 0)
        return NULL;

    size_t n;
    if (mul_size((size_t)n_tokens, (size_t)model->config.hidden_size, &n) != 0)
        return NULL;
    float *out = malloc_floats(n);
    if (!out)
        return NULL;

    if (embed_model_forward_into(model, ws, token_ids, n_tokens, out) != 0) {
        free(out);
        return NULL;
    }
    return out;
}

int embed_model_encode_batch(const embed_model_t *model,
                             embed_workspace_t *ws,
                             const embed_input_t *inputs,
                             int batch,
                             float *out_embeddings) {
    if (!model || !ws || !inputs || batch <= 0 || !out_embeddings)
        return -1;

    if (ensure_offsets(ws, batch) != 0)
        return -1;
    int *offsets = ws->offsets;

    int total_seq = 0;
    int max_seq = 0;
    if (build_offsets(inputs, batch, 1, offsets, &total_seq, &max_seq) != 0)
        return -1;

    int rc = model->config.family == EMBED_FAMILY_BERT
                 ? forward_packed_bert(model, ws, inputs, batch, offsets, total_seq, max_seq)
                 : forward_packed_inplace(model, ws, inputs, batch, offsets, total_seq, max_seq, 0);
    if (rc == 0)
        rc = pool_embeddings(model, ws, offsets, batch, out_embeddings);

    return rc;
}

int embed_pool_batch(const embed_config_t *cfg,
                     const float *states,
                     const int *seq_lengths,
                     int batch,
                     float *out_embeddings) {
    if (!cfg || !states || !seq_lengths || batch <= 0 || !out_embeddings || cfg->hidden_size <= 0)
        return -1;

    int offset = 0;
    for (int b = 0; b < batch; b++) {
        int len = seq_lengths[b];
        if (len <= 0 || offset > INT_MAX - len)
            return -1;

        float *emb = out_embeddings + (size_t)b * cfg->hidden_size;
        if (cfg->pooling_mode == EMBED_POOL_LAST_TOKEN || cfg->pooling_mode == EMBED_POOL_CLS) {
            /* CLS takes the first token, last-token the final one; these states
             * already carry the model's final RMSNorm. */
            int idx = cfg->pooling_mode == EMBED_POOL_CLS ? offset : offset + len - 1;
            const float *row = states + (size_t)idx * cfg->hidden_size;
            memcpy(emb, row, (size_t)cfg->hidden_size * sizeof(float));
        } else {
            memset(emb, 0, (size_t)cfg->hidden_size * sizeof(float));
            for (int t = 0; t < len; t++) {
                const float *row = states + (size_t)(offset + t) * cfg->hidden_size;
                for (int d = 0; d < cfg->hidden_size; d++)
                    emb[d] += row[d];
            }
            float inv_len = 1.0f / (float)len;
            for (int d = 0; d < cfg->hidden_size; d++)
                emb[d] *= inv_len;
        }
        if (cfg->normalize_embeddings && embed_l2_normalize(emb, cfg->hidden_size) != 0)
            return -1;
        offset += len;
    }
    return 0;
}

int embed_model_encode_into(const embed_model_t *model,
                            embed_workspace_t *ws,
                            const int *token_ids,
                            int n_tokens,
                            float *out_embedding) {
    embed_input_t input = {token_ids, n_tokens};
    return embed_model_encode_batch(model, ws, &input, 1, out_embedding);
}

float *embed_model_encode(const embed_model_t *model,
                          embed_workspace_t *ws,
                          const int *token_ids,
                          int n_tokens) {
    if (!model || !ws || !token_ids || n_tokens <= 0)
        return NULL;

    int hidden = model->config.hidden_size;
    if (hidden <= 0)
        return NULL;
    float *emb = malloc_floats((size_t)hidden);
    if (!emb)
        return NULL;

    if (embed_model_encode_into(model, ws, token_ids, n_tokens, emb) != 0) {
        free(emb);
        return NULL;
    }
    return emb;
}
