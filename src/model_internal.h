#ifndef EMBED_MODEL_INTERNAL_H
#define EMBED_MODEL_INTERNAL_H

#include "internal.h"
#include "alloc.h"

/* Cross-file function declarations private to the CPU model code. The companion
 * to internal.h: that header holds the shared struct definitions, this one the
 * shared functions. None are part of the public API (embed.h) and none carry
 * EMBED_API visibility; they are shared between the translation units (model
 * loading, the forward pass, and late interaction) that used to live in one
 * file. The frontends and the GPU backends do not include this header. */

/* Validate that a safetensor matches an expected dtype/rank/shape and that its
 * bytes lie within the mapped file. bf16_ok permits either 16-bit float format
 * (BF16 or F16) in addition to F32. */
int tensor_has_supported_shape(const safetensors_file_t *sf,
                               const safetensor_t *t,
                               const char *name,
                               const int64_t *shape,
                               int ndim,
                               int bf16_ok);

/* Build a weight ref from a located tensor: F32/BF16 borrow the mmap, F16 is
 * widened to an owned F32 buffer (no fused F16 kernel). Shared by the model
 * loader and the late-interaction projection loader; the owner frees ref.data
 * when ref.owned is set. Returns ref.data == NULL on allocation failure. */
embed_weight_ref_t weight_ref_from_tensor(const safetensors_file_t *sf, const safetensor_t *t);

/* True if model_dir carries a 1_Dense projection (a late-interaction model). */
int model_dir_has_late_projection(const char *model_dir);

/* Load (a layer range of) a model. allow_late lets the late loader accept a
 * 1_Dense snapshot that the pooled loader rejects. layer_end == -1 means all. */
embed_model_t *
model_load_range_ex(const char *model_dir, int layer_start, int layer_end, int allow_late);

/* Workspace scratch growth, lazy per forward call. Defined in model.c beside the
 * workspace lifecycle (embed_workspace_nbytes mirrors their sizing); called from
 * the forward pass in forward.c. */
int ensure_buffers(embed_workspace_t *ws, const embed_config_t *c, int seq);
void ensure_attention_scores(embed_workspace_t *ws, const int *offsets, int batch);
int ensure_rope_cache(embed_workspace_t *ws, const embed_config_t *cfg, int n_pos);
int ensure_offsets(embed_workspace_t *ws, int batch);

/* y[seq,out] = x[seq,in] @ w^T, no bias; dispatches F32/BF16 paths. Shared by
 * the forward pass and the late projection. */
void linear_nobias_weight(embed_workspace_t *ws,
                          float *y,
                          const float *x,
                          const embed_weight_ref_t *w,
                          int seq_len,
                          int in_dim,
                          int out_dim);

/* Compute packed sequence offsets [batch+1] plus total and max sequence length
 * over a batch of inputs. require_ids rejects inputs with a NULL id pointer. */
int build_offsets(const embed_input_t *inputs,
                  int batch,
                  int require_ids,
                  int *offsets,
                  int *total_out,
                  int *max_out);

/* Run the full transformer forward over a packed batch in ws, leaving hidden
 * states in ws->x. Used by pooled encoding and by late-interaction encoding. */
int forward_packed_inplace(const embed_model_t *model,
                           embed_workspace_t *ws,
                           const embed_input_t *inputs,
                           int batch,
                           const int *offsets,
                           int total_seq,
                           int max_seq,
                           int apply_final_norm);

#endif /* EMBED_MODEL_INTERNAL_H */
