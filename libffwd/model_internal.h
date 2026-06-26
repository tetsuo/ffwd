/* Cross-file declarations private to CPU model code.
 *
 * model_types.h holds the shared struct definitions; this header holds the
 * shared functions.
 *
 * These are not part of the public API (ffwd.h) and do not use FFWD_API
 * visibility. They are shared between translation units that used to be one
 * file: model loading, the forward pass, and late interaction.
 *
 * Frontends and GPU backends do not include this header.
 */

#ifndef FFWD_MODEL_INTERNAL_H
#define FFWD_MODEL_INTERNAL_H

#include "model_types.h"
#include "alloc.h"

/* Validate that a safetensor has the expected dtype, rank, and shape, and that
 * its bytes are within the mapped file. bf16_ok also allows either 16-bit float
 * format, BF16 or F16, in addition to F32. */
int tensor_has_supported_shape(const safetensors_file_t *sf,
                               const safetensor_t *t,
                               const char *name,
                               const int64_t *shape,
                               int ndim,
                               int bf16_ok);

/* Build a weight ref from a located tensor.
 * F32/BF16 borrow the mmap; F16 is widened to an owned F32 buffer because there
 * is no fused F16 kernel.
 *
 * Shared by the model loader and late-interaction projection loader.
 * The owner frees ref.data when ref.owned is set.
 *
 * Returns ref.data == NULL on allocation failure. */
ffwd_weight_ref_t weight_ref_from_tensor(const safetensors_file_t *sf, const safetensor_t *t);

/* True if model_dir carries a 1_Dense projection (a late-interaction model). */
int model_dir_has_late_projection(const char *model_dir);

/* Load (a layer range of) a model. allow_late lets the late loader accept a
 * 1_Dense snapshot that the pooled loader rejects. layer_end == -1 means all. */
ffwd_model_t *
model_load_range_ex(const char *model_dir, int layer_start, int layer_end, int allow_late);

/* Workspace scratch grows lazily per forward call.
 * Defined in model.c beside the workspace lifecycle. ffwd_workspace_nbytes mirrors this sizing.
 * Called from the forward pass in forward.c. */
int ensure_buffers(ffwd_workspace_t *ws, const ffwd_config_t *c, int seq);
void ensure_attention_scores(ffwd_workspace_t *ws, const int *offsets, int batch);
int ensure_rope_cache(ffwd_workspace_t *ws, const ffwd_config_t *cfg, int n_pos);
int ensure_offsets(ffwd_workspace_t *ws, int batch);

/* y[seq,out] = x[seq,in] @ w^T, no bias; dispatches F32/BF16 paths. Shared by
 * the forward pass and the late projection. */
void linear_nobias_weight(ffwd_workspace_t *ws,
                          float *y,
                          const float *x,
                          const ffwd_weight_ref_t *w,
                          int seq_len,
                          int in_dim,
                          int out_dim);

/* Compute packed sequence offsets [batch+1] plus total and max sequence length
 * over a batch of inputs. require_ids rejects inputs with a NULL id pointer. */
int build_offsets(const ffwd_input_t *inputs,
                  int batch,
                  int require_ids,
                  int *offsets,
                  int *total_out,
                  int *max_out);

/* Run the full transformer forward over a packed batch in ws, leaving hidden
 * states in ws->x. Used by pooled encoding and by late-interaction encoding. */
int forward_packed_inplace(const ffwd_model_t *model,
                           ffwd_workspace_t *ws,
                           const ffwd_input_t *inputs,
                           int batch,
                           const int *offsets,
                           int total_seq,
                           int max_seq,
                           int apply_final_norm);

#endif /* FFWD_MODEL_INTERNAL_H */
