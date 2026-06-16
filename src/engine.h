#ifndef EMBED_ENGINE_H
#define EMBED_ENGINE_H

#include "internal.h"
#include "alloc.h"

/* Cross-file declarations internal to the CPU engine. These functions are not
 * part of the public API (embed.h) and carry no EMBED_API visibility; they are
 * shared between the engine translation units (model loading, the forward pass,
 * and late interaction) that used to live in one file. The frontends and the
 * GPU backends do not include this header. */

/* Validate that a safetensor matches an expected dtype/rank/shape and that its
 * bytes lie within the mapped file. bf16_ok permits BF16 in addition to F32. */
int tensor_has_supported_shape(const safetensors_file_t *sf,
                               const safetensor_t *t,
                               const char *name,
                               const int64_t *shape,
                               int ndim,
                               int bf16_ok);

/* True if model_dir carries a 1_Dense projection (a late-interaction model). */
int model_dir_has_late_projection(const char *model_dir);

/* Load (a layer range of) a model. allow_late lets the late loader accept a
 * 1_Dense snapshot that the pooled loader rejects. layer_end == -1 means all. */
embed_model_t *
model_load_range_ex(const char *model_dir, int layer_start, int layer_end, int allow_late);

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

#endif /* EMBED_ENGINE_H */
