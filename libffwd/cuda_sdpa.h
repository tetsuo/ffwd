/* cuDNN-backed fused scaled-dot-product attention for the CUDA backend.
 *
 * Wraps the cuDNN graph API (via the vendored cudnn-frontend headers) behind a
 * small C interface so cuda.cu can dispatch packed variable-length attention
 * to NVIDIA's fused flash kernels. libcudnn is dlopen'd at runtime: when it is
 * absent or too old, ffwd_sdpa_create returns NULL and the caller keeps the
 * built-in kernels. Builds without cuDNN headers compile this interface to
 * stubs (see cuda_sdpa.cu).
 */
#ifndef FFWD_CUDA_SDPA_H
#define FFWD_CUDA_SDPA_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ffwd_sdpa ffwd_sdpa_t;

/* Returns NULL when fused SDPA is unavailable (built without cuDNN headers,
 * libcudnn missing at runtime, or an unsupported cuDNN/driver combination). */
ffwd_sdpa_t *ffwd_sdpa_create(void);
void ffwd_sdpa_free(ffwd_sdpa_t *s);

/* Fused SDPA forward over a packed variable-length batch.
 *
 * Layout (all device pointers, 16-bit elements; bf16 when is_f16 == 0, f16
 * otherwise):
 *   q, o: [total_tokens, hq * d], token-major, head then dim within a token
 *   k, v: [total_tokens, hkv * d] (compact GQA: one row per kv head)
 * Softmax and accumulation run in f32 inside the fused kernel.
 *
 * Sequence metadata (device int32, prepared by the caller):
 *   seq_len:    [batch]      per-sequence token counts (0 for padded entries)
 *   ragged_q:   [batch + 1]  prefix offsets in ELEMENTS of q/o rows
 *                            (offsets[b] * hq * d), last entry = total * hq * d
 *   ragged_kv:  [batch + 1]  same for k/v rows (offsets[b] * hkv * d)
 * batch and s_max are the graph dimensions and may exceed the live batch /
 * longest sequence (bucketing): padded batch entries carry seq_len 0 and a
 * repeated final ragged offset.
 *
 * Returns 0 on success; -1 on an execution error; -2 when this shape is
 * unsupported by the installed cuDNN (negative-cached, cheap to re-ask).
 * On nonzero the caller must fall back to the built-in kernels. */
int ffwd_sdpa_run(ffwd_sdpa_t *s,
                  cudaStream_t stream,
                  const void *q,
                  const void *k,
                  const void *v,
                  void *o,
                  const int32_t *seq_len,
                  const int32_t *ragged_q,
                  const int32_t *ragged_kv,
                  int batch,
                  int s_max,
                  int hq,
                  int hkv,
                  int d,
                  float scale,
                  int causal,
                  int is_f16);

#ifdef __cplusplus
}
#endif

#endif /* FFWD_CUDA_SDPA_H */
