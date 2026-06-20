/*
 * ffwd_cuda.h - CUDA/cuBLAS backend
 */

#ifndef FFWD_CUDA_H
#define FFWD_CUDA_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying FFWD_API are exported
 * from the shared library; everything else stays internal. */
#ifndef FFWD_API
#    if defined(__GNUC__)
#        define FFWD_API __attribute__((visibility("default")))
#    else
#        define FFWD_API
#    endif
#endif

#include "ffwd.h"

typedef struct ffwd_cuda_ctx ffwd_cuda_ctx_t;
typedef struct ffwd_cuda_late_ctx ffwd_cuda_late_ctx_t;

FFWD_API int ffwd_cuda_set_fast_gemm(const char *mode);
FFWD_API int ffwd_cuda_set_weights_bf16(int on);

FFWD_API ffwd_cuda_ctx_t *ffwd_cuda_load(const char *model_dir);
FFWD_API void ffwd_cuda_free(ffwd_cuda_ctx_t *ctx);

/*
 * Late-interaction CUDA path. Encoding (transformer forward + projection) runs
 * on the GPU; MaxSim scoring stays on the host, where the grouped-GEMM scorer is
 * much faster than a device graph for this small op. The encoders therefore copy
 * the projected token vectors back to host memory.
 */
FFWD_API ffwd_cuda_late_ctx_t *ffwd_cuda_late_load(const char *model_dir);
FFWD_API void ffwd_cuda_late_free(ffwd_cuda_late_ctx_t *ctx);
FFWD_API const ffwd_config_t *ffwd_cuda_late_config(const ffwd_cuda_late_ctx_t *ctx);
FFWD_API int ffwd_cuda_late_token_dim(const ffwd_cuda_late_ctx_t *ctx);

/* Encode token_ids into out_vectors[n_tokens, token_dim] (host). */
FFWD_API int ffwd_cuda_late_encode_tokens(ffwd_cuda_late_ctx_t *ctx,
                                            const int *token_ids,
                                            int n_tokens,
                                            int normalize,
                                            float *out_vectors);

/*
 * Encode n_docs documents in one packed forward and pack each document's kept
 * token vectors back-to-back into out_vectors[total_keep, token_dim] (host) in
 * document order. out_offsets[n_docs + 1] receives the prefix sum of kept counts
 * (the layout ffwd_late_maxsim_batch consumes). Returns 0 on success.
 */
FFWD_API int ffwd_cuda_late_encode_docs(ffwd_cuda_late_ctx_t *ctx,
                                          const int *const *doc_ids,
                                          const int *n_tokens,
                                          const int *const *keep,
                                          const int *n_keep,
                                          int n_docs,
                                          int normalize,
                                          float *out_vectors,
                                          int *out_offsets);

FFWD_API const ffwd_config_t *ffwd_cuda_config(const ffwd_cuda_ctx_t *ctx);

FFWD_API int ffwd_cuda_encode_into(ffwd_cuda_ctx_t *ctx,
                                     const int *token_ids,
                                     int n_tokens,
                                     float *out_embedding);
FFWD_API float *ffwd_cuda_encode(ffwd_cuda_ctx_t *ctx, const int *token_ids, int n_tokens);
FFWD_API int ffwd_cuda_encode_batch(ffwd_cuda_ctx_t *ctx,
                                      const ffwd_input_t *inputs,
                                      int batch,
                                      float *out_embeddings);
FFWD_API int ffwd_cuda_encode_spans_batch(ffwd_cuda_ctx_t *ctx,
                                            const ffwd_context_input_t *inputs,
                                            int batch,
                                            float *out_embeddings);

#endif /* FFWD_CUDA_H */
