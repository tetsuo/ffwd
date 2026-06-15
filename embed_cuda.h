/*
 * embed_cuda.h - CUDA/cuBLAS backend
 */

#ifndef EMBED_CUDA_H
#define EMBED_CUDA_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying EMBED_API are exported
 * from the shared library; everything else stays internal. */
#ifndef EMBED_API
#if defined(__GNUC__)
#define EMBED_API __attribute__((visibility("default")))
#else
#define EMBED_API
#endif
#endif

#include "embed.h"

typedef struct embed_cuda_ctx embed_cuda_ctx_t;

EMBED_API int embed_cuda_set_fast_gemm(const char *mode);
EMBED_API int embed_cuda_set_weights_bf16(int on);

EMBED_API embed_cuda_ctx_t *embed_cuda_load(const char *model_dir);
EMBED_API void embed_cuda_free(embed_cuda_ctx_t *ctx);

EMBED_API const embed_config_t *embed_cuda_config(const embed_cuda_ctx_t *ctx);

EMBED_API int embed_cuda_encode_into(embed_cuda_ctx_t *ctx,
                                     const int *token_ids,
                                     int n_tokens,
                                     float *out_embedding);
EMBED_API float *embed_cuda_encode(embed_cuda_ctx_t *ctx, const int *token_ids, int n_tokens);
EMBED_API int embed_cuda_encode_batch(embed_cuda_ctx_t *ctx,
                                      const embed_input_t *inputs,
                                      int batch,
                                      float *out_embeddings);
EMBED_API int embed_cuda_encode_spans_batch(embed_cuda_ctx_t *ctx,
                                            const embed_context_input_t *inputs,
                                            int batch,
                                            float *out_embeddings);

#endif /* EMBED_CUDA_H */
