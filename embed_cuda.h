/*
 * embed_cuda.h - CUDA/cuBLAS backend for pplx-embed inference.
 */

#ifndef PPLX_EMBED_CUDA_H
#define PPLX_EMBED_CUDA_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying PPLX_API are exported
 * from the shared library; everything else stays internal. */
#ifndef PPLX_API
#if defined(__GNUC__)
#define PPLX_API __attribute__((visibility("default")))
#else
#define PPLX_API
#endif
#endif

#include "embed.h"

typedef struct pplx_cuda_ctx pplx_cuda_ctx_t;

PPLX_API int pplx_cuda_set_fast_gemm(const char *mode);
PPLX_API int pplx_cuda_set_weights_bf16(int on);

PPLX_API pplx_cuda_ctx_t *pplx_cuda_load(const char *model_dir);
PPLX_API void pplx_cuda_free(pplx_cuda_ctx_t *ctx);

PPLX_API const pplx_config_t *pplx_cuda_config(const pplx_cuda_ctx_t *ctx);

PPLX_API int pplx_cuda_embed_into(pplx_cuda_ctx_t *ctx, const int *token_ids,
                         int n_tokens, float *out_embedding);
PPLX_API float *pplx_cuda_embed(pplx_cuda_ctx_t *ctx, const int *token_ids,
                       int n_tokens);
PPLX_API int pplx_cuda_embed_batch(pplx_cuda_ctx_t *ctx, const pplx_input_t *inputs,
                          int batch, float *out_embeddings);
PPLX_API int pplx_cuda_embed_spans_batch(pplx_cuda_ctx_t *ctx,
                                const pplx_context_input_t *inputs,
                                int batch, float *out_embeddings);

#endif /* PPLX_EMBED_CUDA_H */
