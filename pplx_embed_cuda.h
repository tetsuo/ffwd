/*
 * pplx_embed_cuda.h - CUDA/cuBLAS backend for pplx-embed inference.
 */

#ifndef PPLX_EMBED_CUDA_H
#define PPLX_EMBED_CUDA_H

#include "pplx_embed.h"

typedef struct pplx_cuda_ctx pplx_cuda_ctx_t;

int pplx_cuda_set_fast_gemm(const char *mode);

pplx_cuda_ctx_t *pplx_cuda_load(const char *model_dir);
void pplx_cuda_free(pplx_cuda_ctx_t *ctx);

const pplx_config_t *pplx_cuda_config(const pplx_cuda_ctx_t *ctx);

int pplx_cuda_embed_into(pplx_cuda_ctx_t *ctx, const int *token_ids,
                         int n_tokens, float *out_embedding);
float *pplx_cuda_embed(pplx_cuda_ctx_t *ctx, const int *token_ids,
                       int n_tokens);
int pplx_cuda_embed_batch(pplx_cuda_ctx_t *ctx, const pplx_input_t *inputs,
                          int batch, float *out_embeddings);
int pplx_cuda_embed_spans_batch(pplx_cuda_ctx_t *ctx,
                                const pplx_context_input_t *inputs,
                                int batch, float *out_embeddings);

#endif /* PPLX_EMBED_CUDA_H */
