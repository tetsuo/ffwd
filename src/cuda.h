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
typedef struct embed_cuda_late_ctx embed_cuda_late_ctx_t;

EMBED_API int embed_cuda_set_fast_gemm(const char *mode);
EMBED_API int embed_cuda_set_weights_bf16(int on);

EMBED_API embed_cuda_ctx_t *embed_cuda_load(const char *model_dir);
EMBED_API void embed_cuda_free(embed_cuda_ctx_t *ctx);

/*
 * Late-interaction CUDA path. Encoding (transformer forward + projection) runs
 * on the GPU; MaxSim scoring stays on the host, where the grouped-GEMM scorer is
 * much faster than a device graph for this small op. The encoders therefore copy
 * the projected token vectors back to host memory.
 */
EMBED_API embed_cuda_late_ctx_t *embed_cuda_late_load(const char *model_dir);
EMBED_API void embed_cuda_late_free(embed_cuda_late_ctx_t *ctx);
EMBED_API const embed_config_t *embed_cuda_late_config(const embed_cuda_late_ctx_t *ctx);
EMBED_API int embed_cuda_late_token_dim(const embed_cuda_late_ctx_t *ctx);

/* Encode token_ids into out_vectors[n_tokens, token_dim] (host). */
EMBED_API int embed_cuda_late_encode_tokens(embed_cuda_late_ctx_t *ctx,
                                            const int *token_ids,
                                            int n_tokens,
                                            int normalize,
                                            float *out_vectors);

/*
 * Encode n_docs documents in one packed forward and pack each document's kept
 * token vectors back-to-back into out_vectors[total_keep, token_dim] (host) in
 * document order. out_offsets[n_docs + 1] receives the prefix sum of kept counts
 * (the layout embed_late_maxsim_batch consumes). Returns 0 on success.
 */
EMBED_API int embed_cuda_late_encode_docs(embed_cuda_late_ctx_t *ctx,
                                          const int *const *doc_ids,
                                          const int *n_tokens,
                                          const int *const *keep,
                                          const int *n_keep,
                                          int n_docs,
                                          int normalize,
                                          float *out_vectors,
                                          int *out_offsets);

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
