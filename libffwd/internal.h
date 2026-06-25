/*
 * internal.h - non-exported, in-tree API shared by the frontends,
 * backends, tests, and benchmarks. The public ABI is ffwd.h; none of these
 * symbols carry FFWD_API visibility or appear in the export maps.
 */

#ifndef FFWD_INTERNAL_H
#define FFWD_INTERNAL_H

#include "ffwd.h"

#include <stddef.h>
#include <stdio.h>

typedef struct ffwd_model ffwd_model_t;
typedef struct ffwd_workspace ffwd_workspace_t;
typedef struct ffwd_late_model ffwd_late_model_t;
typedef struct ffwd_late_workspace ffwd_late_workspace_t;

/* Backend-specific help text used by the in-tree frontends. */
void ffwd_cli_help(FILE *f);
void ffwd_server_help(FILE *f);

/*
 * Lower-level CPU model API. These functions are used by backend glue, tests,
 * benchmarks, and parity harnesses; they are not exported from libffwd.
 */
ffwd_model_t *ffwd_model_load(const char *model_dir);
void ffwd_model_free(ffwd_model_t *model);

ffwd_workspace_t *ffwd_workspace_new(const ffwd_model_t *model);
void ffwd_workspace_free(ffwd_workspace_t *ws);

const ffwd_config_t *ffwd_model_config(const ffwd_model_t *model);
size_t ffwd_workspace_nbytes(const ffwd_workspace_t *ws);

float *ffwd_model_encode(const ffwd_model_t *model,
                         ffwd_workspace_t *ws,
                         const int *token_ids,
                         int n_tokens);

int ffwd_model_encode_into(const ffwd_model_t *model,
                           ffwd_workspace_t *ws,
                           const int *token_ids,
                           int n_tokens,
                           float *out_embedding);

int ffwd_model_encode_batch(const ffwd_model_t *model,
                            ffwd_workspace_t *ws,
                            const ffwd_input_t *inputs,
                            int batch,
                            float *out_embeddings);

int ffwd_pool_batch(const ffwd_config_t *cfg,
                    const float *states,
                    const int *seq_lengths,
                    int batch,
                    float *out_embeddings);

float *ffwd_model_forward(const ffwd_model_t *model,
                          ffwd_workspace_t *ws,
                          const int *token_ids,
                          int n_tokens);

int ffwd_model_forward_into(const ffwd_model_t *model,
                            ffwd_workspace_t *ws,
                            const int *token_ids,
                            int n_tokens,
                            float *out_states);

int ffwd_pool_spans(const ffwd_config_t *cfg,
                    const float *states,
                    int n_tokens,
                    const ffwd_span_t *spans,
                    int n_spans,
                    float *out_embeddings);

int ffwd_model_encode_spans(const ffwd_model_t *model,
                            ffwd_workspace_t *ws,
                            const int *token_ids,
                            int n_tokens,
                            const ffwd_span_t *spans,
                            int n_spans,
                            float *out_embeddings);

int ffwd_model_encode_spans_batch(const ffwd_model_t *model,
                                  ffwd_workspace_t *ws,
                                  const ffwd_context_input_t *inputs,
                                  int batch,
                                  float *out_embeddings);

/*
 * Late-interaction CPU model API. The public engine path exposes reranking via
 * ffwd_rerank(); these entry points are for in-tree tests/backends only.
 */
ffwd_late_model_t *ffwd_late_model_load(const char *model_dir);
void ffwd_late_model_free(ffwd_late_model_t *model);

ffwd_late_workspace_t *ffwd_late_workspace_new(const ffwd_late_model_t *model);
void ffwd_late_workspace_free(ffwd_late_workspace_t *ws);

const ffwd_config_t *ffwd_late_model_config(const ffwd_late_model_t *model);
int ffwd_late_model_token_dim(const ffwd_late_model_t *model);

int ffwd_late_model_encode_tokens(const ffwd_late_model_t *model,
                                  ffwd_late_workspace_t *ws,
                                  const int *token_ids,
                                  int n_tokens,
                                  int normalize,
                                  float *out_vectors);

int ffwd_late_model_encode_docs(const ffwd_late_model_t *model,
                                ffwd_late_workspace_t *ws,
                                const int *const *doc_ids,
                                const int *n_tokens,
                                const int *const *keep,
                                const int *n_keep,
                                int n_docs,
                                int normalize,
                                float *out_vectors,
                                int *out_offsets);

float ffwd_late_maxsim(
    const float *query_vectors, int query_tokens, const float *doc_vectors, int doc_tokens, int dim);

int ffwd_late_maxsim_batch(const float *query_vectors,
                           int query_tokens,
                           const float *doc_vectors,
                           const int *doc_offsets,
                           int docs,
                           int dim,
                           float *scores);

#endif /* FFWD_INTERNAL_H */
