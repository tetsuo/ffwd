/* backend_cuda.c - NVIDIA GPU backend (CUDA), implements the engine API in ffwd.h.
 * Linked only into CUDA builds. The context is owned by the inference thread:
 * open records what to load, activate creates the context, worker_free destroys
 * it. CUDA encodes on the device and scores MaxSim on the host. */
#include "internal.h"
#include "cuda.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct embed {
    int is_late;
    char *dir;
    ffwd_cuda_ctx_t *ctx;
    ffwd_cuda_late_ctx_t *late_ctx;
};

const char *ffwd_capability(void) { return FFWD_BACKEND_LABEL; }

void ffwd_cli_help(FILE *f) {
    fputs("  --gpu-gemm-mode MODE\n"
          "               GPU GEMM compute mode: f32, tf32, bf16, or 16f (default: f32)\n"
          "  --gpu-weight-dtype DTYPE\n"
          "               GPU weight storage: f32 or bf16 (default: model dtype).\n"
          "               bf16 halves weight memory and uses BF16 tensor cores\n",
          f);
}

void ffwd_server_help(FILE *f) {
    fputs("  --gpu-gemm-mode MODE      GPU GEMM compute: f32, tf32, bf16, or 16f\n"
          "                            (default: f32, exact)\n"
          "  --gpu-weight-dtype DTYPE  GPU weight storage: f32 or bf16 (default:\n"
          "                            model dtype); bf16 halves weight memory\n",
          f);
}

int ffwd_init(const ffwd_options_t *opts, char *err, size_t errlen) {
    if (opts->gpu_gemm_mode && ffwd_cuda_set_fast_gemm(opts->gpu_gemm_mode) != 0) {
        snprintf(err, errlen, "--gpu-gemm-mode must be f32, tf32, bf16, or 16f");
        return -1;
    }
    if (opts->gpu_weight_dtype) {
        if (!strcmp(opts->gpu_weight_dtype, "bf16"))
            ffwd_cuda_set_weights_bf16(1);
        else if (!strcmp(opts->gpu_weight_dtype, "f32"))
            ffwd_cuda_set_weights_bf16(0);
        else {
            snprintf(err, errlen, "--gpu-weight-dtype must be f32 or bf16");
            return -1;
        }
    }
    if (ffwd_verbose >= 1)
        fprintf(stderr, "Using %s\n", FFWD_BACKEND_LABEL);
    return 0;
}

ffwd_t *ffwd_open(
    const char *model_dir, int is_late, const ffwd_options_t *opts, char *err, size_t errlen) {
    (void)opts;
    ffwd_t *b = (ffwd_t *)calloc(1, sizeof(*b));
    if (!b || !(b->dir = strdup(model_dir))) {
        snprintf(err, errlen, "out of memory");
        free(b);
        return NULL;
    }
    b->is_late = is_late;
    return b;
}

int ffwd_activate(ffwd_t *b, char *err, size_t errlen) {
    if (b->is_late)
        b->late_ctx = ffwd_cuda_late_load(b->dir);
    else
        b->ctx = ffwd_cuda_load(b->dir);
    if (!b->ctx && !b->late_ctx) {
        snprintf(err, errlen, "failed to load model");
        return -1;
    }
    return 0;
}

void ffwd_worker_free(ffwd_t *b) {
    if (!b)
        return;
    if (b->ctx) {
        ffwd_cuda_free(b->ctx);
        b->ctx = NULL;
    }
    if (b->late_ctx) {
        ffwd_cuda_late_free(b->late_ctx);
        b->late_ctx = NULL;
    }
}

void ffwd_free(ffwd_t *b) {
    if (!b)
        return;
    ffwd_worker_free(b);
    free(b->dir);
    free(b);
}

const ffwd_config_t *ffwd_config(const ffwd_t *b) {
    return b->is_late ? ffwd_cuda_late_config(b->late_ctx) : ffwd_cuda_config(b->ctx);
}

int ffwd_token_dim(const ffwd_t *b) {
    return b->is_late ? ffwd_cuda_late_token_dim(b->late_ctx) : 0;
}

int ffwd_uses_dense_batches(const ffwd_t *b) {
    (void)b;
    return 0;
}

int ffwd_encode_batch(ffwd_t *b, const ffwd_input_t *inputs, int batch, float *out) {
    return ffwd_cuda_encode_batch(b->ctx, inputs, batch, out);
}

int ffwd_encode_spans_batch(ffwd_t *b,
                               const ffwd_context_input_t *inputs,
                               int batch,
                               float *out) {
    return ffwd_cuda_encode_spans_batch(b->ctx, inputs, batch, out);
}

int ffwd_rerank(ffwd_t *b, const ffwd_rerank_input_t *in, float *scores) {
    int dim = ffwd_cuda_late_token_dim(b->late_ctx);
    int total_doc_vecs = 0;
    for (int i = 0; i < in->n_docs; i++)
        total_doc_vecs += in->doc_n_keep[i];

    float *query = (float *)malloc((size_t)in->query_n_tokens * dim * sizeof(*query));
    float *docs = (float *)malloc((size_t)total_doc_vecs * dim * sizeof(*docs));
    int *offsets = (int *)malloc((size_t)(in->n_docs + 1) * sizeof(*offsets));
    int rc = (query && docs && offsets) ? 0 : -1;

    /* GPU encodes the query and every candidate in one packed forward; MaxSim
     * runs on the host, where the grouped-GEMM scorer beats a device graph. */
    if (rc == 0)
        rc =
            ffwd_cuda_late_encode_tokens(b->late_ctx, in->query_ids, in->query_n_tokens, 1, query);
    if (rc == 0)
        rc = ffwd_cuda_late_encode_docs(b->late_ctx, in->doc_ids, in->doc_n_tokens, in->doc_keep,
                                           in->doc_n_keep, in->n_docs, 1, docs, offsets);
    if (rc == 0)
        rc = ffwd_late_maxsim_batch(query, in->query_n_keep, docs, offsets, in->n_docs, dim,
                                       scores);
    free(offsets);
    free(docs);
    free(query);
    return rc;
}

size_t ffwd_scratch_nbytes(const ffwd_t *b) {
    (void)b;
    return 0;
}

int ffwd_default_batch_wait_us(void) { return 1000; }

int ffwd_preflight(
    const char *const *paths, int n_paths, const ffwd_options_t *opts, char *err, size_t errlen) {
    (void)paths;
    (void)n_paths;
    (void)opts;
    (void)err;
    (void)errlen;
    return 0;
}
