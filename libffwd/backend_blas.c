/* backend_blas.c - CPU backend (BLAS), implements the engine API in ffwd.h.
 * Linked only into CPU builds. The CPU model loads on the caller's thread, so
 * ffwd_open does the full load and activate/worker_free are no-ops. */
#include "internal.h"
#include "threadpool.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>

struct embed {
    int is_late;
    ffwd_model_t *model;
    ffwd_workspace_t *ws;
    ffwd_late_model_t *late;
    ffwd_late_workspace_t *late_ws;
};

const char *ffwd_capability(void) { return FFWD_BACKEND_LABEL; }

void ffwd_cli_help(FILE *f) {
    fputs("  -t, --threads N\n"
          "               CPU threads (default: available cores, cgroup-aware)\n",
          f);
}

void ffwd_server_help(FILE *f) {
    fputs("  -t, --threads N           CPU threads (default: available cores, cgroup-aware)\n", f);
}

int ffwd_init(const ffwd_options_t *opts, char *err, size_t errlen) {
    (void)err;
    (void)errlen;
    int n = opts->n_threads > 0 ? opts->n_threads : tp_get_num_cpus();
    tp_set_threads(n);
    if (ffwd_verbose >= 1)
        fprintf(stderr, "Using %d CPU thread(s)\n", n);
    return 0;
}

ffwd_t *ffwd_open(
    const char *model_dir, int is_late, const ffwd_options_t *opts, char *err, size_t errlen) {
    (void)opts;
    ffwd_t *b = (ffwd_t *)calloc(1, sizeof(*b));
    if (!b) {
        snprintf(err, errlen, "out of memory");
        return NULL;
    }
    b->is_late = is_late;
    if (is_late) {
        b->late = ffwd_late_model_load(model_dir);
        if (!b->late) {
            snprintf(err, errlen, "failed to load late model");
            free(b);
            return NULL;
        }
        b->late_ws = ffwd_late_workspace_new(b->late);
        if (!b->late_ws) {
            snprintf(err, errlen, "failed to allocate late workspace");
            ffwd_late_model_free(b->late);
            free(b);
            return NULL;
        }
    } else {
        b->model = ffwd_model_load(model_dir);
        if (!b->model) {
            snprintf(err, errlen, "failed to load model");
            free(b);
            return NULL;
        }
        b->ws = ffwd_workspace_new(b->model);
        if (!b->ws) {
            snprintf(err, errlen, "failed to allocate workspace");
            ffwd_model_free(b->model);
            free(b);
            return NULL;
        }
    }
    return b;
}

int ffwd_activate(ffwd_t *b, char *err, size_t errlen) {
    (void)b;
    (void)err;
    (void)errlen;
    return 0;
}

void ffwd_worker_free(ffwd_t *b) { (void)b; }

void ffwd_free(ffwd_t *b) {
    if (!b)
        return;
    ffwd_workspace_free(b->ws);
    ffwd_model_free(b->model);
    ffwd_late_workspace_free(b->late_ws);
    ffwd_late_model_free(b->late);
    free(b);
}

const ffwd_config_t *ffwd_config(const ffwd_t *b) {
    return b->is_late ? ffwd_late_model_config(b->late) : ffwd_model_config(b->model);
}

int ffwd_token_dim(const ffwd_t *b) {
    return b->is_late ? ffwd_late_model_token_dim(b->late) : 0;
}

int ffwd_uses_dense_batches(const ffwd_t *b) {
    (void)b;
    return 0;
}

int ffwd_encode_batch(ffwd_t *b, const ffwd_input_t *inputs, int batch, float *out) {
    return ffwd_model_encode_batch(b->model, b->ws, inputs, batch, out);
}

int ffwd_encode_spans_batch(ffwd_t *b,
                               const ffwd_context_input_t *inputs,
                               int batch,
                               float *out) {
    return ffwd_model_encode_spans_batch(b->model, b->ws, inputs, batch, out);
}

int ffwd_rerank(ffwd_t *b, const ffwd_rerank_input_t *in, float *scores) {
    int dim = ffwd_late_model_token_dim(b->late);
    int total_doc_vecs = 0;
    for (int i = 0; i < in->n_docs; i++)
        total_doc_vecs += in->doc_n_keep[i];

    float *query = (float *)malloc((size_t)in->query_n_tokens * dim * sizeof(*query));
    float *docs = (float *)malloc((size_t)total_doc_vecs * dim * sizeof(*docs));
    int *offsets = (int *)malloc((size_t)(in->n_docs + 1) * sizeof(*offsets));
    int rc = (query && docs && offsets) ? 0 : -1;

    /* Encode every candidate in one block-diagonal forward: the encoder packs
     * each document's kept token vectors back-to-back and fills offsets. */
    if (rc == 0)
        rc = ffwd_late_model_encode_tokens(b->late, b->late_ws, in->query_ids, in->query_n_tokens,
                                              1, query);
    if (rc == 0)
        rc = ffwd_late_model_encode_docs(b->late, b->late_ws, in->doc_ids, in->doc_n_tokens,
                                            in->doc_keep, in->doc_n_keep, in->n_docs, 1, docs,
                                            offsets);
    if (rc == 0)
        rc = ffwd_late_maxsim_batch(query, in->query_n_keep, docs, offsets, in->n_docs, dim,
                                       scores);
    free(offsets);
    free(docs);
    free(query);
    return rc;
}

size_t ffwd_scratch_nbytes(const ffwd_t *b) {
    return (b && b->ws) ? ffwd_workspace_nbytes(b->ws) : 0;
}

int ffwd_default_batch_wait_us(void) { return 0; }

int ffwd_preflight(
    const char *const *paths, int n_paths, const ffwd_options_t *opts, char *err, size_t errlen) {
    (void)paths;
    (void)n_paths;
    (void)opts;
    (void)err;
    (void)errlen;
    return 0;
}
