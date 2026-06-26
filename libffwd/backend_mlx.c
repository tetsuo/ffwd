/* Metal backend.
 * Implements the API in ffwd.h and is linked only into Apple Metal builds.
 *
 * MLX streams are thread-local, so the context is created and destroyed on the
 * inference thread.
 * open only records what to load; activate creates the context; worker_free
 * destroys it.
 */

#include "internal.h"
#include "mlx.h"
#include "platform.h"
#include "safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef __APPLE__
#    include <sys/sysctl.h>
#endif

/* Peak resident memory is estimated as the on-disk tensor payload times this
 * factor; the preflight refuses a model set whose estimate exceeds the budget. */
#define FFWD_MLX_RESIDENT_MULTIPLIER   2
#define FFWD_MLX_MEMORY_BUDGET_PERCENT 90

struct embed {
    int is_late;
    char *dir;
    ffwd_options_t opts;
    ffwd_mlx_ctx_t *ctx;
    ffwd_mlx_late_ctx_t *late_ctx;
};

const char *ffwd_capability(void) { return FFWD_BACKEND_LABEL; }

void ffwd_cli_help(FILE *f) {
    fputs("  --gpu-quant-bits N\n"
          "               Quantize GPU linear weights to 8 bits at load time\n"
          "  --gpu-quant-group-size N\n"
          "               GPU quantization group size (default: 64)\n",
          f);
}

void ffwd_server_help(FILE *f) {
    fputs("  --memory-utilization F    Fraction of physical memory the GPU model-set\n"
          "                            preflight may plan for (default: 0.90; values\n"
          "                            above 1.0 overcommit)\n"
          "  --gpu-quant-bits N        Quantize GPU linear weights to 8 bits at load\n"
          "  --gpu-quant-group-size N  GPU quantization group size (default: 64)\n",
          f);
}

int ffwd_init(const ffwd_options_t *opts, char *err, size_t errlen) {
    if (opts->gpu_quant_bits != 0 && opts->gpu_quant_bits != 8) {
        snprintf(err, errlen, "--gpu-quant-bits must be 0 or 8");
        return -1;
    }
    if (opts->gpu_quant_group_size <= 0) {
        snprintf(err, errlen, "--gpu-quant-group-size must be > 0");
        return -1;
    }
    if (ffwd_verbose >= 1)
        fprintf(stderr, "Using %s\n", FFWD_BACKEND_LABEL);
    return 0;
}

ffwd_t *
ffwd_open(const char *model_dir, int is_late, const ffwd_options_t *opts, char *err, size_t errlen) {
    ffwd_t *b = (ffwd_t *)calloc(1, sizeof(*b));
    if (!b || !(b->dir = strdup(model_dir))) {
        snprintf(err, errlen, "out of memory");
        free(b);
        return NULL;
    }
    b->is_late = is_late;
    b->opts = *opts;
    return b;
}

int ffwd_activate(ffwd_t *b, char *err, size_t errlen) {
    ffwd_mlx_options_t mlx_opts = {
        .quantize_bits = b->opts.gpu_quant_bits,
        .quantize_group_size = b->opts.gpu_quant_group_size,
    };
    if (b->is_late)
        b->late_ctx = ffwd_mlx_late_load_with_options(b->dir, &mlx_opts);
    else
        b->ctx = ffwd_mlx_load_with_options(b->dir, &mlx_opts);
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
        ffwd_mlx_free(b->ctx);
        b->ctx = NULL;
    }
    if (b->late_ctx) {
        ffwd_mlx_late_free(b->late_ctx);
        b->late_ctx = NULL;
    }
}

void ffwd_free(ffwd_t *b) {
    if (!b)
        return;
    /* The inference thread releases the context via ffwd_worker_free;
     * by the time the main thread frees the handle both are already NULL. The
     * single-threaded CLI loads and frees on one thread, so this also releases
     * a context that worker_free never ran on. */
    ffwd_worker_free(b);
    free(b->dir);
    free(b);
}

const ffwd_config_t *ffwd_config(const ffwd_t *b) {
    return b->is_late ? ffwd_mlx_late_config(b->late_ctx) : ffwd_mlx_config(b->ctx);
}

int ffwd_token_dim(const ffwd_t *b) { return b->is_late ? ffwd_mlx_late_token_dim(b->late_ctx) : 0; }

int ffwd_uses_dense_batches(const ffwd_t *b) { return b->ctx != NULL; }

int ffwd_encode_batch(ffwd_t *b, const ffwd_input_t *inputs, int batch, float *out) {
    return ffwd_mlx_encode_batch(b->ctx, inputs, batch, out);
}

int ffwd_encode_spans_batch(ffwd_t *b, const ffwd_context_input_t *inputs, int batch, float *out) {
    return ffwd_mlx_encode_spans_batch(b->ctx, inputs, batch, out);
}

int ffwd_rerank(ffwd_t *b, const ffwd_rerank_input_t *in, float *scores) {
    int *offsets = (int *)malloc((size_t)(in->n_docs + 1) * sizeof(*offsets));
    ffwd_mlx_late_vectors_t *query =
        ffwd_mlx_late_encode_tokens_device(b->late_ctx, in->query_ids, in->query_n_tokens, 1);
    ffwd_mlx_late_vectors_t *packed =
        (query && offsets)
            ? ffwd_mlx_late_encode_docs_device(b->late_ctx, in->doc_ids, in->doc_n_tokens,
                                               in->doc_keep, in->doc_n_keep, in->n_docs, 1, offsets)
            : NULL;
    int rc = packed ? ffwd_mlx_late_maxsim_batch_device(b->late_ctx, query, packed, offsets,
                                                        in->n_docs, scores)
                    : -1;
    ffwd_mlx_late_vectors_free(packed);
    ffwd_mlx_late_vectors_free(query);
    free(offsets);
    return rc;
}

size_t ffwd_scratch_nbytes(const ffwd_t *b) {
    (void)b;
    return 0;
}

int ffwd_default_batch_wait_us(void) { return 0; }

static uint64_t physical_memory_nbytes(void) {
#ifdef __APPLE__
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) == 0 && len == sizeof(bytes) && bytes > 0)
        return bytes;
#endif
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0 || (uint64_t)pages > UINT64_MAX / (uint64_t)page_size)
        return 0;
    return (uint64_t)pages * (uint64_t)page_size;
}

int ffwd_preflight(
    const char *const *paths, int n_paths, const ffwd_options_t *opts, char *err, size_t errlen) {
    uint64_t payload = 0;
    for (int i = 0; i < n_paths; i++) {
        multi_safetensors_t *ms = multi_safetensors_open(paths[i]);
        size_t model_payload = 0;
        if (!ms || multi_safetensors_data_nbytes(ms, &model_payload) != 0) {
            multi_safetensors_close(ms);
            snprintf(err, errlen, "failed to inspect GPU model: %s", paths[i]);
            return -1;
        }
        multi_safetensors_close(ms);
        if ((uint64_t)model_payload > UINT64_MAX - payload) {
            snprintf(err, errlen, "GPU model payload size overflow");
            return -1;
        }
        payload += (uint64_t)model_payload;
    }
    if (payload > UINT64_MAX / FFWD_MLX_RESIDENT_MULTIPLIER) {
        snprintf(err, errlen, "GPU resident-memory estimate overflow");
        return -1;
    }
    uint64_t estimated = payload * FFWD_MLX_RESIDENT_MULTIPLIER;
    uint64_t physical = physical_memory_nbytes();
    if (physical == 0) {
        fprintf(stderr, "ffwd-server: warning: could not determine physical memory; "
                        "skipping GPU memory preflight\n");
        return 0;
    }
    double utilization = opts->memory_utilization > 0.0 ? opts->memory_utilization
                                                        : FFWD_MLX_MEMORY_BUDGET_PERCENT / 100.0;
    uint64_t budget = (uint64_t)((double)physical * utilization);
    const double gib = 1024.0 * 1024.0 * 1024.0;
    fprintf(stderr,
            "ffwd-server: GPU memory preflight: %.1f GiB tensors, %.1f GiB estimated "
            "resident, %.1f GiB budget (%.2f of physical memory)\n",
            (double)payload / gib, (double)estimated / gib, (double)budget / gib, utilization);
    if (opts->gpu_quant_bits)
        fprintf(stderr,
                "ffwd-server: GPU %d-bit quantization enabled; preflight uses source "
                "payload as a conservative peak estimate\n",
                opts->gpu_quant_bits);
    if (estimated <= budget)
        return 0;
    snprintf(err, errlen,
             "ffwd-server: GPU model set exceeds the host-memory budget; use BF16 artifacts, "
             "load fewer models, or raise --memory-utilization");
    return -1;
}
