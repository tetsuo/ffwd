/* bench/bench_model.c - end-to-end embedding throughput against real
 * weights, for regression tracking (kernels are covered by bench_kernels.c;
 * this catches slowdowns in the orchestration between them).
 *
 *   make bench-model MODEL_DIR=/path/to/pplx-embed-v1-0.6b
 *
 * Uses deterministic synthetic token ids, so only config.json and
 * model.safetensors are needed (no tokenizer). Single-threaded by default;
 * --threads N opts into the pool.
 */

#include "bench.h"
#include "internal.h"
#include "threadpool.h"

static ffwd_model_t *g_model;
static ffwd_workspace_t *g_ws;
static int g_hidden;

enum { MAX_BATCH = 8 };

static void bm_ffwd(bench_state_t *b, int batch, int len) {
    int *ids = (int *)malloc((size_t)batch * len * sizeof(int));
    float *out = (float *)malloc((size_t)batch * g_hidden * sizeof(float));
    if (!ids || !out)
        exit(2);
    unsigned s = 99u;
    for (int i = 0; i < batch * len; i++) {
        s = s * 1664525u + 1013904223u;
        ids[i] = (int)(s % 1000u); /* ordinary vocab rows */
    }
    ffwd_input_t inputs[MAX_BATCH];
    for (int i = 0; i < batch; i++) {
        inputs[i].ids = ids + (size_t)i * len;
        inputs[i].n_tokens = len;
    }
    bench_begin(b);
    for (long i = 0; i < b->n; i++) {
        if (ffwd_model_encode_batch(g_model, g_ws, inputs, batch, out) != 0)
            exit(2);
        bench_sink += out[0];
    }
    free(out);
    free(ids);
}

static void bm_ffwd_b1_len32(bench_state_t *b) { bm_ffwd(b, 1, 32); }
static void bm_ffwd_b8_len32(bench_state_t *b) { bm_ffwd(b, 8, 32); }
static void bm_ffwd_b1_len256(bench_state_t *b) { bm_ffwd(b, 1, 256); }

static const bench_case_t CASES[] = {
    {"ffwd/b1_len32", bm_ffwd_b1_len32},
    {"ffwd/b8_len32", bm_ffwd_b8_len32},
    {"ffwd/b1_len256", bm_ffwd_b1_len256},
};

int main(int argc, char **argv) {
    if (argc < 2 || argv[1][0] == '-') {
        fprintf(stderr, "usage: %s MODEL_DIR [--threads N] [bench flags]\n", argv[0]);
        return 2;
    }

    int threads = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            threads = atoi(argv[i + 1]);
    }
    if (threads < 1)
        threads = 1;
    tp_set_threads(threads);

    g_model = ffwd_model_load(argv[1]);
    if (!g_model) {
        fprintf(stderr, "failed to load model from %s\n", argv[1]);
        return 1;
    }
    g_ws = ffwd_workspace_new(g_model);
    if (!g_ws) {
        ffwd_model_free(g_model);
        return 1;
    }
    g_hidden = ffwd_model_config(g_model)->hidden_size;

    char meta[160];
    snprintf(meta, sizeof(meta), "suite=model threads=%d hidden=%d", threads, g_hidden);

    bench_opts_t opts = {0};
    opts.meta = meta;
    bench_parse_args(&opts, argc - 1, argv + 1);
    int rc = bench_main(&opts, CASES, (int)(sizeof(CASES) / sizeof(CASES[0])));

    ffwd_workspace_free(g_ws);
    ffwd_model_free(g_model);
    return rc;
}
