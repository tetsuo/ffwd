/* bench/bench_late_rerank.c - late-interaction MaxSim rerank benchmark.
 * Build via `make bench-late-rerank`; run with:
 *   ./bench_late_rerank MODEL_DIR CANDIDATES DOC_REPEAT RUNS
 * Encodes a query and N candidate documents to token vectors, then times the
 * packed MaxSim rerank. check it against bench_late_rerank.py, which builds and
 * runs this binary. */
#include "internal.h"
#ifdef CHECK_MLX
#    include "mlx.h"
#endif
#include "bpe.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QUERY_LENGTH    32
#define MASK_TOKEN_ID   151642
#define QUERY_PREFIX_ID 151669
#define DOC_PREFIX_ID   151670

typedef struct {
    int *ids;
    int n_ids;
    float *vecs;
    int n_vecs;
#ifdef CHECK_MLX
    ffwd_mlx_late_vectors_t *dev;
#endif
} late_item_t;

#ifdef CHECK_MLX
typedef ffwd_mlx_late_ctx_t test_model_t;
typedef void test_workspace_t;
static test_model_t *test_model_load(const char *dir) { return ffwd_mlx_late_load(dir); }
static void test_model_free(test_model_t *m) { ffwd_mlx_late_free(m); }
static test_workspace_t *test_workspace_new(test_model_t *m) {
    (void)m;
    return (test_workspace_t *)1;
}
static void test_workspace_free(test_workspace_t *ws) { (void)ws; }
static const ffwd_config_t *test_config(test_model_t *m) { return ffwd_mlx_late_config(m); }
static int test_token_dim(test_model_t *m) { return ffwd_mlx_late_token_dim(m); }
static int test_encode(
    test_model_t *m, test_workspace_t *ws, const int *ids, int n_ids, int normalize, float *out) {
    (void)ws;
    return ffwd_mlx_late_encode_tokens(m, ids, n_ids, normalize, out);
}
#else
typedef ffwd_late_model_t test_model_t;
typedef ffwd_late_workspace_t test_workspace_t;
static test_model_t *test_model_load(const char *dir) { return ffwd_late_model_load(dir); }
static void test_model_free(test_model_t *m) { ffwd_late_model_free(m); }
static test_workspace_t *test_workspace_new(test_model_t *m) { return ffwd_late_workspace_new(m); }
static void test_workspace_free(test_workspace_t *ws) { ffwd_late_workspace_free(ws); }
static const ffwd_config_t *test_config(test_model_t *m) { return ffwd_late_model_config(m); }
static int test_token_dim(test_model_t *m) { return ffwd_late_model_token_dim(m); }
static int test_encode(
    test_model_t *m, test_workspace_t *ws, const int *ids, int n_ids, int normalize, float *out) {
    return ffwd_late_model_encode_tokens(m, ws, ids, n_ids, normalize, out);
}
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int encode_with_prefix_id(
    tok_bpe_t *tok, int prefix_id, const char *text, int expand_query, late_item_t *out) {
    out->ids = tok_bpe_encode(tok, text, &out->n_ids);
    if (!out->ids || out->n_ids <= 0)
        return -1;

    int target_len = out->n_ids + 1;
    if (expand_query && target_len < QUERY_LENGTH)
        target_len = QUERY_LENGTH;

    int *tmp = (int *)realloc(out->ids, (size_t)target_len * sizeof(int));
    if (!tmp)
        return -1;
    out->ids = tmp;

    memmove(out->ids + 2, out->ids + 1, (size_t)(out->n_ids - 1) * sizeof(int));
    out->ids[1] = prefix_id;
    out->n_ids++;

    if (expand_query && out->n_ids < QUERY_LENGTH) {
        for (int i = out->n_ids; i < QUERY_LENGTH; i++)
            out->ids[i] = MASK_TOKEN_ID;
        out->n_ids = QUERY_LENGTH;
    }
    return 0;
}

static int encode_late_item(test_model_t *model, test_workspace_t *ws, late_item_t *item, int dim) {
    item->n_vecs = item->n_ids;
    item->vecs = (float *)malloc((size_t)item->n_vecs * dim * sizeof(float));
    if (!item->vecs)
        return -1;
    if (test_encode(model, ws, item->ids, item->n_ids, 1, item->vecs) != 0)
        return -1;
#ifdef CHECK_MLX
    item->dev = ffwd_mlx_late_encode_tokens_device(model, item->ids, item->n_ids, 1);
    if (!item->dev)
        return -1;
#endif
    return 0;
}

static void free_item(late_item_t *item) {
#ifdef CHECK_MLX
    ffwd_mlx_late_vectors_free(item->dev);
#endif
    free(item->ids);
    free(item->vecs);
}

static char *make_doc_text(int index, int repeat) {
    static const char *base[] = {"Scientists explore the universe driven by curiosity",
                                 "Children learn through curious exploration",
                                 "Historical discoveries began with careful questions",
                                 "Redis stores data structures in memory for fast access",
                                 "SQLite keeps relational data in a local database file",
                                 "Embedding models map text into semantic vector spaces",
                                 "Paris is a capital city and a major European destination",
                                 "Neural search systems compare query and document representations"};
    enum { NBASE = 8 };
    size_t cap = 64;
    for (int i = 0; i < repeat; i++)
        cap += strlen(base[(index + i) % NBASE]) + 2;
    char *out = (char *)malloc(cap);
    if (!out)
        return NULL;
    out[0] = '\0';
    for (int i = 0; i < repeat; i++) {
        if (i)
            strcat(out, ". ");
        strcat(out, base[(index + i) % NBASE]);
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL_DIR CANDIDATES DOC_REPEAT RUNS\n", argv[0]);
        return 2;
    }
    const char *model_dir = argv[1];
    int candidates = atoi(argv[2]);
    int doc_repeat = atoi(argv[3]);
    int runs = atoi(argv[4]);
    if (candidates <= 0 || doc_repeat <= 0 || runs <= 0)
        return 2;

    char vocab_path[4096];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    tok_bpe_t *tok = tok_bpe_load(vocab_path);
    test_model_t *model = test_model_load(model_dir);
    test_workspace_t *ws = test_workspace_new(model);
    if (!tok || !model || !ws) {
        fprintf(stderr, "failed to initialize late benchmark\n");
        return 1;
    }

    int dim = test_token_dim(model);
    const ffwd_config_t *cfg = test_config(model);
    if (dim <= 0 || !cfg) {
        fprintf(stderr, "bad late model config\n");
        return 1;
    }

    late_item_t q = {0};
    int rc = 1;
    float *packed = NULL;
#ifdef CHECK_MLX
    ffwd_mlx_late_vectors_t *packed_dev = NULL;
#endif
    late_item_t *docs = (late_item_t *)calloc((size_t)candidates, sizeof(*docs));
    int *offsets = (int *)calloc((size_t)candidates + 1, sizeof(*offsets));
    float *scores = (float *)malloc((size_t)candidates * sizeof(float));
    float *device_scores = (float *)malloc((size_t)candidates * sizeof(float));
    if (!docs || !offsets || !scores || !device_scores)
        goto cleanup;

    if (encode_with_prefix_id(tok, QUERY_PREFIX_ID, "What motivates scientific discovery?", 1, &q) !=
            0 ||
        encode_late_item(model, ws, &q, dim) != 0)
        goto cleanup;

    int total_doc_tokens = 0;
    offsets[0] = 0;
    for (int i = 0; i < candidates; i++) {
        char *text = make_doc_text(i, doc_repeat);
        if (!text)
            goto cleanup;
        int enc = encode_with_prefix_id(tok, DOC_PREFIX_ID, text, 0, &docs[i]);
        free(text);
        if (enc != 0 || encode_late_item(model, ws, &docs[i], dim) != 0)
            goto cleanup;
        total_doc_tokens += docs[i].n_vecs;
        offsets[i + 1] = total_doc_tokens;
    }

    packed = (float *)malloc((size_t)total_doc_tokens * dim * sizeof(float));
    if (!packed)
        goto cleanup;
    int pos = 0;
    for (int i = 0; i < candidates; i++) {
        memcpy(packed + (size_t)pos * dim, docs[i].vecs,
               (size_t)docs[i].n_vecs * dim * sizeof(float));
        pos += docs[i].n_vecs;
    }

    if (ffwd_late_maxsim_batch(q.vecs, q.n_vecs, packed, offsets, candidates, dim, scores) != 0)
        goto cleanup;

#ifdef CHECK_MLX
    const ffwd_mlx_late_vectors_t **dev_docs =
        (const ffwd_mlx_late_vectors_t **)malloc((size_t)candidates * sizeof(*dev_docs));
    if (!dev_docs)
        goto cleanup;
    for (int i = 0; i < candidates; i++)
        dev_docs[i] = docs[i].dev;
    packed_dev = ffwd_mlx_late_vectors_concat(model, dev_docs, candidates);
    free(dev_docs);
    if (!packed_dev)
        goto cleanup;
    if (ffwd_mlx_late_maxsim_batch_device(model, q.dev, packed_dev, offsets, candidates,
                                          device_scores) != 0)
        goto cleanup;
    float worst_diff = 0.0f;
    for (int i = 0; i < candidates; i++) {
        float d = fabsf(scores[i] - device_scores[i]);
        if (d > worst_diff)
            worst_diff = d;
        if (d > 3e-4f) {
            fprintf(stderr, "device score mismatch at %d: %.9g %.9g\n", i, device_scores[i],
                    scores[i]);
            goto cleanup;
        }
    }
#endif

    double start = now_ms();
    for (int r = 0; r < runs; r++) {
        if (ffwd_late_maxsim_batch(q.vecs, q.n_vecs, packed, offsets, candidates, dim, scores) != 0)
            goto cleanup;
    }
    double cpu_ms = now_ms() - start;

    printf("late_rerank backend=%s candidates=%d query_tokens=%d "
           "doc_tokens_total=%d doc_tokens_avg=%.1f dim=%d runs=%d\n",
#ifdef CHECK_MLX
           "mlx",
#else
           "cpu",
#endif
           candidates, q.n_vecs, total_doc_tokens, (double)total_doc_tokens / (double)candidates, dim,
           runs);
    printf("cpu_packed_maxsim_ms %.6f candidates/s %.0f token_pairs/s %.0f\n", cpu_ms / (double)runs,
           ((double)runs * candidates) / (cpu_ms / 1000.0),
           ((double)runs * q.n_vecs * total_doc_tokens) / (cpu_ms / 1000.0));

#ifdef CHECK_MLX
    start = now_ms();
    for (int r = 0; r < runs; r++) {
        if (ffwd_mlx_late_maxsim_batch_device(model, q.dev, packed_dev, offsets, candidates,
                                              device_scores) != 0)
            goto cleanup;
    }
    double mlx_ms = now_ms() - start;
    printf("mlx_device_maxsim_ms %.6f candidates/s %.0f token_pairs/s %.0f "
           "max_diff %.8g\n",
           mlx_ms / (double)runs, ((double)runs * candidates) / (mlx_ms / 1000.0),
           ((double)runs * q.n_vecs * total_doc_tokens) / (mlx_ms / 1000.0), worst_diff);
#endif

    rc = 0;
cleanup:
#ifdef CHECK_MLX
    ffwd_mlx_late_vectors_free(packed_dev);
#endif
    if (docs)
        for (int i = 0; i < candidates; i++)
            free_item(&docs[i]);
    free_item(&q);
    free(packed);
    free(device_scores);
    free(scores);
    free(offsets);
    free(docs);
    test_workspace_free(ws);
    test_model_free(model);
    tok_bpe_free(tok);
    return rc;
}
