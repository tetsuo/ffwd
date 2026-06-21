/* Late-interaction smoke driver: encodes a query and three documents to
 * token vectors, runs scalar and batched MaxSim, and checks the expected
 * ranking. With --json it emits vectors and scores for the optional PyLate
 * comparison in check_late_interaction.py, which builds this via
 * `make late-check-driver` (or late-check-driver-mlx). */
#include "internal.h"
#ifdef CHECK_MLX
#    include "mlx.h"
#endif
#include "bpe.h"

#include <math.h>
#include <float.h>
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

static int make_skiplist(tok_bpe_t *tok, int *skip, int max_skip) {
    const char *punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    int n_skip = 0;
    for (const char *p = punct; *p && n_skip < max_skip; p++) {
        char s[2] = {*p, 0};
        int n = 0;
        int *ids = tok_bpe_encode(tok, s, &n);
        if (ids && n == 1)
            skip[n_skip++] = ids[0];
        free(ids);
    }
    return n_skip;
}

static int is_skipped(int id, const int *skip, int n_skip) {
    for (int i = 0; i < n_skip; i++) {
        if (skip[i] == id)
            return 1;
    }
    return 0;
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

static int encode_late_item(test_model_t *model,
                            test_workspace_t *ws,
                            late_item_t *item,
                            int dim,
                            const int *skip,
                            int n_skip) {
    float *all = (float *)malloc((size_t)item->n_ids * dim * sizeof(float));
    if (!all)
        return -1;
    if (test_encode(model, ws, item->ids, item->n_ids, 1, all) != 0) {
        free(all);
        return -1;
    }

    item->vecs = (float *)malloc((size_t)item->n_ids * dim * sizeof(float));
    if (!item->vecs) {
        free(all);
        return -1;
    }
    for (int i = 0; i < item->n_ids; i++) {
        if (n_skip > 0 && is_skipped(item->ids[i], skip, n_skip))
            continue;
        memcpy(item->vecs + (size_t)item->n_vecs * dim, all + (size_t)i * dim,
               (size_t)dim * sizeof(float));
        item->n_vecs++;
    }
    free(all);

#ifdef CHECK_MLX
    ffwd_mlx_late_vectors_t *all_dev =
        ffwd_mlx_late_encode_tokens_device(model, item->ids, item->n_ids, 1);
    if (!all_dev)
        return -1;
    if (n_skip > 0) {
        int *keep = (int *)malloc((size_t)item->n_vecs * sizeof(int));
        if (!keep) {
            ffwd_mlx_late_vectors_free(all_dev);
            return -1;
        }
        int n_keep = 0;
        for (int i = 0; i < item->n_ids; i++) {
            if (!is_skipped(item->ids[i], skip, n_skip))
                keep[n_keep++] = i;
        }
        item->dev = ffwd_mlx_late_vectors_select(model, all_dev, keep, n_keep);
        free(keep);
        ffwd_mlx_late_vectors_free(all_dev);
        if (!item->dev)
            return -1;
    } else {
        item->dev = all_dev;
    }
    if (ffwd_mlx_late_vectors_token_count(item->dev) != item->n_vecs ||
        ffwd_mlx_late_vectors_dim(item->dev) != dim) {
        ffwd_mlx_late_vectors_free(item->dev);
        item->dev = NULL;
        return -1;
    }
#endif

    return item->n_vecs > 0 ? 0 : -1;
}

static void free_item(late_item_t *item) {
#ifdef CHECK_MLX
    ffwd_mlx_late_vectors_free(item->dev);
#endif
    free(item->ids);
    free(item->vecs);
}

static void print_matrix_json(const char *name, const float *x, int rows, int dim) {
    printf("\"%s\":[", name);
    for (int r = 0; r < rows; r++) {
        if (r)
            putchar(',');
        putchar('[');
        for (int d = 0; d < dim; d++) {
            if (d)
                putchar(',');
            printf("%.9g", x[(size_t)r * dim + d]);
        }
        putchar(']');
    }
    putchar(']');
}

int main(int argc, char **argv) {
    if (argc < 2)
        return 2;
    const char *model_dir = argv[1];
    int json_mode = 0;
    int runs = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) {
            json_mode = 1;
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else {
            return 2;
        }
    }
    if (runs < 0)
        return 2;

    char vocab_path[4096];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);

    tok_bpe_t *tok = tok_bpe_load(vocab_path);
    test_model_t *model = test_model_load(model_dir);
    test_workspace_t *ws = test_workspace_new(model);
    if (!tok || !model || !ws) {
        fprintf(stderr, "failed to initialize late model\n");
        return 1;
    }

    int dim = test_token_dim(model);
    const ffwd_config_t *cfg = test_config(model);
    if (dim != 128 || !cfg || cfg->hidden_size != 1024) {
        fprintf(stderr, "unexpected late dimensions: hidden=%d token_dim=%d\n",
                cfg ? cfg->hidden_size : -1, dim);
        return 1;
    }

    const char *query = "What motivates scientific discovery?";
    const char *docs[] = {
        "Scientists explore the universe driven by curiosity.",
        "Children learn through curious exploration.",
        "Historical discoveries began with curious questions.",
    };
    enum { NDOCS = 3 };

    int skip[64];
    int n_skip = make_skiplist(tok, skip, 64);

    late_item_t q = {0};
    late_item_t d[NDOCS] = {{0}};
    if (encode_with_prefix_id(tok, QUERY_PREFIX_ID, query, 1, &q) != 0 ||
        encode_late_item(model, ws, &q, dim, NULL, 0) != 0)
        return 1;

    for (int i = 0; i < NDOCS; i++) {
        if (encode_with_prefix_id(tok, DOC_PREFIX_ID, docs[i], 0, &d[i]) != 0 ||
            encode_late_item(model, ws, &d[i], dim, skip, n_skip) != 0)
            return 1;
    }

    int best = -1;
    float best_score = -FLT_MAX;
    float scores[NDOCS];
    float scalar_scores[NDOCS];
    int offsets[NDOCS + 1];
    int total_doc_tokens = 0;
    offsets[0] = 0;
    for (int i = 0; i < NDOCS; i++) {
        total_doc_tokens += d[i].n_vecs;
        offsets[i + 1] = total_doc_tokens;
    }

    float *packed_docs = (float *)malloc((size_t)total_doc_tokens * dim * sizeof(float));
    if (!packed_docs)
        return 1;
    int packed_pos = 0;
    for (int i = 0; i < NDOCS; i++) {
        memcpy(packed_docs + (size_t)packed_pos * dim, d[i].vecs,
               (size_t)d[i].n_vecs * dim * sizeof(float));
        packed_pos += d[i].n_vecs;
    }

    for (int i = 0; i < NDOCS; i++) {
        scalar_scores[i] = ffwd_late_maxsim(q.vecs, q.n_vecs, d[i].vecs, d[i].n_vecs, dim);
    }
    if (ffwd_late_maxsim_batch(q.vecs, q.n_vecs, packed_docs, offsets, NDOCS, dim, scores) != 0) {
        fprintf(stderr, "late batch MaxSim failed\n");
        return 1;
    }
    for (int i = 0; i < NDOCS; i++) {
        float score = scores[i];
        if (fabsf(score - scalar_scores[i]) > 1e-5f) {
            fprintf(stderr, "late batch MaxSim mismatch at doc%d: %.9g %.9g\n", i, score,
                    scalar_scores[i]);
            return 1;
        }
        if (!json_mode) {
            printf("% .6f  doc%d  q_tokens=%d d_tokens=%d  %s\n", score, i, q.n_vecs, d[i].n_vecs,
                   docs[i]);
        }
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }

#ifdef CHECK_MLX
    const ffwd_mlx_late_vectors_t *dev_docs[NDOCS];
    for (int i = 0; i < NDOCS; i++)
        dev_docs[i] = d[i].dev;
    ffwd_mlx_late_vectors_t *packed_dev = ffwd_mlx_late_vectors_concat(model, dev_docs, NDOCS);
    if (!packed_dev) {
        fprintf(stderr, "late MLX device doc concat failed\n");
        return 1;
    }
    float device_scores[NDOCS];
    if (ffwd_mlx_late_maxsim_batch_device(model, q.dev, packed_dev, offsets, NDOCS, device_scores) !=
        0) {
        fprintf(stderr, "late MLX device MaxSim failed\n");
        ffwd_mlx_late_vectors_free(packed_dev);
        return 1;
    }
    float worst_device_diff = 0.0f;
    for (int i = 0; i < NDOCS; i++) {
        float diff = fabsf(device_scores[i] - scores[i]);
        if (diff > worst_device_diff)
            worst_device_diff = diff;
        if (diff > 2e-4f) {
            fprintf(stderr, "late MLX device MaxSim mismatch at doc%d: %.9g %.9g\n", i,
                    device_scores[i], scores[i]);
            ffwd_mlx_late_vectors_free(packed_dev);
            return 1;
        }
    }
    if (!json_mode)
        printf("mlx device maxsim: max_diff=%.8g\n", worst_device_diff);
#endif

    if (runs > 0) {
        double start = now_ms();
        for (int i = 0; i < runs; i++) {
            if (ffwd_late_maxsim_batch(q.vecs, q.n_vecs, packed_docs, offsets, NDOCS, dim, scores) !=
                0) {
                fprintf(stderr, "timed late MaxSim failed at run %d\n", i);
                return 1;
            }
        }
        double elapsed = now_ms() - start;
        double candidates = (double)runs * NDOCS;
        double token_pairs = candidates * q.n_vecs * ((double)total_doc_tokens / NDOCS);
        if (!json_mode)
            printf("benchmark: late maxsim docs=%d query_tokens=%d "
                   "doc_tokens=%d runs=%d per_run_ms=%.6f candidates/s=%.0f "
                   "token_pairs/s=%.0f\n",
                   NDOCS, q.n_vecs, total_doc_tokens, runs, elapsed / (double)runs,
                   candidates / (elapsed / 1000.0), token_pairs / (elapsed / 1000.0));

#ifdef CHECK_MLX
        start = now_ms();
        for (int i = 0; i < runs; i++) {
            if (ffwd_mlx_late_maxsim_batch_device(model, q.dev, packed_dev, offsets, NDOCS,
                                                  device_scores) != 0) {
                fprintf(stderr, "timed late MLX device MaxSim failed at run %d\n", i);
                ffwd_mlx_late_vectors_free(packed_dev);
                return 1;
            }
        }
        elapsed = now_ms() - start;
        if (!json_mode)
            printf("benchmark: late mlx_device_maxsim docs=%d query_tokens=%d "
                   "doc_tokens=%d runs=%d per_run_ms=%.6f candidates/s=%.0f "
                   "token_pairs/s=%.0f\n",
                   NDOCS, q.n_vecs, total_doc_tokens, runs, elapsed / (double)runs,
                   candidates / (elapsed / 1000.0), token_pairs / (elapsed / 1000.0));
#endif
    }

    if (json_mode) {
        printf("{\"dim\":%d,\"best\":%d,\"scores\":[", dim, best);
        for (int i = 0; i < NDOCS; i++) {
            if (i)
                putchar(',');
            printf("%.9g", scores[i]);
        }
        printf("],");
        print_matrix_json("query", q.vecs, q.n_vecs, dim);
        printf(",\"documents\":[");
        for (int i = 0; i < NDOCS; i++) {
            if (i)
                putchar(',');
            putchar('{');
            printf("\"tokens\":%d,", d[i].n_vecs);
            print_matrix_json("vectors", d[i].vecs, d[i].n_vecs, dim);
            putchar('}');
        }
        printf("]}\n");
    }

#ifdef CHECK_MLX
    ffwd_mlx_late_vectors_free(packed_dev);
#endif
    for (int i = 0; i < NDOCS; i++)
        free_item(&d[i]);
    free_item(&q);
    free(packed_docs);
    test_workspace_free(ws);
    test_model_free(model);
    tok_bpe_free(tok);

    if (best != 0) {
        fprintf(stderr, "late ranking failed: expected doc0, got doc%d\n", best);
        return 1;
    }
    if (!json_mode)
        printf("ok: late token_dim=%d best=doc%d\n", dim, best);
    return 0;
}
