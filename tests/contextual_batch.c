/* Parity driver for contextual span embeddings: compares singleton encoding
 * (one document at a time) against the batched path, for the same documents.
 * No external library is involved; it checks the C code against itself. With a
 * positive run count it also times the batched path. check_contextual_batch_parity.py
 * builds this with `make parity-context-driver` (or parity-context-driver-mlx)
 * and passes the model dir plus tolerances as positional arguments. */
#include "internal.h"
#ifdef CHECK_MLX
#    include "mlx.h"
#endif
#include "bpe.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int *ids;
    ffwd_span_t *spans;
    int n_spans;
    int n_tokens;
} test_doc_t;

#ifdef CHECK_MLX
typedef ffwd_mlx_ctx_t test_model_t;
typedef void test_workspace_t;
static int g_mlx_quantize_bits = 0;
static int g_mlx_quantize_group_size = 64;
static test_model_t *test_model_load(const char *dir) {
    if (g_mlx_quantize_bits) {
        ffwd_mlx_options_t opts = {
            .quantize_bits = g_mlx_quantize_bits,
            .quantize_group_size = g_mlx_quantize_group_size,
        };
        return ffwd_mlx_load_with_options(dir, &opts);
    }
    return ffwd_mlx_load(dir);
}
static void test_model_free(test_model_t *m) { ffwd_mlx_free(m); }
static test_workspace_t *test_workspace_new(test_model_t *m) { return m; }
static void test_workspace_free(test_workspace_t *ws) { (void)ws; }
static const ffwd_config_t *test_config(test_model_t *m) { return ffwd_mlx_config(m); }
static int test_ffwd_spans(test_model_t *m, test_workspace_t *ws, const test_doc_t *doc, float *out) {
    (void)ws;
    return ffwd_mlx_encode_spans(m, doc->ids, doc->n_tokens, doc->spans, doc->n_spans, out);
}
static int test_ffwd_spans_batch(test_model_t *m,
                                 test_workspace_t *ws,
                                 const ffwd_context_input_t *inputs,
                                 int batch,
                                 float *out) {
    (void)ws;
    return ffwd_mlx_encode_spans_batch(m, inputs, batch, out);
}
#else
typedef ffwd_model_t test_model_t;
typedef ffwd_workspace_t test_workspace_t;
static test_model_t *test_model_load(const char *dir) { return ffwd_model_load(dir); }
static void test_model_free(test_model_t *m) { ffwd_model_free(m); }
static test_workspace_t *test_workspace_new(test_model_t *m) { return ffwd_workspace_new(m); }
static void test_workspace_free(test_workspace_t *ws) { ffwd_workspace_free(ws); }
static const ffwd_config_t *test_config(test_model_t *m) { return ffwd_model_config(m); }
static int test_ffwd_spans(test_model_t *m, test_workspace_t *ws, const test_doc_t *doc, float *out) {
    return ffwd_model_encode_spans(m, ws, doc->ids, doc->n_tokens, doc->spans, doc->n_spans, out);
}
static int test_ffwd_spans_batch(test_model_t *m,
                                 test_workspace_t *ws,
                                 const ffwd_context_input_t *inputs,
                                 int batch,
                                 float *out) {
    return ffwd_model_encode_spans_batch(m, ws, inputs, batch, out);
}
#endif

static int
init_doc(test_doc_t *doc, tok_bpe_t *tok, const char **chunks, int n_chunks, int separator_id) {
    int **ids = calloc((size_t)n_chunks, sizeof(*ids));
    int *lengths = calloc((size_t)n_chunks, sizeof(*lengths));
    doc->spans = calloc((size_t)n_chunks, sizeof(*doc->spans));
    if (!ids || !lengths || !doc->spans)
        return -1;
    doc->n_spans = n_chunks;
    doc->n_tokens = n_chunks > 1 ? n_chunks - 1 : 0;
    for (int i = 0; i < n_chunks; i++) {
        ids[i] = tok_bpe_encode(tok, chunks[i], &lengths[i]);
        if (!ids[i] || lengths[i] <= 0)
            return -1;
        doc->n_tokens += lengths[i];
    }
    doc->ids = malloc((size_t)doc->n_tokens * sizeof(*doc->ids));
    if (!doc->ids)
        return -1;
    int pos = 0;
    for (int i = 0; i < n_chunks; i++) {
        doc->spans[i].start = pos;
        doc->spans[i].n_tokens = lengths[i];
        memcpy(doc->ids + pos, ids[i], (size_t)lengths[i] * sizeof(*doc->ids));
        pos += lengths[i];
        if (i + 1 < n_chunks) {
            doc->ids[pos++] = separator_id;
        }
        free(ids[i]);
    }
    free(lengths);
    free(ids);
    return 0;
}

static void free_doc(test_doc_t *doc) {
    free(doc->ids);
    free(doc->spans);
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > worst)
            worst = diff;
    }
    return worst;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int cmp_double(const void *a, const void *b) {
    double aa = *(const double *)a;
    double bb = *(const double *)b;
    return (aa > bb) - (aa < bb);
}

static char *make_synthetic_chunk(int doc, int chunk, int repeats) {
    const char *base = "Contextual embedding inference processes complete documents before "
                       "pooling chunk spans. This deterministic text stresses long "
                       "bidirectional attention and compact span result handling. ";
    if (repeats < 1)
        repeats = 1;
    size_t base_len = strlen(base);
    size_t cap = 128 + (size_t)repeats * base_len;
    char *s = malloc(cap);
    if (!s)
        return NULL;
    int n = snprintf(s, cap, "document %d chunk %d: ", doc, chunk);
    if (n < 0 || (size_t)n >= cap) {
        free(s);
        return NULL;
    }
    size_t off = (size_t)n;
    for (int i = 0; i < repeats; i++) {
        if (off + base_len + 1 > cap) {
            free(s);
            return NULL;
        }
        memcpy(s + off, base, base_len);
        off += base_len;
    }
    s[off] = '\0';
    return s;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 11)
        return 2;
    int runs = argc >= 3 ? atoi(argv[2]) : 0;
    float max_allowed_diff = argc >= 4 ? strtof(argv[3], NULL) : 0.00005f;
    float min_allowed_cosine = argc >= 5 ? strtof(argv[4], NULL) : 0.99999f;
#ifdef CHECK_MLX
    g_mlx_quantize_bits = argc >= 6 ? atoi(argv[5]) : 0;
    g_mlx_quantize_group_size = argc >= 7 ? atoi(argv[6]) : 64;
    if (g_mlx_quantize_bits != 0 && g_mlx_quantize_bits != 8)
        return 2;
    if (g_mlx_quantize_group_size <= 0)
        return 2;
#endif
    int synthetic_docs = argc >= 8 ? atoi(argv[7]) : 0;
    int synthetic_chunks = argc >= 9 ? atoi(argv[8]) : 0;
    int synthetic_repeats = argc >= 10 ? atoi(argv[9]) : 1;
    int synthetic_ragged = argc >= 11 ? atoi(argv[10]) : 0;
    if (runs < 0)
        return 2;
    if (synthetic_docs < 0 || synthetic_chunks < 0 || synthetic_repeats < 0)
        return 2;
    char vocab_path[4096];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", argv[1]);

    test_model_t *model = test_model_load(argv[1]);
    test_workspace_t *ws = test_workspace_new(model);
    tok_bpe_t *tok = tok_bpe_load(vocab_path);
    if (!model || !ws || !tok) {
        fprintf(stderr, "failed to initialize model or tokenizer\n");
        return 1;
    }
    int separator_id = FFWD_CONTEXT_SEPARATOR_TOKEN_ID;

    int batch = synthetic_docs > 0 ? synthetic_docs : 3;
    int total_spans = 0;
    test_doc_t *docs = calloc((size_t)batch, sizeof(*docs));
    ffwd_context_input_t *inputs = calloc((size_t)batch, sizeof(*inputs));
    if (!docs || !inputs)
        return 1;

    if (synthetic_docs > 0) {
        if (synthetic_chunks <= 0)
            return 2;
        for (int i = 0; i < batch; i++) {
            int doc_chunks = synthetic_ragged ? synthetic_chunks * (i + 1) : synthetic_chunks;
            char **texts = calloc((size_t)doc_chunks, sizeof(*texts));
            if (!texts)
                return 1;
            int ok = 1;
            for (int c = 0; c < doc_chunks; c++) {
                texts[c] = make_synthetic_chunk(i, c, synthetic_repeats);
                if (!texts[c])
                    ok = 0;
            }
            if (!ok || init_doc(&docs[i], tok, (const char **)texts, doc_chunks, separator_id) != 0) {
                fprintf(stderr, "failed to prepare synthetic document %d\n", i);
                for (int c = 0; c < doc_chunks; c++)
                    free(texts[c]);
                free(texts);
                return 1;
            }
            for (int c = 0; c < doc_chunks; c++)
                free(texts[c]);
            free(texts);
            total_spans += docs[i].n_spans;
        }
    } else {
        const char *doc0[] = {"Python is useful for scripting.",
                              "Paris is the capital city of France."};
        const char *doc1[] = {"Redis stores in-memory data structures.",
                              "SQLite stores relational data in a local file.",
                              "Neural networks produce useful representations for retrieval."};
        const char *doc2[] = {"A deliberately short final document."};
        const char **chunks[] = {doc0, doc1, doc2};
        int chunk_counts[] = {2, 3, 1};
        for (int i = 0; i < batch; i++) {
            if (init_doc(&docs[i], tok, chunks[i], chunk_counts[i], separator_id) != 0) {
                fprintf(stderr, "failed to prepare document %d\n", i);
                return 1;
            }
            total_spans += docs[i].n_spans;
        }
    }

    for (int i = 0; i < batch; i++) {
        inputs[i].input.ids = docs[i].ids;
        inputs[i].input.n_tokens = docs[i].n_tokens;
        inputs[i].spans = docs[i].spans;
        inputs[i].n_spans = docs[i].n_spans;
    }

    int hidden = test_config(model)->hidden_size;
    float *expected = calloc((size_t)total_spans * hidden, sizeof(*expected));
    float *actual = calloc((size_t)total_spans * hidden, sizeof(*actual));
    if (!expected || !actual)
        return 1;

    int span_offset = 0;
    for (int i = 0; i < batch; i++) {
        if (test_ffwd_spans(model, ws, &docs[i], expected + (size_t)span_offset * hidden) != 0) {
            fprintf(stderr, "singleton contextual embedding failed at %d\n", i);
            return 1;
        }
        span_offset += docs[i].n_spans;
    }
    if (test_ffwd_spans_batch(model, ws, inputs, batch, actual) != 0) {
        fprintf(stderr, "batched contextual embedding failed\n");
        return 1;
    }

    float worst_diff = 0.0f;
    float worst_cosine = 1.0f;
    for (int i = 0; i < total_spans; i++) {
        float *want = expected + (size_t)i * hidden;
        float *got = actual + (size_t)i * hidden;
        float diff = max_abs_diff(want, got, hidden);
        float cosine = ffwd_cosine_similarity(want, got, hidden);
        if (diff > worst_diff)
            worst_diff = diff;
        if (cosine < worst_cosine)
            worst_cosine = cosine;
    }

    if (runs > 0) {
        double *times = calloc((size_t)runs, sizeof(*times));
        if (!times)
            return 1;
        for (int i = 0; i < runs; i++) {
            double start = now_ms();
            if (test_ffwd_spans_batch(model, ws, inputs, batch, actual) != 0) {
                fprintf(stderr, "timed contextual batch failed at %d\n", i);
                return 1;
            }
            times[i] = now_ms() - start;
        }
        qsort(times, (size_t)runs, sizeof(*times), cmp_double);
        printf("benchmark: contextual batch docs=%d spans=%d runs=%d p50_ms=%.3f\n", batch,
               total_spans, runs, times[runs / 2]);
        free(times);
    }

    free(actual);
    free(expected);
    for (int i = 0; i < batch; i++)
        free_doc(&docs[i]);
    free(inputs);
    free(docs);
    tok_bpe_free(tok);
    test_workspace_free(ws);
    test_model_free(model);

    if (worst_diff > max_allowed_diff || worst_cosine < min_allowed_cosine) {
        fprintf(stderr, "contextual batch parity failed: max_abs_diff=%g cosine=%g\n", worst_diff,
                worst_cosine);
        return 1;
    }
    printf("ok: contextual batch parity docs=%d spans=%d max_abs_diff=%g cosine=%g\n", batch,
           total_spans, worst_diff, worst_cosine);
    return 0;
}
