/* bench/bench_tokenizer.c - tokenizer encode-path benchmark.
 * Build via `make bench-tokenizer`; needs a model dir (vocab.json). */

#include "tokenizer_bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int embed_verbose = 0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int same_ids(const int *a, int na, const int *b, int nb) {
    if (na != nb)
        return 0;
    for (int i = 0; i < na; i++) {
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int status = 1;
    int **gold = NULL;
    int *gold_n = NULL;
    int **bufs = NULL;
    int *caps = NULL;

    if (argc < 4) {
        fprintf(stderr, "usage: %s VOCAB_JSON RUNS TEXT...\n", argv[0]);
        return 2;
    }

    const char *vocab = argv[1];
    int runs = atoi(argv[2]);
    if (runs <= 0)
        return 2;
    int n_texts = argc - 3;
    char **texts = argv + 3;

    embed_tokenizer_t *tok = embed_tokenizer_load(vocab);
    embed_tokenizer_workspace_t *ws = embed_tokenizer_workspace_new();
    if (!tok || !ws) {
        fprintf(stderr, "failed to initialize tokenizer\n");
        goto cleanup;
    }

    gold = (int **)calloc((size_t)n_texts, sizeof(*gold));
    gold_n = (int *)calloc((size_t)n_texts, sizeof(*gold_n));
    bufs = (int **)calloc((size_t)n_texts, sizeof(*bufs));
    caps = (int *)calloc((size_t)n_texts, sizeof(*caps));
    if (!gold || !gold_n || !bufs || !caps)
        goto cleanup;

    int tokens_per_pass = 0;
    for (int i = 0; i < n_texts; i++) {
        gold[i] = embed_tokenizer_encode(tok, texts[i], &gold_n[i]);
        if (!gold[i] || gold_n[i] <= 0)
            goto cleanup;
        tokens_per_pass += gold_n[i];

        int n_ws = 0;
        int *ids_ws = embed_tokenizer_encode_with_workspace(tok, ws, texts[i], &n_ws);
        if (!ids_ws || !same_ids(gold[i], gold_n[i], ids_ws, n_ws)) {
            free(ids_ws);
            goto cleanup;
        }
        free(ids_ws);

        caps[i] = (int)(strlen(texts[i]) * 2 + 8);
        if (caps[i] < gold_n[i])
            caps[i] = gold_n[i];
        bufs[i] = (int *)malloc((size_t)caps[i] * sizeof(int));
        if (!bufs[i])
            goto cleanup;
        int n_into = 0;
        if (embed_tokenizer_encode_into(tok, ws, texts[i], bufs[i], caps[i], &n_into) != 0 ||
            !same_ids(gold[i], gold_n[i], bufs[i], n_into))
            goto cleanup;
    }

    double t0 = now_ms();
    long long old_tokens = 0;
    for (int r = 0; r < runs; r++) {
        for (int i = 0; i < n_texts; i++) {
            int n = 0;
            int *ids = embed_tokenizer_encode(tok, texts[i], &n);
            if (!ids || n != gold_n[i]) {
                free(ids);
                goto cleanup;
            }
            old_tokens += n;
            free(ids);
        }
    }
    double old_ms = now_ms() - t0;

    t0 = now_ms();
    long long ws_tokens = 0;
    for (int r = 0; r < runs; r++) {
        for (int i = 0; i < n_texts; i++) {
            int n = 0;
            int *ids = embed_tokenizer_encode_with_workspace(tok, ws, texts[i], &n);
            if (!ids || n != gold_n[i]) {
                free(ids);
                goto cleanup;
            }
            ws_tokens += n;
            free(ids);
        }
    }
    double ws_ms = now_ms() - t0;

    t0 = now_ms();
    long long into_tokens = 0;
    for (int r = 0; r < runs; r++) {
        for (int i = 0; i < n_texts; i++) {
            int n = 0;
            int rc = embed_tokenizer_encode_into(tok, ws, texts[i], bufs[i], caps[i], &n);
            if (rc != 0 || n != gold_n[i])
                goto cleanup;
            into_tokens += n;
        }
    }
    double into_ms = now_ms() - t0;

    printf("texts=%d runs=%d tokens/pass=%d\n", n_texts, runs, tokens_per_pass);
    printf("old_malloc:       %.3f ms  %.0f tokens/s\n", old_ms,
           (double)old_tokens / (old_ms / 1000.0));
    printf("workspace_malloc: %.3f ms  %.0f tokens/s  speedup=%.2fx\n", ws_ms,
           (double)ws_tokens / (ws_ms / 1000.0), old_ms / ws_ms);
    printf("workspace_into:   %.3f ms  %.0f tokens/s  speedup=%.2fx\n", into_ms,
           (double)into_tokens / (into_ms / 1000.0), old_ms / into_ms);

    status = 0;

cleanup:
    for (int i = 0; i < n_texts; i++) {
        if (gold)
            free(gold[i]);
        if (bufs)
            free(bufs[i]);
    }
    free(gold);
    free(gold_n);
    free(bufs);
    free(caps);
    embed_tokenizer_workspace_free(ws);
    embed_tokenizer_free(tok);
    return status;
}
