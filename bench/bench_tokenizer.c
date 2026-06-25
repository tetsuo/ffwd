/* bench/bench_tokenizer.c - tokenizer encode-path benchmark.
 * Build via `make bench-tokenizer`; accepts a model dir or vocab.json. */

#include "bpe.h"
#include "sentencepiece.h"
#include "wordpiece.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

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

typedef enum {
    TOK_BPE,
    TOK_WORDPIECE,
    TOK_SENTENCEPIECE,
} tok_kind_t;

typedef struct {
    tok_kind_t kind;
    void *tok;
    void *ws;
} bench_tok_t;

static int path_exists(const char *path) { return access(path, R_OK) == 0; }

static int load_tokenizer(bench_tok_t *bt, const char *path) {
    char p[1024];
    snprintf(p, sizeof(p), "%s/vocab.txt", path);
    if (path_exists(p)) {
        bt->kind = TOK_WORDPIECE;
        bt->tok = tok_wp_load(path);
        bt->ws = tok_wp_workspace_new();
        return bt->tok && bt->ws ? 0 : -1;
    }
    snprintf(p, sizeof(p), "%s/sentencepiece.bpe.model", path);
    if (!path_exists(p)) {
        snprintf(p, sizeof(p), "%s/tokenizer.model", path);
        if (!path_exists(p))
            snprintf(p, sizeof(p), "%s/spiece.model", path);
    }
    if (path_exists(p)) {
        bt->kind = TOK_SENTENCEPIECE;
        bt->tok = tok_spm_load(path);
        bt->ws = tok_spm_workspace_new();
        return bt->tok && bt->ws ? 0 : -1;
    }
    bt->kind = TOK_BPE;
    const char *vocab = path;
    snprintf(p, sizeof(p), "%s/vocab.json", path);
    if (path_exists(p))
        vocab = p;
    bt->tok = tok_bpe_load(vocab);
    bt->ws = tok_bpe_workspace_new();
    return bt->tok && bt->ws ? 0 : -1;
}

static int *tok_encode(bench_tok_t *bt, const char *text, int *n) {
    switch (bt->kind) {
    case TOK_WORDPIECE:
        return tok_wp_encode((tok_wp_t *)bt->tok, text, n);
    case TOK_SENTENCEPIECE:
        return tok_spm_encode((tok_spm_t *)bt->tok, text, n);
    default:
        return tok_bpe_encode((tok_bpe_t *)bt->tok, text, n);
    }
}

static int *tok_encode_ws(bench_tok_t *bt, const char *text, int *n) {
    switch (bt->kind) {
    case TOK_WORDPIECE:
        return tok_wp_encode_with_workspace((tok_wp_t *)bt->tok, (tok_wp_workspace_t *)bt->ws, text,
                                            n);
    case TOK_SENTENCEPIECE:
        return tok_spm_encode_with_workspace((tok_spm_t *)bt->tok, (tok_spm_workspace_t *)bt->ws,
                                             text, n);
    default:
        return tok_bpe_encode_with_workspace((tok_bpe_t *)bt->tok, (tok_bpe_workspace_t *)bt->ws,
                                             text, n);
    }
}

static int tok_encode_into(bench_tok_t *bt, const char *text, int *out, int cap, int *n) {
    switch (bt->kind) {
    case TOK_WORDPIECE:
        return tok_wp_encode_into((tok_wp_t *)bt->tok, (tok_wp_workspace_t *)bt->ws, text, out, cap,
                                  n);
    case TOK_SENTENCEPIECE:
        return tok_spm_encode_into((tok_spm_t *)bt->tok, (tok_spm_workspace_t *)bt->ws, text, out,
                                   cap, n);
    default:
        return tok_bpe_encode_into((tok_bpe_t *)bt->tok, (tok_bpe_workspace_t *)bt->ws, text, out,
                                   cap, n);
    }
}

static void free_tokenizer(bench_tok_t *bt) {
    switch (bt->kind) {
    case TOK_WORDPIECE:
        tok_wp_workspace_free((tok_wp_workspace_t *)bt->ws);
        tok_wp_free((tok_wp_t *)bt->tok);
        break;
    case TOK_SENTENCEPIECE:
        tok_spm_workspace_free((tok_spm_workspace_t *)bt->ws);
        tok_spm_free((tok_spm_t *)bt->tok);
        break;
    default:
        tok_bpe_workspace_free((tok_bpe_workspace_t *)bt->ws);
        tok_bpe_free((tok_bpe_t *)bt->tok);
        break;
    }
}

int main(int argc, char **argv) {
    int status = 1;
    int **gold = NULL;
    int *gold_n = NULL;
    int **bufs = NULL;
    int *caps = NULL;

    bench_tok_t bt = {0};

    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL_DIR_OR_VOCAB_JSON RUNS TEXT...\n", argv[0]);
        return 2;
    }

    const char *vocab = argv[1];
    int runs = atoi(argv[2]);
    if (runs <= 0)
        return 2;
    int n_texts = argc - 3;
    char **texts = argv + 3;

    if (load_tokenizer(&bt, vocab) != 0) {
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
        gold[i] = tok_encode(&bt, texts[i], &gold_n[i]);
        if (gold_n[i] < 0 || (!gold[i] && gold_n[i] != 0))
            goto cleanup;
        tokens_per_pass += gold_n[i];

        int n_ws = 0;
        int *ids_ws = tok_encode_ws(&bt, texts[i], &n_ws);
        if ((!ids_ws && n_ws != 0) || !same_ids(gold[i], gold_n[i], ids_ws, n_ws)) {
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
        if (tok_encode_into(&bt, texts[i], bufs[i], caps[i], &n_into) != 0 ||
            !same_ids(gold[i], gold_n[i], bufs[i], n_into))
            goto cleanup;
    }

    double t0 = now_ms();
    long long old_tokens = 0;
    for (int r = 0; r < runs; r++) {
        for (int i = 0; i < n_texts; i++) {
            int n = 0;
            int *ids = tok_encode(&bt, texts[i], &n);
            if ((!ids && n != 0) || n != gold_n[i]) {
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
            int *ids = tok_encode_ws(&bt, texts[i], &n);
            if ((!ids && n != 0) || n != gold_n[i]) {
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
            int rc = tok_encode_into(&bt, texts[i], bufs[i], caps[i], &n);
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
    free_tokenizer(&bt);
    return status;
}
