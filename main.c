/* main.c - pplx_embed command-line tool */

#include "pplx_embed.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_tokenizer.h"

#ifdef USE_MLX
#include "pplx_embed_mlx.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -d <model_dir> [options] [text...]\n"
        "\n"
        "Options:\n"
        "  -d <dir>     Model directory (required)\n"
        "  --mlx        Use Apple MLX GPU backend\n"
        "  --daemon     Read lines from stdin, write JSON embeddings to stdout\n"
        "  -b <n>       Max texts per engine batch (default: all; daemon: 1)\n"
        "  -t <n>       CPU threads (default: all cores)\n"
        "  -e           Print raw embeddings (with multiple texts)\n"
        "  -v           Verbose (-vv for debug)\n"
        "  -h           Show this help\n"
        "\n"
        "Modes:\n"
        "  1  text arg     Print embedding as space-separated floats\n"
        "  2+ text args    Print cosine similarity matrix\n"
        "  no args         Batch: read all stdin lines, then similarity matrix\n"
        "  --daemon        Streaming: read stdin lines, write JSON per line\n"
        "\n"
        "Examples:\n"
        "  %s -d ./model \"query: what is AI?\"\n"
        "  %s -d ./model --mlx --daemon < texts.txt\n",
        prog, prog, prog);
}

/* ========================================================================
 * Embed one text: tokenize > forward > return float[dim]
 * ======================================================================== */

typedef struct {
    pplx_ctx_t       *ctx;
#ifdef USE_MLX
    pplx_mlx_ctx_t   *mlx_ctx;
#endif
    qwen_tokenizer_t *tok;
    int               dim;
} engine_t;

typedef struct {
    int *ids;
    int  n_tokens;
} token_buf_t;

static int engine_embed_batch(engine_t *e, const pplx_input_t *inputs,
                              int batch, float *out_embeddings)
{
#ifdef USE_MLX
    if (e->mlx_ctx)
        return pplx_mlx_embed_batch(e->mlx_ctx, inputs, batch, out_embeddings);
    else
#endif
        return pplx_embed_batch(e->ctx, inputs, batch, out_embeddings);
}

static int tokenize_text(engine_t *e, const char *text, token_buf_t *out)
{
    memset(out, 0, sizeof(*out));
    out->ids = qwen_tokenizer_encode(e->tok, text, &out->n_tokens);
    if (!out->ids || out->n_tokens == 0) {
        fprintf(stderr, "tokenization failed: %s\n", text);
        free(out->ids);
        out->ids = NULL;
        out->n_tokens = 0;
        return -1;
    }

    if (pplx_verbose >= 1) {
        fprintf(stderr, "tokens (%d): ", out->n_tokens);
        for (int i = 0; i < out->n_tokens && i < 20; i++)
            fprintf(stderr, "%d ", out->ids[i]);
        if (out->n_tokens > 20) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }
    return 0;
}

static void free_tokens(token_buf_t *tokens, int n)
{
    if (!tokens) return;
    for (int i = 0; i < n; i++)
        free(tokens[i].ids);
}

/* ========================================================================
 * Output helpers
 * ======================================================================== */

static void print_embedding_raw(const float *emb, int dim)
{
    for (int i = 0; i < dim; i++) {
        if (i > 0) putchar(' ');
        printf("%.8f", (double)emb[i]);
    }
    putchar('\n');
}

static void print_embedding_json(const float *emb, int dim, int n_tokens, double ms)
{
    printf("{\"embedding\":[");
    for (int i = 0; i < dim; i++) {
        if (i > 0) putchar(',');
        printf("%.8f", (double)emb[i]);
    }
    printf("],\"dim\":%d,\"tokens\":%d,\"ms\":%.1f}\n", dim, n_tokens, ms);
    fflush(stdout);
}

/* ========================================================================
 * Daemon mode: read lines from stdin, write JSON to stdout
 * ======================================================================== */

static int process_daemon_batch(engine_t *e, char **lines, int n_lines)
{
    if (n_lines <= 0) return 0;

    token_buf_t *tokens = (token_buf_t *)calloc((size_t)n_lines, sizeof(token_buf_t));
    pplx_input_t *inputs = (pplx_input_t *)malloc((size_t)n_lines * sizeof(pplx_input_t));
    int *input_to_line = (int *)malloc((size_t)n_lines * sizeof(int));
    float *embs = (float *)malloc((size_t)n_lines * e->dim * sizeof(float));
    if (!tokens || !inputs || !input_to_line || !embs) {
        fprintf(stderr, "OOM\n");
        free(tokens); free(inputs); free(input_to_line); free(embs);
        for (int i = 0; i < n_lines; i++)
            printf("{\"error\":\"embedding failed\"}\n");
        fflush(stdout);
        return 1;
    }

    int n_inputs = 0;
    for (int i = 0; i < n_lines; i++) {
        if (tokenize_text(e, lines[i], &tokens[i]) == 0) {
            inputs[n_inputs].ids = tokens[i].ids;
            inputs[n_inputs].n_tokens = tokens[i].n_tokens;
            input_to_line[n_inputs] = i;
            n_inputs++;
        }
    }

    double ms = 0.0;
    int rc = 0;
    if (n_inputs > 0) {
        double t0 = now_ms();
        rc = engine_embed_batch(e, inputs, n_inputs, embs);
        ms = now_ms() - t0;
        if (pplx_verbose >= 1)
            fprintf(stderr, "embed batch: %d texts in %.1f ms\n", n_inputs, ms);
    }

    int next_valid = 0;
    for (int i = 0; i < n_lines; i++) {
        if (!tokens[i].ids) {
            printf("{\"error\":\"tokenization failed\"}\n");
            continue;
        }
        if (rc != 0) {
            printf("{\"error\":\"embedding failed\"}\n");
            continue;
        }
        while (next_valid < n_inputs && input_to_line[next_valid] != i)
            next_valid++;
        if (next_valid >= n_inputs) {
            printf("{\"error\":\"embedding failed\"}\n");
            continue;
        }
        print_embedding_json(embs + (size_t)next_valid * e->dim,
                             e->dim, tokens[i].n_tokens, ms);
        next_valid++;
    }
    fflush(stdout);

    free_tokens(tokens, n_lines);
    free(tokens); free(inputs); free(input_to_line); free(embs);
    return rc == 0 ? 0 : 1;
}

static int run_daemon(engine_t *e, int batch_size)
{
    char line[65536];
    if (batch_size <= 0) batch_size = 1;

    if (pplx_verbose >= 1)
        fprintf(stderr, "daemon: ready, reading from stdin (batch_size=%d)\n",
                batch_size);

    char **batch_lines = (char **)calloc((size_t)batch_size, sizeof(char *));
    if (!batch_lines) {
        fprintf(stderr, "OOM\n");
        return 1;
    }
    int n_batch = 0;
    int rc = 0;

    while (fgets(line, sizeof(line), stdin)) {
        /* Strip newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        if (pplx_verbose >= 1)
            fprintf(stderr, "daemon: \"%.*s%s\"\n",
                    (int)(len > 60 ? 60 : len), line, len > 60 ? "..." : "");

        batch_lines[n_batch] = strdup(line);
        if (!batch_lines[n_batch]) {
            fprintf(stderr, "OOM\n");
            rc = 1;
            break;
        }
        n_batch++;

        if (n_batch == batch_size) {
            if (process_daemon_batch(e, batch_lines, n_batch) != 0)
                rc = 1;
            for (int i = 0; i < n_batch; i++) {
                free(batch_lines[i]);
                batch_lines[i] = NULL;
            }
            n_batch = 0;
        }
    }

    if (n_batch > 0) {
        if (process_daemon_batch(e, batch_lines, n_batch) != 0)
            rc = 1;
        for (int i = 0; i < n_batch; i++)
            free(batch_lines[i]);
    }
    free(batch_lines);

    if (pplx_verbose >= 1)
        fprintf(stderr, "daemon: stdin EOF\n");

    return rc;
}

/* ========================================================================
 * Batch mode: embed args or stdin lines, then print similarity or vectors
 * ======================================================================== */

static int append_text(char ***texts, int *n_texts, int *cap, const char *s)
{
    if (*n_texts == *cap) {
        int new_cap = *cap ? *cap * 2 : 8;
        char **new_texts = (char **)realloc(*texts, (size_t)new_cap * sizeof(char *));
        if (!new_texts) return -1;
        *texts = new_texts;
        *cap = new_cap;
    }

    (*texts)[*n_texts] = strdup(s);
    if (!(*texts)[*n_texts]) return -1;
    (*n_texts)++;
    return 0;
}

static int run_batch(engine_t *e, int argc, char **argv, int arg_start,
                     int print_embs, int batch_size)
{
    int     n_texts = 0, cap = 0;
    char  **texts = NULL;
    int     rc = 0;

    if (arg_start < argc) {
        for (int i = arg_start; i < argc; i++) {
            if (append_text(&texts, &n_texts, &cap, argv[i]) != 0)
                goto oom_texts;
        }
    } else {
        char line[65536];
        while (fgets(line, sizeof(line), stdin)) {
            size_t l = strlen(line);
            while (l > 0 && (line[l-1] == '\n' || line[l-1] == '\r'))
                line[--l] = '\0';
            if (l == 0) continue;
            if (append_text(&texts, &n_texts, &cap, line) != 0)
                goto oom_texts;
        }
    }

    if (n_texts == 0) {
        fprintf(stderr, "no input texts\n");
        free(texts);
        return 1;
    }

    int dim = e->dim;
    token_buf_t *tokens = (token_buf_t *)calloc((size_t)n_texts, sizeof(token_buf_t));
    pplx_input_t *inputs = (pplx_input_t *)malloc((size_t)n_texts * sizeof(pplx_input_t));
    float *embs = (float *)malloc((size_t)n_texts * dim * sizeof(float));
    if (!tokens || !inputs || !embs) {
        fprintf(stderr, "OOM\n");
        free(tokens); free(inputs); free(embs);
        for (int i = 0; i < n_texts; i++) free(texts[i]);
        free(texts);
        return 1;
    }

    for (int i = 0; i < n_texts; i++) {
        if (pplx_verbose >= 1)
            fprintf(stderr, "[%d/%d] \"%s\"\n", i+1, n_texts, texts[i]);
        if (tokenize_text(e, texts[i], &tokens[i]) != 0) {
            rc = 1;
            goto done;
        }
        inputs[i].ids = tokens[i].ids;
        inputs[i].n_tokens = tokens[i].n_tokens;
    }

    int max_batch = batch_size > 0 ? batch_size : n_texts;
    for (int start = 0; start < n_texts; start += max_batch) {
        int cur = n_texts - start;
        if (cur > max_batch) cur = max_batch;

        double t0 = now_ms();
        if (engine_embed_batch(e, inputs + start, cur,
                               embs + (size_t)start * dim) != 0) {
            fprintf(stderr, "forward pass failed\n");
            rc = 1;
            goto done;
        }
        if (pplx_verbose >= 1)
            fprintf(stderr, "embed batch: [%d..%d] %d texts in %.1f ms\n",
                    start, start + cur - 1, cur, now_ms() - t0);
    }

    if (n_texts == 1) {
        print_embedding_raw(embs, dim);
    } else {
        printf("Cosine similarity matrix (%d texts):\n", n_texts);
        printf("%-4s", "");
        for (int j = 0; j < n_texts; j++) printf("  [%d]  ", j);
        printf("\n");
        for (int i = 0; i < n_texts; i++) {
            printf("[%d] ", i);
            for (int j = 0; j < n_texts; j++)
                printf("  %.4f", (double)pplx_cosine_similarity(
                    embs + (size_t)i * dim, embs + (size_t)j * dim, dim));
            printf("  \"%s\"\n", texts[i]);
        }
        if (print_embs) {
            printf("\nEmbeddings:\n");
            for (int i = 0; i < n_texts; i++) {
                printf("[%d] ", i);
                print_embedding_raw(embs + (size_t)i * dim, dim);
            }
        }
    }

done:
    free_tokens(tokens, n_texts);
    for (int i = 0; i < n_texts; i++) free(texts[i]);
    free(texts); free(tokens); free(inputs); free(embs);
    return rc;

oom_texts:
    fprintf(stderr, "OOM\n");
    for (int i = 0; i < n_texts; i++) free(texts[i]);
    free(texts);
    return 1;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[])
{
    const char *model_dir = NULL;
    int n_threads  = 0;
    int print_embs = 0;
    int verbose    = 0;
    int use_mlx    = 0;
    int daemon     = 0;
    int batch_size = 0;

    int arg_start = 1;
    while (arg_start < argc && argv[arg_start][0] == '-') {
        const char *f = argv[arg_start];
        if      (!strcmp(f, "-d"))      { model_dir = argv[++arg_start]; }
        else if (!strcmp(f, "-t"))      { n_threads = atoi(argv[++arg_start]); }
        else if (!strcmp(f, "-b") || !strcmp(f, "--batch-size")) {
            batch_size = atoi(argv[++arg_start]);
        }
        else if (!strcmp(f, "-e"))      { print_embs = 1; }
        else if (!strcmp(f, "-v"))      { verbose++; }
        else if (!strcmp(f, "-vv"))     { verbose = 2; }
        else if (!strcmp(f, "--daemon")){ daemon = 1; }
        else if (!strcmp(f, "--mlx"))   {
#ifdef USE_MLX
            use_mlx = 1;
#else
            fprintf(stderr, "--mlx not available (build with: make mlx)\n"); return 1;
#endif
        }
        else if (!strcmp(f, "-h") || !strcmp(f, "--help")) { print_usage(argv[0]); return 0; }
        else break;
        arg_start++;
    }

    if (!model_dir || !model_dir[0]) {
        fprintf(stderr, "model directory required (-d <dir>)\n");
        print_usage(argv[0]);
        return 1;
    }

    pplx_verbose = verbose;
    qwen_verbose = verbose;

    /* Threads (CPU backend) */
    if (!use_mlx) {
        if (n_threads <= 0) n_threads = qwen_get_num_cpus();
        qwen_set_threads(n_threads);
        if (verbose >= 1) fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
    } else {
        if (verbose >= 1) fprintf(stderr, "Using MLX GPU backend\n");
    }

    /* Tokenizer */
    char vocab_path[1024];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", model_dir);
    double t0 = now_ms();
    qwen_tokenizer_t *tok = qwen_tokenizer_load(vocab_path);
    if (!tok) { fprintf(stderr, "failed to load tokenizer: %s\n", vocab_path); return 1; }
    if (verbose >= 1) fprintf(stderr, "Tokenizer: %.0f ms\n", now_ms() - t0);

    /* Model */
    engine_t e = {0};
    e.tok = tok;
    t0 = now_ms();

#ifdef USE_MLX
    if (use_mlx) {
        e.mlx_ctx = pplx_mlx_load(model_dir);
        if (!e.mlx_ctx) { fprintf(stderr, "failed to load model\n"); return 1; }
        e.dim = pplx_mlx_config(e.mlx_ctx)->hidden_size;
    } else
#endif
    {
        e.ctx = pplx_load(model_dir);
        if (!e.ctx) { fprintf(stderr, "failed to load model\n"); return 1; }
        e.dim = e.ctx->config.hidden_size;
    }
    if (verbose >= 1)
        fprintf(stderr, "Model: %d-dim, %.0f ms%s\n",
                e.dim, now_ms() - t0, use_mlx ? " (MLX)" : "");

    /* Run */
    int rc;
    if (daemon)
        rc = run_daemon(&e, batch_size > 0 ? batch_size : 1);
    else
        rc = run_batch(&e, argc, argv, arg_start, print_embs, batch_size);

    /* Cleanup */
    if (e.ctx) pplx_free(e.ctx);
#ifdef USE_MLX
    if (e.mlx_ctx) pplx_mlx_free(e.mlx_ctx);
#endif
    qwen_tokenizer_free(tok);
    return rc;
}
