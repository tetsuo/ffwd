/* main.c - pplx_embed command-line tool */

#include "pplx_embed.h"
#include "pplx_distributed.h"
#include "pplx_server.h"
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
        "  --stdin      Read lines from stdin, write JSON embeddings to stdout\n"
        "  --serve      Serve the Perplexity-compatible HTTP API\n"
        "  --dist-worker\n"
        "               Serve one final distributed layer shard\n"
        "  --dist-remote HOST:PORT\n"
        "               Use one remote final shard for CLI or --stdin inference\n"
        "  --layers A:B Distributed half-open layer range, for example 0:14\n"
        "  --dist-activation-bits N\n"
        "               Distributed activation transport width: 32 or 16 (default: 32)\n"
        "  --model ID=DIR\n"
        "               Load model ID from directory for --serve (repeatable)\n"
        "  --backend cpu|mlx\n"
        "               Backend for --serve (default: cpu)\n"
        "  --mlx-quantize-bits N\n"
        "               Quantize MLX linear weights to 8 bits at load time\n"
        "  --mlx-quantize-group-size N\n"
        "               MLX quantization group size (default: 64)\n"
        "  --allow-memory-overcommit\n"
        "               Allow an MLX server model set above the host-memory safety budget\n"
        "  --host HOST  Server bind host (default: 127.0.0.1)\n"
        "  --port N     Server bind port (default: 8000)\n"
        "  --cors       Enable CORS headers in --serve mode\n"
        "  --api-key K  Require Authorization: Bearer K in --serve mode\n"
        "  -b <n>       Max texts per engine batch (default: all; stdin: 1; server: 8)\n"
        "  --max-batch-tokens N\n"
        "               Server max tokens per standard inference batch (default: 16384)\n"
        "  --batch-wait-us N\n"
        "               Server micro-batch wait in microseconds (default: 1000; 0 disables)\n"
        "  -t <n>       CPU threads (default: all cores)\n"
        "  -e           Print raw embeddings (with multiple texts)\n"
        "  -v           Verbose (-vv for debug)\n"
        "  -h           Show this help\n"
        "\n"
        "Modes:\n"
        "  1  text arg     Print embedding as space-separated floats\n"
        "  2+ text args    Print cosine similarity matrix\n"
        "  no args         Batch: read all stdin lines, then similarity matrix\n"
        "  --stdin         Streaming: read stdin lines, write JSON per line\n"
        "  --serve         HTTP server for /v1/embeddings and /v1/contextualizedembeddings\n"
        "  --dist-worker   Final layer-shard worker\n"
        "\n"
        "Examples:\n"
        "  %s -d ./model \"query: what is AI?\"\n"
        "  %s -d ./model --mlx --stdin < texts.txt\n"
        "  %s --serve --model pplx-embed-v1-0.6b=./model --port 8000\n"
        "  %s -d ./model --dist-worker --layers 14:28 --port 9000\n"
        "  %s -d ./model --dist-remote 127.0.0.1:9000 --layers 0:14 \"text\"\n",
        prog, prog, prog, prog, prog, prog);
}

/* ========================================================================
 * Embed one text: tokenize > forward > return float[dim]
 * ======================================================================== */

typedef struct {
    pplx_model_t     *model;
    pplx_workspace_t *workspace;
    pplx_dist_client_t *dist;
#ifdef USE_MLX
    pplx_mlx_ctx_t   *mlx_ctx;
#endif
    qwen_tokenizer_t *tok;
    qwen_tokenizer_workspace_t *tok_ws;
    int               dim;
} engine_t;

typedef struct {
    int *ids;
    int  n_tokens;
} token_buf_t;

typedef struct {
    pplx_server_model_spec_t *v;
    int n;
    int cap;
} model_specs_t;

static int engine_embed_batch(engine_t *e, const pplx_input_t *inputs,
                              int batch, float *out_embeddings)
{
    if (e->dist)
        return pplx_dist_client_embed_batch(e->dist, inputs, batch,
                                            out_embeddings);
#ifdef USE_MLX
    if (e->mlx_ctx)
        return pplx_mlx_embed_batch(e->mlx_ctx, inputs, batch, out_embeddings);
    else
#endif
        return pplx_model_embed_batch(e->model, e->workspace,
                                      inputs, batch, out_embeddings);
}

static int parse_layers(const char *arg, int *start, int *end)
{
    int used = 0;
    if (!arg || sscanf(arg, "%d:%d%n", start, end, &used) != 2 ||
        arg[used] != '\0' || *start < 0 || *start >= *end)
        return -1;
    return 0;
}

static int parse_endpoint(const char *arg, char *host, size_t hostlen, int *port)
{
    const char *colon = arg ? strrchr(arg, ':') : NULL;
    if (!colon || colon == arg || !colon[1]) return -1;
    size_t n = (size_t)(colon - arg);
    if (n >= hostlen) return -1;
    memcpy(host, arg, n);
    host[n] = '\0';
    int used = 0;
    if (sscanf(colon + 1, "%d%n", port, &used) != 1 ||
        colon[1 + used] != '\0' || *port <= 0 || *port > 65535)
        return -1;
    return 0;
}

static int tokenize_text(engine_t *e, const char *text, token_buf_t *out)
{
    memset(out, 0, sizeof(*out));
    out->ids = qwen_tokenizer_encode_with_workspace(e->tok, e->tok_ws,
                                                    text, &out->n_tokens);
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

static int append_model_spec(model_specs_t *specs, const char *arg)
{
    const char *eq = strchr(arg, '=');
    if (!eq || eq == arg || !eq[1]) {
        fprintf(stderr, "--model expects MODEL_ID=DIR\n");
        return -1;
    }
    if (specs->n == specs->cap) {
        int new_cap = specs->cap ? specs->cap * 2 : 4;
        void *p = realloc(specs->v, (size_t)new_cap * sizeof(specs->v[0]));
        if (!p) return -1;
        specs->v = p;
        specs->cap = new_cap;
    }
    size_t id_len = (size_t)(eq - arg);
    char *id = (char *)malloc(id_len + 1);
    char *path = strdup(eq + 1);
    if (id) {
        memcpy(id, arg, id_len);
        id[id_len] = '\0';
    }
    specs->v[specs->n].id = id;
    specs->v[specs->n].path = path;
    if (!specs->v[specs->n].id || !specs->v[specs->n].path)
        return -1;
    specs->n++;
    return 0;
}

static void free_model_specs(model_specs_t *specs)
{
    if (!specs) return;
    for (int i = 0; i < specs->n; i++) {
        free((char *)specs->v[i].id);
        free((char *)specs->v[i].path);
    }
    free(specs->v);
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

static size_t engine_workspace_nbytes(const engine_t *e)
{
    return (e && e->workspace) ? pplx_workspace_nbytes(e->workspace) : 0;
}

static void print_embedding_json(const float *emb, int dim, int n_tokens,
                                 double ms, size_t workspace_bytes)
{
    printf("{\"embedding\":[");
    for (int i = 0; i < dim; i++) {
        if (i > 0) putchar(',');
        printf("%.8f", (double)emb[i]);
    }
    printf("],\"dim\":%d,\"tokens\":%d,\"ms\":%.1f,\"workspace_bytes\":%zu}\n",
           dim, n_tokens, ms, workspace_bytes);
    fflush(stdout);
}

/* ========================================================================
 * Stdin mode: read lines from stdin, write JSON to stdout
 * ======================================================================== */

static int process_stdin_batch(engine_t *e, char **lines, int n_lines)
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
                             e->dim, tokens[i].n_tokens, ms,
                             engine_workspace_nbytes(e));
        next_valid++;
    }
    fflush(stdout);

    free_tokens(tokens, n_lines);
    free(tokens); free(inputs); free(input_to_line); free(embs);
    return rc == 0 ? 0 : 1;
}

static int run_stdin(engine_t *e, int batch_size)
{
    char line[65536];
    if (batch_size <= 0) batch_size = 1;

    if (pplx_verbose >= 1)
        fprintf(stderr, "stdin: ready, reading from stdin (batch_size=%d)\n",
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
            fprintf(stderr, "stdin: \"%.*s%s\"\n",
                    (int)(len > 60 ? 60 : len), line, len > 60 ? "..." : "");

        batch_lines[n_batch] = strdup(line);
        if (!batch_lines[n_batch]) {
            fprintf(stderr, "OOM\n");
            rc = 1;
            break;
        }
        n_batch++;

        if (n_batch == batch_size) {
            if (process_stdin_batch(e, batch_lines, n_batch) != 0)
                rc = 1;
            for (int i = 0; i < n_batch; i++) {
                free(batch_lines[i]);
                batch_lines[i] = NULL;
            }
            n_batch = 0;
        }
    }

    if (n_batch > 0) {
        if (process_stdin_batch(e, batch_lines, n_batch) != 0)
            rc = 1;
        for (int i = 0; i < n_batch; i++)
            free(batch_lines[i]);
    }
    free(batch_lines);

    if (pplx_verbose >= 1)
        fprintf(stderr, "stdin: EOF\n");

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
    int stdin_mode = 0;
    int serve_mode = 0;
    int dist_worker_mode = 0;
    int allow_memory_overcommit = 0;
    const char *dist_remote = NULL;
    int layer_start = -1;
    int layer_end = -1;
    int batch_size = 0;
    int max_batch_tokens = 0;
    int batch_wait_us = -1;
    int mlx_quantize_bits = 0;
    int mlx_quantize_group_size = 64;
    int dist_activation_bits = 32;
    const char *host = "127.0.0.1";
    const char *api_key = NULL;
    int port = 8000;
    int cors = 0;
    model_specs_t model_specs = {0};

    int arg_start = 1;
    while (arg_start < argc && argv[arg_start][0] == '-') {
        const char *f = argv[arg_start];
        if      (!strcmp(f, "-d"))      { model_dir = argv[++arg_start]; }
        else if (!strcmp(f, "-t"))      { n_threads = atoi(argv[++arg_start]); }
        else if (!strcmp(f, "-b") || !strcmp(f, "--batch-size")) {
            batch_size = atoi(argv[++arg_start]);
        }
        else if (!strcmp(f, "--batch-wait-us")) {
            batch_wait_us = atoi(argv[++arg_start]);
            if (batch_wait_us < 0) {
                fprintf(stderr, "--batch-wait-us must be >= 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--max-batch-tokens")) {
            max_batch_tokens = atoi(argv[++arg_start]);
            if (max_batch_tokens <= 0) {
                fprintf(stderr, "--max-batch-tokens must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--model")) {
            if (append_model_spec(&model_specs, argv[++arg_start]) != 0) {
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--backend")) {
            const char *backend = argv[++arg_start];
            if (!strcmp(backend, "mlx")) {
#ifdef USE_MLX
                use_mlx = 1;
#else
                fprintf(stderr, "mlx backend not available (build with: make mlx)\n");
                free_model_specs(&model_specs);
                return 1;
#endif
            } else if (!strcmp(backend, "cpu")) {
                use_mlx = 0;
            } else {
                fprintf(stderr, "invalid --backend: %s\n", backend);
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--mlx-quantize-bits")) {
            mlx_quantize_bits = atoi(argv[++arg_start]);
            if (mlx_quantize_bits != 0 && mlx_quantize_bits != 8) {
                fprintf(stderr, "--mlx-quantize-bits must be 0 or 8\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--mlx-quantize-group-size")) {
            mlx_quantize_group_size = atoi(argv[++arg_start]);
            if (mlx_quantize_group_size <= 0) {
                fprintf(stderr, "--mlx-quantize-group-size must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--host"))   { host = argv[++arg_start]; }
        else if (!strcmp(f, "--port"))   { port = atoi(argv[++arg_start]); }
        else if (!strcmp(f, "--allow-memory-overcommit")) {
            allow_memory_overcommit = 1;
        }
        else if (!strcmp(f, "--layers")) {
            if (parse_layers(argv[++arg_start], &layer_start, &layer_end) != 0) {
                fprintf(stderr, "--layers expects START:END with START < END\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--dist-activation-bits")) {
            dist_activation_bits = atoi(argv[++arg_start]);
            if (dist_activation_bits != 32 && dist_activation_bits != 16) {
                fprintf(stderr, "--dist-activation-bits must be 32 or 16\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--cors"))   { cors = 1; }
        else if (!strcmp(f, "--api-key")){ api_key = argv[++arg_start]; }
        else if (!strcmp(f, "-e"))      { print_embs = 1; }
        else if (!strcmp(f, "-v"))      { verbose++; }
        else if (!strcmp(f, "-vv"))     { verbose = 2; }
        else if (!strcmp(f, "--stdin")) { stdin_mode = 1; }
        else if (!strcmp(f, "--serve")) { serve_mode = 1; }
        else if (!strcmp(f, "--dist-worker")) { dist_worker_mode = 1; }
        else if (!strcmp(f, "--dist-remote")) { dist_remote = argv[++arg_start]; }
        else if (!strcmp(f, "--mlx"))   {
#ifdef USE_MLX
            use_mlx = 1;
#else
            fprintf(stderr, "--mlx not available (build with: make mlx)\n");
            free_model_specs(&model_specs);
            return 1;
#endif
        }
        else if (!strcmp(f, "-h") || !strcmp(f, "--help")) {
            print_usage(argv[0]);
            free_model_specs(&model_specs);
            return 0;
        }
        else break;
        arg_start++;
    }

    pplx_verbose = verbose;
    qwen_verbose = verbose;

    if (serve_mode) {
        if (stdin_mode || dist_worker_mode || dist_remote) {
            fprintf(stderr, "--serve cannot be combined with --stdin or distributed CLI modes\n");
            free_model_specs(&model_specs);
            return 1;
        }
        if (model_specs.n == 0) {
            fprintf(stderr, "--serve requires at least one --model MODEL_ID=DIR\n");
            print_usage(argv[0]);
            free_model_specs(&model_specs);
            return 1;
        }
        if (dist_activation_bits != 32) {
            fprintf(stderr, "--dist-activation-bits is only valid with --dist-remote\n");
            free_model_specs(&model_specs);
            return 1;
        }
        if (mlx_quantize_bits && !use_mlx) {
            fprintf(stderr, "--mlx-quantize-bits requires --backend mlx\n");
            free_model_specs(&model_specs);
            return 1;
        }
        if (!use_mlx) {
            if (n_threads <= 0) n_threads = qwen_get_num_cpus();
            qwen_set_threads(n_threads);
            if (verbose >= 1) fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
        } else if (verbose >= 1) {
            fprintf(stderr, "Using MLX GPU backend\n");
        }
        pplx_server_config_t scfg = {
            .models = model_specs.v,
            .n_models = model_specs.n,
            .host = host,
            .port = port,
            .batch_size = batch_size > 0 ? batch_size : 8,
            .max_batch_tokens = max_batch_tokens,
            .batch_wait_us = batch_wait_us,
            .use_mlx = use_mlx,
            .mlx_quantize_bits = mlx_quantize_bits,
            .mlx_quantize_group_size = mlx_quantize_group_size,
            .allow_memory_overcommit = allow_memory_overcommit,
            .enable_cors = cors,
            .api_key = api_key,
        };
        int rc = pplx_run_server(&scfg);
        free_model_specs(&model_specs);
        return rc;
    }

    if (model_specs.n != 0) {
        fprintf(stderr, "--model is only valid with --serve\n");
        free_model_specs(&model_specs);
        return 1;
    }

    if (!model_dir || !model_dir[0]) {
        fprintf(stderr, "model directory required (-d <dir>)\n");
        print_usage(argv[0]);
        free_model_specs(&model_specs);
        return 1;
    }

    if (dist_worker_mode && dist_remote) {
        fprintf(stderr, "--dist-worker and --dist-remote are mutually exclusive\n");
        return 1;
    }
    if ((dist_worker_mode || dist_remote) &&
        (layer_start < 0 || layer_end < 0)) {
        fprintf(stderr, "distributed mode requires --layers START:END\n");
        return 1;
    }
    if (dist_activation_bits != 32 && !dist_remote) {
        fprintf(stderr, "--dist-activation-bits is only valid with --dist-remote\n");
        return 1;
    }
    if (dist_remote && layer_start != 0) {
        fprintf(stderr, "--dist-remote coordinator layers must start at zero\n");
        return 1;
    }

    /* Threads (CPU backend) */
    if (!use_mlx) {
        if (n_threads <= 0) n_threads = qwen_get_num_cpus();
        qwen_set_threads(n_threads);
        if (verbose >= 1) fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
    } else {
        if (verbose >= 1) fprintf(stderr, "Using MLX GPU backend\n");
    }

    if (mlx_quantize_bits && !use_mlx) {
        fprintf(stderr, "--mlx-quantize-bits requires --mlx or --backend mlx\n");
        return 1;
    }

    pplx_dist_options_t dist_opts = {
        .mlx_quantize_bits = mlx_quantize_bits,
        .mlx_quantize_group_size = mlx_quantize_group_size,
        .activation_bits = dist_activation_bits,
    };

    if (dist_worker_mode)
        return pplx_dist_run_worker_with_options(
            model_dir, host, port, layer_start, layer_end, use_mlx,
            &dist_opts);

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
    if (dist_remote) {
        char remote_host[1024];
        int remote_port = 0;
        if (parse_endpoint(dist_remote, remote_host, sizeof(remote_host),
                           &remote_port) != 0) {
            fprintf(stderr, "--dist-remote expects HOST:PORT\n");
            qwen_tokenizer_free(tok);
            return 1;
        }
        e.dist = pplx_dist_client_open_with_options(
            model_dir, remote_host, remote_port, layer_end, use_mlx,
            &dist_opts);
        if (!e.dist) {
            fprintf(stderr, "failed to connect distributed worker\n");
            qwen_tokenizer_free(tok);
            return 1;
        }
        e.dim = pplx_dist_client_config(e.dist)->hidden_size;
    } else if (use_mlx) {
        pplx_mlx_options_t mlx_opts = {
            .quantize_bits = mlx_quantize_bits,
            .quantize_group_size = mlx_quantize_group_size,
        };
        e.mlx_ctx = pplx_mlx_load_with_options(model_dir, &mlx_opts);
        if (!e.mlx_ctx) {
            fprintf(stderr, "failed to load model\n");
            qwen_tokenizer_free(tok);
            return 1;
        }
        e.dim = pplx_mlx_config(e.mlx_ctx)->hidden_size;
    } else
#endif
    {
        if (dist_remote) {
            char remote_host[1024];
            int remote_port = 0;
            if (parse_endpoint(dist_remote, remote_host, sizeof(remote_host),
                               &remote_port) != 0) {
                fprintf(stderr, "--dist-remote expects HOST:PORT\n");
                qwen_tokenizer_free(tok);
                return 1;
            }
            e.dist = pplx_dist_client_open_with_options(
                model_dir, remote_host, remote_port, layer_end, use_mlx,
                &dist_opts);
            if (!e.dist) {
                fprintf(stderr, "failed to connect distributed worker\n");
                qwen_tokenizer_free(tok);
                return 1;
            }
            e.dim = pplx_dist_client_config(e.dist)->hidden_size;
        } else {
            e.model = pplx_model_load(model_dir);
            if (!e.model) {
                fprintf(stderr, "failed to load model\n");
                qwen_tokenizer_free(tok);
                return 1;
            }
            e.workspace = pplx_workspace_new(e.model);
            if (!e.workspace) {
                fprintf(stderr, "failed to allocate workspace\n");
                pplx_model_free(e.model);
                qwen_tokenizer_free(tok);
                return 1;
            }
            e.dim = pplx_model_config(e.model)->hidden_size;
        }
    }
    if (verbose >= 1)
        fprintf(stderr, "Model: %d-dim, %.0f ms%s\n",
                e.dim, now_ms() - t0, use_mlx ? " (MLX)" : "");

    e.tok_ws = qwen_tokenizer_workspace_new();
    if (!e.tok_ws) {
        fprintf(stderr, "failed to allocate tokenizer workspace\n");
        if (e.workspace) pplx_workspace_free(e.workspace);
        if (e.model) pplx_model_free(e.model);
        if (e.dist) pplx_dist_client_free(e.dist);
#ifdef USE_MLX
        if (e.mlx_ctx) pplx_mlx_free(e.mlx_ctx);
#endif
        qwen_tokenizer_free(tok);
        free_model_specs(&model_specs);
        return 1;
    }

    /* Run */
    int rc;
    if (stdin_mode)
        rc = run_stdin(&e, batch_size > 0 ? batch_size : 1);
    else
        rc = run_batch(&e, argc, argv, arg_start, print_embs, batch_size);

    /* Cleanup */
    if (e.workspace) pplx_workspace_free(e.workspace);
    if (e.model) pplx_model_free(e.model);
    if (e.dist) pplx_dist_client_free(e.dist);
#ifdef USE_MLX
    if (e.mlx_ctx) pplx_mlx_free(e.mlx_ctx);
#endif
    qwen_tokenizer_workspace_free(e.tok_ws);
    qwen_tokenizer_free(tok);
    free_model_specs(&model_specs);
    return rc;
}
