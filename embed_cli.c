/* embed_cli.c - embed command-line tool */

#include "embed.h"
#include "qwen_kernels.h"
#include "qwen_tokenizer.h"

#ifdef USE_MLX
#include "embed_mlx.h"
#endif
#ifdef USE_CUDA
#include "embed_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

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
#ifdef USE_MLX
        "  --mlx        Use Apple MLX GPU backend\n"
#endif
#ifdef USE_CUDA
        "  --cuda       Use CUDA/cuBLAS GPU backend\n"
#endif
        "  --stream     Read lines from stdin, write one JSON embedding per line\n"
#ifdef USE_MLX
        "  --mlx-quant-bits N\n"
        "               Quantize MLX linear weights to 8 bits at load time\n"
        "  --mlx-quant-group-size N\n"
        "               MLX quantization group size (default: 64)\n"
#endif
#ifdef USE_CUDA
        "  --cuda-gemm-mode MODE\n"
        "               CUDA GEMM compute mode: f32, tf32, bf16, or 16f (default: f32)\n"
        "  --cuda-weight-dtype DTYPE\n"
        "               CUDA weight storage: f32 or bf16 (default: f32). bf16 halves\n"
        "               weight memory and uses BF16 tensor cores\n"
#endif
        "  -b, --batch-size N\n"
        "               Max texts per engine batch (default: all; --stream: 1)\n"
        "  -t, --threads N\n"
        "               CPU threads (default: all cores)\n"
        "  -e, --embeddings\n"
        "               Print raw embeddings (with multiple texts)\n"
        "  -v, --verbose\n"
        "               Verbose (-vv for debug)\n"
        "  -h           Show this help\n"
        "\n"
        "Modes:\n"
        "  1  text arg     Print embedding as space-separated floats\n"
        "  2+ text args    Print cosine similarity matrix\n"
        "  no args         Batch: read all stdin lines, then similarity matrix\n"
        "  --stream        Streaming: read stdin lines, write JSON per line\n"
        "\n"
        "Examples:\n"
        "  %s -d ./model \"query: what is AI?\"\n"
#ifdef USE_MLX
        "  %s -d ./model --mlx --stream < texts.txt\n",
        prog, prog, prog);
#else
        ,
        prog, prog);
#endif
}

/* ========================================================================
 * Embed one text: tokenize > forward > return float[dim]
 * ======================================================================== */

typedef struct {
    embed_model_t     *model;
    embed_workspace_t *workspace;
#ifdef USE_MLX
    embed_mlx_ctx_t   *mlx_ctx;
#endif
#ifdef USE_CUDA
    embed_cuda_ctx_t  *cuda_ctx;
#endif
    qwen_tokenizer_t *tok;
    qwen_tokenizer_workspace_t *tok_ws;
    int               dim;
    int               terminal_token_id;
    int               append_terminal_token;
} engine_t;

typedef struct {
    int *ids;
    int  n_tokens;
} token_buf_t;

static int engine_embed_batch(engine_t *e, const embed_input_t *inputs,
                              int batch, float *out_embeddings)
{
#ifdef USE_MLX
    if (e->mlx_ctx)
        return embed_mlx_encode_batch(e->mlx_ctx, inputs, batch, out_embeddings);
#endif
#ifdef USE_CUDA
    if (e->cuda_ctx)
        return embed_cuda_encode_batch(e->cuda_ctx, inputs, batch, out_embeddings);
#endif
    return embed_model_encode_batch(e->model, e->workspace,
                                  inputs, batch, out_embeddings);
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
    if (e->append_terminal_token) {
        if (out->n_tokens == INT_MAX) {
            free(out->ids);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        int *ids = (int *)realloc(
            out->ids, (size_t)(out->n_tokens + 1) * sizeof(*out->ids));
        if (!ids) {
            free(out->ids);
            memset(out, 0, sizeof(*out));
            return -1;
        }
        out->ids = ids;
        out->ids[out->n_tokens++] = e->terminal_token_id;
    }

    if (embed_verbose >= 1) {
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

static size_t engine_workspace_nbytes(const engine_t *e)
{
    return (e && e->workspace) ? embed_workspace_nbytes(e->workspace) : 0;
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
    embed_input_t *inputs = (embed_input_t *)malloc((size_t)n_lines * sizeof(embed_input_t));
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
        if (embed_verbose >= 1)
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

    if (embed_verbose >= 1)
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

        if (embed_verbose >= 1)
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

    if (embed_verbose >= 1)
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
    embed_input_t *inputs = (embed_input_t *)malloc((size_t)n_texts * sizeof(embed_input_t));
    float *embs = (float *)malloc((size_t)n_texts * dim * sizeof(float));
    if (!tokens || !inputs || !embs) {
        fprintf(stderr, "OOM\n");
        free(tokens); free(inputs); free(embs);
        for (int i = 0; i < n_texts; i++) free(texts[i]);
        free(texts);
        return 1;
    }

    for (int i = 0; i < n_texts; i++) {
        if (embed_verbose >= 1)
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
        if (embed_verbose >= 1)
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
                printf("  %.4f", (double)embed_cosine_similarity(
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
    int stdin_mode = 0;
    int batch_size = 0;
#ifdef USE_MLX
    int use_mlx = 0;
    int mlx_quantize_bits = 0;
    int mlx_quantize_group_size = 64;
#endif
#ifdef USE_CUDA
    int use_cuda = 0;
    const char *cuda_fast_gemm = NULL;
    const char *cuda_weights = NULL;
#endif

    int arg_start = 1;
    while (arg_start < argc && argv[arg_start][0] == '-') {
        const char *f = argv[arg_start];
        if      (!strcmp(f, "-d"))      { model_dir = argv[++arg_start]; }
        else if (!strcmp(f, "-t") || !strcmp(f, "--threads")) {
            n_threads = atoi(argv[++arg_start]);
        }
        else if (!strcmp(f, "-b") || !strcmp(f, "--batch-size")) {
            batch_size = atoi(argv[++arg_start]);
        }
#ifdef USE_MLX
        else if (!strcmp(f, "--mlx-quant-bits")) {
            mlx_quantize_bits = atoi(argv[++arg_start]);
            if (mlx_quantize_bits != 0 && mlx_quantize_bits != 8) {
                fprintf(stderr, "--mlx-quant-bits must be 0 or 8\n");
                return 1;
            }
        }
        else if (!strcmp(f, "--mlx-quant-group-size")) {
            mlx_quantize_group_size = atoi(argv[++arg_start]);
            if (mlx_quantize_group_size <= 0) {
                fprintf(stderr, "--mlx-quant-group-size must be > 0\n");
                return 1;
            }
        }
#endif
#ifdef USE_CUDA
        else if (!strcmp(f, "--cuda-gemm-mode")) {
            cuda_fast_gemm = argv[++arg_start];
        }
        else if (!strcmp(f, "--cuda-weight-dtype")) {
            cuda_weights = argv[++arg_start];
        }
#endif
        else if (!strcmp(f, "-e") || !strcmp(f, "--embeddings")) {
            print_embs = 1;
        }
        else if (!strcmp(f, "-v") || !strcmp(f, "--verbose")) { verbose++; }
        else if (!strcmp(f, "-vv"))     { verbose = 2; }
        else if (!strcmp(f, "--stream")) { stdin_mode = 1; }
#ifdef USE_MLX
        else if (!strcmp(f, "--mlx"))   { use_mlx = 1; }
#endif
#ifdef USE_CUDA
        else if (!strcmp(f, "--cuda"))  { use_cuda = 1; }
#endif
        else if (!strcmp(f, "-h") || !strcmp(f, "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        else break;
        arg_start++;
    }

    embed_verbose = verbose;
    qwen_verbose = verbose;

#ifdef USE_CUDA
    if (cuda_fast_gemm) {
        if (!use_cuda) {
            fprintf(stderr, "--cuda-gemm-mode requires --cuda\n");
            return 1;
        }
        if (embed_cuda_set_fast_gemm(cuda_fast_gemm) != 0) {
            fprintf(stderr, "--cuda-gemm-mode must be f32, tf32, bf16, or 16f\n");
            return 1;
        }
    }
    if (cuda_weights) {
        if (!use_cuda) {
            fprintf(stderr, "--cuda-weight-dtype requires --cuda\n");
            return 1;
        }
        if (!strcmp(cuda_weights, "bf16")) {
            embed_cuda_set_weights_bf16(1);
        } else if (!strcmp(cuda_weights, "f32")) {
            embed_cuda_set_weights_bf16(0);
        } else {
            fprintf(stderr, "--cuda-weight-dtype must be f32 or bf16\n");
            return 1;
        }
    }
#endif
#ifdef USE_MLX
    if (mlx_quantize_bits && !use_mlx) {
        fprintf(stderr, "--mlx-quant-bits requires --mlx\n");
        return 1;
    }
#endif

    if (!model_dir || !model_dir[0]) {
        fprintf(stderr, "model directory required (-d <dir>)\n");
        print_usage(argv[0]);
        return 1;
    }

#ifdef USE_MLX
    if (use_mlx) {
        if (verbose >= 1) fprintf(stderr, "Using MLX GPU backend\n");
    } else
#endif
#ifdef USE_CUDA
    if (use_cuda) {
        if (verbose >= 1) fprintf(stderr, "Using CUDA GPU backend\n");
    } else
#endif
    {
        if (n_threads <= 0) n_threads = qwen_get_num_cpus();
        qwen_set_threads(n_threads);
        if (verbose >= 1) fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
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
    const embed_config_t *config = NULL;

#ifdef USE_MLX
    if (use_mlx) {
        embed_mlx_options_t mlx_opts = {
            .quantize_bits = mlx_quantize_bits,
            .quantize_group_size = mlx_quantize_group_size,
        };
        e.mlx_ctx = embed_mlx_load_with_options(model_dir, &mlx_opts);
        if (!e.mlx_ctx) {
            fprintf(stderr, "failed to load model\n");
            qwen_tokenizer_free(tok);
            return 1;
        }
        config = embed_mlx_config(e.mlx_ctx);
        e.dim = config->hidden_size;
    } else
#endif
#ifdef USE_CUDA
    if (use_cuda) {
        e.cuda_ctx = embed_cuda_load(model_dir);
        if (!e.cuda_ctx) {
            fprintf(stderr, "failed to load CUDA model\n");
            qwen_tokenizer_free(tok);
            return 1;
        }
        config = embed_cuda_config(e.cuda_ctx);
        e.dim = config->hidden_size;
    } else
#endif
    {
        e.model = embed_model_load(model_dir);
        if (!e.model) {
            fprintf(stderr, "failed to load model\n");
            qwen_tokenizer_free(tok);
            return 1;
        }
        e.workspace = embed_workspace_new(e.model);
        if (!e.workspace) {
            fprintf(stderr, "failed to allocate workspace\n");
            embed_model_free(e.model);
            qwen_tokenizer_free(tok);
            return 1;
        }
        config = embed_model_config(e.model);
        e.dim = config->hidden_size;
    }
    e.append_terminal_token = config->append_terminal_token;
    e.terminal_token_id = config->terminal_token_id;
    if (verbose >= 1)
        fprintf(stderr, "Model: %d-dim, %.0f ms%s\n",
                e.dim, now_ms() - t0,
#ifdef USE_MLX
                use_mlx ? " (MLX)" :
#endif
#ifdef USE_CUDA
                use_cuda ? " (CUDA)" :
#endif
                "");

    e.tok_ws = qwen_tokenizer_workspace_new();
    if (!e.tok_ws) {
        fprintf(stderr, "failed to allocate tokenizer workspace\n");
        if (e.workspace) embed_workspace_free(e.workspace);
        if (e.model) embed_model_free(e.model);
#ifdef USE_MLX
        if (e.mlx_ctx) embed_mlx_free(e.mlx_ctx);
#endif
#ifdef USE_CUDA
        if (e.cuda_ctx) embed_cuda_free(e.cuda_ctx);
#endif
        qwen_tokenizer_free(tok);
        return 1;
    }

    /* Run */
    int rc;
    if (stdin_mode)
        rc = run_stdin(&e, batch_size > 0 ? batch_size : 1);
    else
        rc = run_batch(&e, argc, argv, arg_start, print_embs, batch_size);

    /* Cleanup */
    if (e.workspace) embed_workspace_free(e.workspace);
    if (e.model) embed_model_free(e.model);
#ifdef USE_MLX
    if (e.mlx_ctx) embed_mlx_free(e.mlx_ctx);
#endif
#ifdef USE_CUDA
    if (e.cuda_ctx) embed_cuda_free(e.cuda_ctx);
#endif
    qwen_tokenizer_workspace_free(e.tok_ws);
    qwen_tokenizer_free(tok);
    return rc;
}
