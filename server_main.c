/* server_main.c - pplx-embed-server: Perplexity-compatible embeddings API */

#include "pplx_embed.h"
#include "pplx_server.h"
#include "qwen_asr_kernels.h"

#ifdef USE_CUDA
#include "pplx_embed_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --model ID=DIR [options]\n"
        "\n"
        "Serves POST /v1/embeddings and /v1/contextualizedembeddings.\n"
        "\n"
        "Options:\n"
        "  --model ID=DIR            Model to serve (repeatable)\n"
        "  --backend cpu|mlx|cuda    Execution backend (default: cpu); --mlx and\n"
        "                            --cuda are shorthands\n"
        "  --host HOST               Bind host (default: 127.0.0.1)\n"
        "  --port N                  Bind port (default: 8000)\n"
        "  --api-key K               Require Authorization: Bearer K\n"
        "  --cors                    Enable CORS headers\n"
        "  -b, --batch-size N        Max texts or documents per inference batch\n"
        "                            (default: 8)\n"
        "  --max-batch-tokens N      Max tokens per inference batch (default: 16384)\n"
        "  --batch-wait-us N         Micro-batch wait in microseconds\n"
        "                            (default: 1000; 0 drains only queued work)\n"
        "  --memory-utilization F    Fraction of physical memory the MLX model-set\n"
        "                            preflight may plan for (default: 0.90; values\n"
        "                            above 1.0 overcommit)\n"
        "  --mlx-quant-bits N        Quantize MLX linear weights to 8 bits at load\n"
        "  --mlx-quant-group-size N  MLX quantization group size (default: 64)\n"
        "  --cuda-gemm-mode MODE     CUDA GEMM compute: f32, tf32, bf16, or 16f\n"
        "                            (default: f32, exact)\n"
        "  --cuda-weight-dtype DTYPE CUDA weight storage: f32 or bf16 (default:\n"
        "                            f32); bf16 halves weight memory\n"
        "  -t, --threads N           CPU threads (default: all cores)\n"
        "  -v, --verbose             Verbose (-vv for debug)\n"
        "  -h, --help                Show this help\n"
        "\n"
        "Examples:\n"
        "  %s --model pplx-embed-v1-0.6b=./model --port 8000\n"
        "  %s --backend mlx --mlx-quant-bits 8 \\\n"
        "      --model pplx-embed-v1-4b=./model-4b-bf16\n",
        prog, prog, prog);
}

typedef struct {
    pplx_server_model_spec_t *v;
    int n;
    int cap;
} model_specs_t;

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

int main(int argc, char *argv[])
{
    int n_threads = 0;
    int verbose = 0;
    int use_mlx = 0;
    int use_cuda = 0;
    int batch_size = 0;
    int max_batch_tokens = 0;
    int batch_wait_us = -1;
    int mlx_quantize_bits = 0;
    int mlx_quantize_group_size = 64;
    const char *cuda_fast_gemm = NULL;
    const char *cuda_weights = NULL;
    double memory_utilization = 0.0;
    const char *host = "127.0.0.1";
    const char *api_key = NULL;
    int port = 8000;
    int cors = 0;
    model_specs_t model_specs = {0};

    int arg = 1;
    while (arg < argc && argv[arg][0] == '-') {
        const char *f = argv[arg];
        if (!strcmp(f, "--model")) {
            if (append_model_spec(&model_specs, argv[++arg]) != 0) {
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--backend")) {
            const char *backend = argv[++arg];
            if (!strcmp(backend, "mlx")) {
#ifdef USE_MLX
                use_mlx = 1;
                use_cuda = 0;
#else
                fprintf(stderr, "mlx backend not available (build with: make mlx)\n");
                free_model_specs(&model_specs);
                return 1;
#endif
            } else if (!strcmp(backend, "cuda")) {
#ifdef USE_CUDA
                use_cuda = 1;
                use_mlx = 0;
#else
                fprintf(stderr, "cuda backend not available (build with: make cuda)\n");
                free_model_specs(&model_specs);
                return 1;
#endif
            } else if (!strcmp(backend, "cpu")) {
                use_mlx = 0;
                use_cuda = 0;
            } else {
                fprintf(stderr, "invalid --backend: %s\n", backend);
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--mlx")) {
#ifdef USE_MLX
            use_mlx = 1;
            use_cuda = 0;
#else
            fprintf(stderr, "--mlx not available (build with: make mlx)\n");
            free_model_specs(&model_specs);
            return 1;
#endif
        }
        else if (!strcmp(f, "--cuda")) {
#ifdef USE_CUDA
            use_cuda = 1;
            use_mlx = 0;
#else
            fprintf(stderr, "--cuda not available (build with: make cuda)\n");
            free_model_specs(&model_specs);
            return 1;
#endif
        }
        else if (!strcmp(f, "--host"))    { host = argv[++arg]; }
        else if (!strcmp(f, "--port"))    { port = atoi(argv[++arg]); }
        else if (!strcmp(f, "--api-key")) { api_key = argv[++arg]; }
        else if (!strcmp(f, "--cors"))    { cors = 1; }
        else if (!strcmp(f, "-b") || !strcmp(f, "--batch-size")) {
            batch_size = atoi(argv[++arg]);
        }
        else if (!strcmp(f, "--max-batch-tokens")) {
            max_batch_tokens = atoi(argv[++arg]);
            if (max_batch_tokens <= 0) {
                fprintf(stderr, "--max-batch-tokens must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--batch-wait-us")) {
            batch_wait_us = atoi(argv[++arg]);
            if (batch_wait_us < 0) {
                fprintf(stderr, "--batch-wait-us must be >= 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--memory-utilization")) {
            memory_utilization = atof(argv[++arg]);
            if (memory_utilization <= 0.0) {
                fprintf(stderr, "--memory-utilization must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--mlx-quant-bits")) {
            mlx_quantize_bits = atoi(argv[++arg]);
            if (mlx_quantize_bits != 0 && mlx_quantize_bits != 8) {
                fprintf(stderr, "--mlx-quant-bits must be 0 or 8\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--mlx-quant-group-size")) {
            mlx_quantize_group_size = atoi(argv[++arg]);
            if (mlx_quantize_group_size <= 0) {
                fprintf(stderr, "--mlx-quant-group-size must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        }
        else if (!strcmp(f, "--cuda-gemm-mode")) {
            cuda_fast_gemm = argv[++arg];
        }
        else if (!strcmp(f, "--cuda-weight-dtype")) {
            cuda_weights = argv[++arg];
        }
        else if (!strcmp(f, "-t") || !strcmp(f, "--threads")) {
            n_threads = atoi(argv[++arg]);
        }
        else if (!strcmp(f, "-v") || !strcmp(f, "--verbose")) { verbose++; }
        else if (!strcmp(f, "-vv")) { verbose = 2; }
        else if (!strcmp(f, "-h") || !strcmp(f, "--help")) {
            print_usage(argv[0]);
            free_model_specs(&model_specs);
            return 0;
        }
        else {
            fprintf(stderr, "unknown option: %s\n", f);
            print_usage(argv[0]);
            free_model_specs(&model_specs);
            return 1;
        }
        arg++;
    }

    pplx_verbose = verbose;
    qwen_verbose = verbose;

    if (arg < argc) {
        fprintf(stderr, "unexpected argument: %s\n", argv[arg]);
        print_usage(argv[0]);
        free_model_specs(&model_specs);
        return 1;
    }
    if (model_specs.n == 0) {
        fprintf(stderr, "at least one --model MODEL_ID=DIR is required\n");
        print_usage(argv[0]);
        free_model_specs(&model_specs);
        return 1;
    }
    if (mlx_quantize_bits && !use_mlx) {
        fprintf(stderr, "--mlx-quant-bits requires --backend mlx\n");
        free_model_specs(&model_specs);
        return 1;
    }

    if (cuda_fast_gemm) {
#ifdef USE_CUDA
        if (!use_cuda) {
            fprintf(stderr, "--cuda-gemm-mode requires --backend cuda\n");
            free_model_specs(&model_specs);
            return 1;
        }
        if (pplx_cuda_set_fast_gemm(cuda_fast_gemm) != 0) {
            fprintf(stderr, "--cuda-gemm-mode must be f32, tf32, bf16, or 16f\n");
            free_model_specs(&model_specs);
            return 1;
        }
#else
        fprintf(stderr, "--cuda-gemm-mode requires a CUDA build\n");
        free_model_specs(&model_specs);
        return 1;
#endif
    }

    if (cuda_weights) {
#ifdef USE_CUDA
        if (!use_cuda) {
            fprintf(stderr, "--cuda-weight-dtype requires --backend cuda\n");
            free_model_specs(&model_specs);
            return 1;
        }
        if (!strcmp(cuda_weights, "bf16")) {
            pplx_cuda_set_weights_bf16(1);
        } else if (!strcmp(cuda_weights, "f32")) {
            pplx_cuda_set_weights_bf16(0);
        } else {
            fprintf(stderr, "--cuda-weight-dtype must be f32 or bf16\n");
            free_model_specs(&model_specs);
            return 1;
        }
#else
        fprintf(stderr, "--cuda-weight-dtype requires a CUDA build\n");
        free_model_specs(&model_specs);
        return 1;
#endif
    }

    if (!use_mlx && !use_cuda) {
        if (n_threads <= 0) n_threads = qwen_get_num_cpus();
        qwen_set_threads(n_threads);
        if (verbose >= 1) fprintf(stderr, "Using %d CPU thread(s)\n", n_threads);
    } else if (verbose >= 1) {
        fprintf(stderr, "Using %s GPU backend\n", use_mlx ? "MLX" : "CUDA");
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
        .use_cuda = use_cuda,
        .mlx_quantize_bits = mlx_quantize_bits,
        .mlx_quantize_group_size = mlx_quantize_group_size,
        .memory_utilization = memory_utilization,
        .enable_cors = cors,
        .api_key = api_key,
    };
    int rc = pplx_run_server(&scfg);
    free_model_specs(&model_specs);
    return rc;
}
