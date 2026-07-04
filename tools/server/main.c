/* ffwd HTTP frontend.
 *
 * Handles argument parsing, model specs, and main().
 * Server logic lives in server.c; this file only builds ffwd_server_config_t
 * and calls ffwd_run_server.
 *
 * Split from server.c so tests can include server.c for internal functions
 * without a conflicting main() or compile-out guard.
 */

#include "build.h"
#include "server.h"
#include "server_internal.h"
#include "ffwd.h"
#include "internal.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --model LABEL=DIR [options]\n"
            "\n"
            "Options:\n"
            "  --model LABEL=DIR[:key=val]\n"
            "                            Standard embedding model (repeatable)\n"
            "  --contextual-model LABEL=DIR[:key=val]\n"
            "                            Contextual embedding model (repeatable)\n"
            "  --late-model LABEL=DIR[:key=val]\n"
            "                            Late-interaction rerank model (repeatable)\n"
            "                            keys: api=openai|perplexity, min_dim=N,\n"
            "                            query=TEXT (last key; \\n escapes newline)\n"
            "  --host HOST               Bind host (default: 127.0.0.1)\n"
            "  --port N                  Bind port (default: 8000)\n"
            "  --api-key K               Require Authorization: Bearer K\n"
            "  -b, --batch-size N        Max texts or documents per inference batch\n"
            "                            (default: 32)\n"
            "  --max-batch-tokens N      Max tokens per inference batch (default: 16384)\n"
            "  --max-request-tokens N    Max total tokens per API request (default: 120000)\n"
            "  --max-concurrent-requests N\n"
            "                            In-flight request cap; excess requests get 429\n"
            "                            (default: %d)\n"
            "  --auto-truncate           Truncate inputs that exceed the per-input token\n"
            "                            cap instead of rejecting them with 422\n"
            "  --batch-wait-us N         First-arrival micro-batch deadline in us\n"
            "                            (default: %d)\n",
            prog, FFWD_SERVER_DEFAULT_MAX_CONCURRENT, ffwd_default_batch_wait_us());
    ffwd_server_help(stderr);
    fprintf(stderr,
            "  -v, --verbose             Verbose (-vv for debug)\n"
            "  -V, --version             Print version and exit\n"
            "  --build-info              Print build details and exit\n"
            "  -h, --help                Show this help\n"
            "\n"
            "Examples:\n"
            "  %s --model docs=./model --port 8000\n"
            "  %s --model pplx=./model:api=perplexity:min_dim=128 --port 8000\n",
            prog, prog);
}

typedef struct {
    ffwd_server_model_spec_t *v;
    int n;
    int cap;
} model_specs_t;

static char *dup_range(const char *p, size_t n) {
    char *s = (char *)malloc(n + 1);
    if (!s)
        return NULL;
    memcpy(s, p, n);
    s[n] = '\0';
    return s;
}

static char *unescape_model_option(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out)
        return NULL;
    size_t w = 0;
    for (size_t r = 0; r < n; r++) {
        if (s[r] == '\\' && r + 1 < n) {
            r++;
            if (s[r] == 'n')
                out[w++] = '\n';
            else if (s[r] == 't')
                out[w++] = '\t';
            else if (s[r] == 'r')
                out[w++] = '\r';
            else
                out[w++] = s[r];
        } else {
            out[w++] = s[r];
        }
    }
    out[w] = '\0';
    return out;
}

static int parse_model_option(ffwd_server_model_spec_t *spec, const char *opt, const char *flag) {
    const char *eq = strchr(opt, '=');
    if (!eq || eq == opt || !eq[1]) {
        fprintf(stderr, "%s option expects key=value: %s\n", flag, opt);
        return -1;
    }
    size_t key_len = (size_t)(eq - opt);
    const char *val = eq + 1;
    if (key_len == 3 && strncmp(opt, "api", key_len) == 0) {
        if (!strcmp(val, "openai")) {
            spec->api = FFWD_SERVER_API_OPENAI;
        } else if (!strcmp(val, "perplexity") || !strcmp(val, "pplx")) {
            spec->api = FFWD_SERVER_API_PERPLEXITY;
        } else if (!strcmp(val, "default")) {
            spec->api = FFWD_SERVER_API_DEFAULT;
        } else {
            fprintf(stderr, "%s api must be openai or perplexity\n", flag);
            return -1;
        }
        return 0;
    }
    if (key_len == 7 && strncmp(opt, "min_dim", key_len) == 0) {
        char *end = NULL;
        long v = strtol(val, &end, 10);
        if (!end || *end || v <= 0 || v > INT_MAX) {
            fprintf(stderr, "%s min_dim must be a positive integer\n", flag);
            return -1;
        }
        spec->min_dim = (int)v;
        return 0;
    }
    fprintf(stderr, "%s unknown model option: %.*s\n", flag, (int)key_len, opt);
    return -1;
}

static int append_model_spec(model_specs_t *specs,
                             const char *arg,
                             ffwd_server_model_kind_t kind,
                             const char *flag) {
    const char *eq = strchr(arg, '=');
    if (!eq || eq == arg || !eq[1]) {
        fprintf(stderr, "%s expects LABEL=DIR\n", flag);
        return -1;
    }
    if (specs->n == specs->cap) {
        int new_cap = specs->cap ? specs->cap * 2 : 4;
        void *p = realloc(specs->v, (size_t)new_cap * sizeof(specs->v[0]));
        if (!p)
            return -1;
        specs->v = p;
        specs->cap = new_cap;
    }
    size_t id_len = (size_t)(eq - arg);
    const char *path_start = eq + 1;
    const char *opts = strchr(path_start, ':');
    size_t path_len = opts ? (size_t)(opts - path_start) : strlen(path_start);
    if (path_len == 0) {
        fprintf(stderr, "%s expects a non-empty DIR\n", flag);
        return -1;
    }
    ffwd_server_model_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.id = dup_range(arg, id_len);
    spec.path = dup_range(path_start, path_len);
    spec.kind = kind;
    if (!spec.id || !spec.path) {
        free((char *)spec.id);
        free((char *)spec.path);
        return -1;
    }
    const char *p = opts ? opts + 1 : NULL;
    while (p && *p) {
        if (!strncmp(p, "query=", 6) || !strncmp(p, "query_instruct=", 15)) {
            const char *val = p + (!strncmp(p, "query=", 6) ? 6 : 15);
            spec.query_instruct = unescape_model_option(val);
            if (!spec.query_instruct)
                goto fail;
            break;
        }
        const char *next = strchr(p, ':');
        size_t opt_len = next ? (size_t)(next - p) : strlen(p);
        char *opt = dup_range(p, opt_len);
        if (!opt)
            goto fail;
        int rc = parse_model_option(&spec, opt, flag);
        free(opt);
        if (rc != 0)
            goto fail;
        p = next ? next + 1 : NULL;
    }
    specs->v[specs->n] = spec;
    specs->n++;
    return 0;

fail:
    free((char *)spec.id);
    free((char *)spec.path);
    free((char *)spec.query_instruct);
    return -1;
}

static void free_model_specs(model_specs_t *specs) {
    if (!specs)
        return;
    for (int i = 0; i < specs->n; i++) {
        free((char *)specs->v[i].id);
        free((char *)specs->v[i].path);
        free((char *)specs->v[i].query_instruct);
    }
    free(specs->v);
}

int main(int argc, char *argv[]) {
    int verbose = 0;
    int batch_size = 0;
    int max_batch_tokens = 0;
    int max_request_tokens = 0;
    int max_concurrent_requests = 0;
    int auto_truncate = 0;
    int batch_wait_us = -1;
    /* Backend tuning flags parse into opts regardless of backend; the linked
     * backend uses what it supports. */
    ffwd_options_t opts = {.gpu_quant_group_size = 64};
    const char *host = "127.0.0.1";
    const char *api_key = NULL;
    int port = 8000;
    model_specs_t model_specs = {0};

    const char *prog = ffwd_prog_name(argv[0]);
    int arg = 1;
    while (arg < argc && argv[arg][0] == '-') {
        const char *f = argv[arg];
        if (!strcmp(f, "-V") || !strcmp(f, "--version")) {
            ffwd_print_version(prog);
            free_model_specs(&model_specs);
            return 0;
        } else if (!strcmp(f, "--build-info")) {
            ffwd_print_build_info();
            free_model_specs(&model_specs);
            return 0;
        } else if (!strcmp(f, "--model")) {
            if (arg + 1 >= argc ||
                append_model_spec(&model_specs, argv[++arg], FFWD_SERVER_MODEL_STANDARD, f) != 0) {
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--contextual-model")) {
            if (arg + 1 >= argc ||
                append_model_spec(&model_specs, argv[++arg], FFWD_SERVER_MODEL_CONTEXTUAL, f) != 0) {
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--late-model")) {
            if (arg + 1 >= argc ||
                append_model_spec(&model_specs, argv[++arg], FFWD_SERVER_MODEL_LATE, f) != 0) {
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--gpu-quant-bits")) {
            opts.gpu_quant_bits = atoi(argv[++arg]);
        } else if (!strcmp(f, "--gpu-quant-group-size")) {
            opts.gpu_quant_group_size = atoi(argv[++arg]);
        } else if (!strcmp(f, "--memory-utilization")) {
            opts.memory_utilization = atof(argv[++arg]);
            if (opts.memory_utilization <= 0.0) {
                fprintf(stderr, "--memory-utilization must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--gpu-gemm-mode")) {
            opts.gpu_gemm_mode = argv[++arg];
        } else if (!strcmp(f, "--gpu-weight-dtype")) {
            opts.gpu_weight_dtype = argv[++arg];
        } else if (!strcmp(f, "--host")) {
            host = argv[++arg];
        } else if (!strcmp(f, "--port")) {
            port = atoi(argv[++arg]);
        } else if (!strcmp(f, "--api-key")) {
            api_key = argv[++arg];
        } else if (!strcmp(f, "-b") || !strcmp(f, "--batch-size")) {
            batch_size = atoi(argv[++arg]);
        } else if (!strcmp(f, "--max-batch-tokens")) {
            max_batch_tokens = atoi(argv[++arg]);
            if (max_batch_tokens <= 0) {
                fprintf(stderr, "--max-batch-tokens must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--max-request-tokens")) {
            max_request_tokens = atoi(argv[++arg]);
            if (max_request_tokens <= 0) {
                fprintf(stderr, "--max-request-tokens must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--max-concurrent-requests")) {
            max_concurrent_requests = atoi(argv[++arg]);
            if (max_concurrent_requests <= 0) {
                fprintf(stderr, "--max-concurrent-requests must be > 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "--auto-truncate")) {
            auto_truncate = 1;
        } else if (!strcmp(f, "--batch-wait-us")) {
            batch_wait_us = atoi(argv[++arg]);
            if (batch_wait_us < 0) {
                fprintf(stderr, "--batch-wait-us must be >= 0\n");
                free_model_specs(&model_specs);
                return 1;
            }
        } else if (!strcmp(f, "-t") || !strcmp(f, "--threads")) {
            opts.n_threads = atoi(argv[++arg]);
        } else if (!strcmp(f, "-v") || !strcmp(f, "--verbose")) {
            verbose++;
        } else if (!strcmp(f, "-vv")) {
            verbose = 2;
        } else if (!strcmp(f, "-h") || !strcmp(f, "--help")) {
            print_usage(prog);
            free_model_specs(&model_specs);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", f);
            print_usage(prog);
            free_model_specs(&model_specs);
            return 1;
        }
        arg++;
    }

    ffwd_verbose = verbose;

    if (arg < argc) {
        fprintf(stderr, "unexpected argument: %s\n", argv[arg]);
        print_usage(prog);
        free_model_specs(&model_specs);
        return 1;
    }
    if (model_specs.n == 0) {
        fprintf(stderr, "at least one --model LABEL=DIR is required\n");
        print_usage(prog);
        free_model_specs(&model_specs);
        return 1;
    }
    char err[256];
    if (ffwd_init(&opts, err, sizeof(err)) != 0) {
        fprintf(stderr, "%s\n", err);
        free_model_specs(&model_specs);
        return 1;
    }

    ffwd_server_config_t scfg = {
        .models = model_specs.v,
        .n_models = model_specs.n,
        .host = host,
        .port = port,
        .batch_size = batch_size > 0 ? batch_size : FFWD_SERVER_DEFAULT_BATCH_SIZE,
        .max_batch_tokens = max_batch_tokens,
        .max_request_tokens = max_request_tokens,
        .max_concurrent_requests = max_concurrent_requests,
        .auto_truncate = auto_truncate,
        .batch_wait_us = batch_wait_us,
        .gpu_quantize_bits = opts.gpu_quant_bits,
        .gpu_quantize_group_size = opts.gpu_quant_group_size,
        .memory_utilization = opts.memory_utilization,
        .api_key = api_key,
    };
    int rc = ffwd_run_server(&scfg);
    free_model_specs(&model_specs);
    return rc;
}
