/* Write a tiny model fixture to a directory.
 *
 * Thin command-line front end over tests/model_fixture.h, so out-of-process
 * callers (shell scripts, Python) can materialize the same synthetic model the
 * tests build in memory. The weights are deterministic but meaningless: the
 * fixture exercises loading and CLI/server plumbing, not embedding quality.
 *
 * Usage: gen_fixture --out DIR [--dtype F32|BF16|F16] [--model base|qwen3|qwen2]
 *                    [--hidden N] [--heads N] [--kv-heads N] [--head-dim N]
 *                    [--intermediate N] [--vocab N] [--eos-id N] [--no-vocab]
 *                    [--late-dim N]
 *
 * --late-dim N also writes a 1_Dense/linear.weight projection (token_dim N over
 * the hidden size), turning the fixture into a late-interaction/rerank model.
 */
#include "model_fixture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *f) {
    fputs("usage: gen_fixture --out DIR [--dtype F32|BF16|F16]\n"
          "                   [--model base|qwen3|qwen2]\n"
          "                   [--hidden N] [--heads N] [--kv-heads N] [--head-dim N]\n"
          "                   [--intermediate N] [--vocab N] [--eos-id N] [--no-vocab]\n"
          "                   [--late-dim N]\n",
          f);
}

int main(int argc, char **argv) {
    const char *out = NULL;
    const char *dtype = "F32";
    mf_model_kind_t model = MF_MODEL_BASE;
    int eos_id = TF_EOT_ID;
    int write_vocab = 1;
    int late_dim = 0;
    tm_dims_t d = mf_default_dims();

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "--out") && i + 1 < argc) {
            out = argv[++i];
        } else if (!strcmp(a, "--dtype") && i + 1 < argc) {
            dtype = argv[++i];
        } else if (!strcmp(a, "--model") && i + 1 < argc) {
            if (mf_parse_model_kind(argv[++i], &model) != 0) {
                fprintf(stderr, "gen_fixture: --model must be base, qwen3, or qwen2\n");
                return 2;
            }
        } else if (!strcmp(a, "--hidden") && i + 1 < argc) {
            d.hidden = atoi(argv[++i]);
        } else if (!strcmp(a, "--heads") && i + 1 < argc) {
            d.heads = atoi(argv[++i]);
        } else if (!strcmp(a, "--kv-heads") && i + 1 < argc) {
            d.kv_heads = atoi(argv[++i]);
        } else if (!strcmp(a, "--head-dim") && i + 1 < argc) {
            d.head_dim = atoi(argv[++i]);
        } else if (!strcmp(a, "--intermediate") && i + 1 < argc) {
            d.intermediate = atoi(argv[++i]);
        } else if (!strcmp(a, "--vocab") && i + 1 < argc) {
            d.vocab = atoi(argv[++i]);
        } else if (!strcmp(a, "--eos-id") && i + 1 < argc) {
            eos_id = atoi(argv[++i]);
        } else if (!strcmp(a, "--late-dim") && i + 1 < argc) {
            late_dim = atoi(argv[++i]);
        } else if (!strcmp(a, "--no-vocab")) {
            write_vocab = 0;
        } else {
            fprintf(stderr, "gen_fixture: unknown or incomplete option: %s\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (!out) {
        fprintf(stderr, "gen_fixture: --out DIR is required\n");
        usage(stderr);
        return 2;
    }
    if (!mf_valid_dtype(dtype)) {
        fprintf(stderr, "gen_fixture: --dtype must be F32, BF16, or F16\n");
        return 2;
    }

    if (mf_write_fixture(out, dtype, &d, model, eos_id, write_vocab, /*qwen2_zero_bias=*/0) != 0) {
        fprintf(stderr, "gen_fixture: failed to write fixture to %s\n", out);
        return 1;
    }

    /* A late-interaction/rerank model adds a Dense projection over the hidden
     * size; the base weights and vocab above are otherwise unchanged. */
    if (late_dim > 0 && tm_write_late_projection(out, dtype, late_dim, d.hidden) != 0) {
        fprintf(stderr, "gen_fixture: failed to write late projection to %s\n", out);
        return 1;
    }

    printf("%s\n", out);
    return 0;
}
