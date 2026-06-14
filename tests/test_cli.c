/* tests/test_cli.c - drives the embed CLI binary end to end.
 * Hermetic: synthesizes a tiny model + tokenizer fixture, then runs the
 * binary given as argv[1] through its modes and flag validation paths,
 * checking exit codes and output shapes. Runs via `make test`. */

#include "tiny_model.h"
#include "tok_fixture.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static char g_dir[1024];       /* fixture model dir */
static char g_out[1100];       /* captured stdout */
static char g_err[1100];       /* captured stderr */
static char g_in[1100];        /* stdin source */
static const char *g_cli;
static int g_failures;

/* Run the CLI with sh-formatted args; returns its exit status (-1 on a
 * harness error). stdin_text == NULL feeds /dev/null. */
static int run_cli(const char *args, const char *stdin_text)
{
    if (stdin_text) {
        FILE *f = fopen(g_in, "w");
        if (!f) return -1;
        fputs(stdin_text, f);
        fclose(f);
    }
    char cmd[4096];
    int n = snprintf(cmd, sizeof(cmd), "'%s' %s > '%s' 2> '%s' < '%s'",
                     g_cli, args, g_out, g_err,
                     stdin_text ? g_in : "/dev/null");
    if (n < 0 || (size_t)n >= sizeof(cmd)) return -1;
    int st = system(cmd);
    if (st == -1 || !WIFEXITED(st)) return -1;
    return WEXITSTATUS(st);
}

static void expect(int cond, const char *what)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

static int file_contains(const char *path, const char *needle)
{
    static char buf[1 << 16];
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, needle) != NULL;
}

static int count_lines(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int lines = 0, c;
    while ((c = fgetc(f)) != EOF)
        if (c == '\n') lines++;
    fclose(f);
    return lines;
}

/* First stdout line must hold exactly `dim` finite floats. */
static int stdout_is_embedding(int dim)
{
    FILE *f = fopen(g_out, "r");
    if (!f) return 0;
    int count = 0;
    float v;
    while (fscanf(f, "%f", &v) == 1) {
        if (!isfinite((double)v)) {
            fclose(f);
            return 0;
        }
        count++;
    }
    fclose(f);
    return count == dim;
}

int main(int argc, char **argv)
{
    g_cli = argc > 1 ? argv[1] : "./embed";

    tm_dims_t dims = {4, 2, 1, 2, 8, TF_VOCAB_SIZE};
    snprintf(g_dir, sizeof(g_dir), "%s/embed-cli-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(g_dir) || tf_write_vocab(g_dir) != 0 ||
        tm_write_model_dims(g_dir, "F32", &dims) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }
    snprintf(g_out, sizeof(g_out), "%s/stdout.txt", g_dir);
    snprintf(g_err, sizeof(g_err), "%s/stderr.txt", g_dir);
    snprintf(g_in, sizeof(g_in), "%s/stdin.txt", g_dir);

    char args[2048];

    /* Help and missing-argument handling. */
    expect(run_cli("-h", NULL) == 0, "-h exits 0");
    expect(run_cli("", NULL) == 1, "no arguments exits 1");
    expect(run_cli("-d /nonexistent-embed-dir hi", NULL) == 1,
           "missing model dir exits 1");

    /* Mode 1: one text prints one embedding line. */
    snprintf(args, sizeof(args), "-d '%s' 'hello world'", g_dir);
    expect(run_cli(args, NULL) == 0, "single text exits 0");
    expect(stdout_is_embedding(dims.hidden), "single text prints dim floats");

    /* Thread count. */
    snprintf(args, sizeof(args),
             "-d '%s' -t 2 -v 'hello world'", g_dir);
    expect(run_cli(args, NULL) == 0, "-t 2 -v exits 0");
    expect(file_contains(g_err, "Using 2 CPU thread(s)"),
           "-v reports the thread count");

    /* Mode 2+: several texts print a similarity matrix. */
    snprintf(args, sizeof(args), "-d '%s' 'hello world' 'held'", g_dir);
    expect(run_cli(args, NULL) == 0, "two texts exit 0");
    expect(file_contains(g_out, "Cosine similarity matrix (2 texts):"),
           "two texts print a similarity matrix");
    expect(file_contains(g_out, "1.0000"), "matrix diagonal is 1.0000");

    snprintf(args, sizeof(args),
             "-d '%s' -e -b 1 'hello' 'world' 'held'", g_dir);
    expect(run_cli(args, NULL) == 0, "-e -b 1 with three texts exits 0");
    expect(file_contains(g_out, "Cosine similarity matrix (3 texts):"),
           "-b 1 still prints the full matrix");
    expect(file_contains(g_out, "Embeddings:"), "-e prints raw embeddings");

    /* Batch mode reads stdin when no texts are given. */
    snprintf(args, sizeof(args), "-d '%s'", g_dir);
    expect(run_cli(args, "hello world\nheld\n") == 0, "stdin batch exits 0");
    expect(file_contains(g_out, "Cosine similarity matrix (2 texts):"),
           "stdin batch prints a similarity matrix");
    expect(run_cli(args, "\n\n") == 1, "blank stdin exits 1");

    /* Streaming mode writes one JSON object per input line. */
    snprintf(args, sizeof(args), "-d '%s' --stream", g_dir);
    expect(run_cli(args, "hello\nworld\n") == 0, "--stream exits 0");
    expect(count_lines(g_out) == 2, "--stream writes one line per input");
    expect(file_contains(g_out, "{\"embedding\":["),
           "--stream writes embedding JSON");

    snprintf(args, sizeof(args), "-d '%s' --stream -b 2 -v", g_dir);
    expect(run_cli(args, "hello\nworld\nheld\n") == 0,
           "--stream -b 2 exits 0");
    expect(count_lines(g_out) == 3, "--stream -b 2 covers the tail batch");

    /* Flag validation: missing model dir must exit 1. */
    static const struct { const char *args; const char *what; } bad[] = {
        {"-d '' t", "empty model dir"},
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
        expect(run_cli(bad[i].args, NULL) == 1, bad[i].what);

    if (g_failures) {
        fprintf(stderr, "%d CLI check(s) failed\n", g_failures);
        return 1;
    }
    printf("ok: CLI modes and flag validation against %s\n", g_cli);
    return 0;
}
