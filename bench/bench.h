/* bench/bench.h - minimal Go-style microbenchmark harness.
 *
 * A benchmark function receives a bench_state_t and must run its measured
 * loop exactly b->n times. Call bench_begin(b) after setup so allocation
 * and fill time stay out of the measurement:
 *
 *     static void bm_dot(bench_state_t *b) {
 *         float *x = make_input();          // setup, not measured
 *         bench_begin(b);
 *         for (long i = 0; i < b->n; i++)
 *             bench_sink += qwen_dot_f32(x, x, 1024);
 *         free(x);
 *     }
 *
 * The harness calibrates n until one run takes at least --benchtime-ms
 * (default 300), then takes --count (default 10) timed samples and prints
 * one Go-style line per benchmark with the median ns/op and the spread.
 * --json PATH writes every sample plus run metadata for benchstat.py.
 * --filter SUBSTR runs the matching subset.
 */

#ifndef BENCH_H
#define BENCH_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    long n;          /* iterations: the measured loop must run exactly n times */
    double t0_ns;    /* set by bench_begin(); end of un-measured setup */
} bench_state_t;

typedef void (*bench_fn_t)(bench_state_t *);

typedef struct {
    const char *name;
    bench_fn_t fn;
} bench_case_t;

/* Accumulate something cheap into this so results stay observable and the
 * compiler cannot delete the measured loop. */
static volatile float bench_sink;

static double bench_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void bench_begin(bench_state_t *b)
{
    b->t0_ns = bench_now_ns();
}

/* One timed run of fn at n iterations; returns ns/op. */
static double bench_run_once(bench_fn_t fn, long n)
{
    bench_state_t b = { n, 0.0 };
    b.t0_ns = bench_now_ns();
    fn(&b);
    double elapsed = bench_now_ns() - b.t0_ns;
    return elapsed / (double)n;
}

static int bench_cmp_double(const void *a, const void *c)
{
    double x = *(const double *)a, y = *(const double *)c;
    return (x > y) - (x < y);
}

static double bench_median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(*v), bench_cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

typedef struct {
    long benchtime_ms;
    int count;
    const char *filter;
    const char *json_path;
    /* free-form metadata recorded into the JSON, e.g. "threads=1" */
    const char *meta;
} bench_opts_t;

static void bench_parse_args(bench_opts_t *o, int argc, char **argv)
{
    o->benchtime_ms = 300;
    o->count = 10;
    o->filter = NULL;
    o->json_path = NULL;
    if (!o->meta) o->meta = "";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--benchtime-ms") && i + 1 < argc)
            o->benchtime_ms = atol(argv[++i]);
        else if (!strcmp(argv[i], "--count") && i + 1 < argc)
            o->count = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--filter") && i + 1 < argc)
            o->filter = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc)
            o->json_path = argv[++i];
    }
    if (o->benchtime_ms < 1) o->benchtime_ms = 1;
    if (o->count < 1) o->count = 1;
    if (o->count > 1000) o->count = 1000;
}

/* Run all cases (subject to the filter) and report. Returns 0 on success. */
static int bench_main(const bench_opts_t *o,
                      const bench_case_t *cases, int n_cases)
{
    FILE *jf = NULL;
    if (o->json_path) {
        jf = fopen(o->json_path, "w");
        if (!jf) {
            fprintf(stderr, "cannot write %s\n", o->json_path);
            return 1;
        }
        char host[256] = "unknown";
        gethostname(host, sizeof(host) - 1);
        fprintf(jf, "{\"host\":\"%s\",\"benchtime_ms\":%ld,\"count\":%d,"
                "\"meta\":\"%s\",\"benchmarks\":[",
                host, o->benchtime_ms, o->count, o->meta);
    }

    double *samples = (double *)malloc((size_t)o->count * sizeof(double));
    if (!samples) {
        if (jf) fclose(jf);
        return 1;
    }

    long *iters = (long *)calloc((size_t)n_cases, sizeof(long));
    double *all = (double *)calloc((size_t)n_cases * o->count, sizeof(double));
    if (!iters || !all) {
        free(all);
        free(iters);
        free(samples);
        if (jf) fclose(jf);
        return 1;
    }

    /* Calibrate every selected case first: grow n predictively until one
     * run is long enough. The calibration run doubles as warmup. */
    int ran = 0;
    double target_ns = (double)o->benchtime_ms * 1e6;
    for (int c = 0; c < n_cases; c++) {
        if (o->filter && !strstr(cases[c].name, o->filter))
            continue;
        long n = 1;
        double ns_per_op = bench_run_once(cases[c].fn, n);
        while (ns_per_op * (double)n < target_ns) {
            double want = target_ns / (ns_per_op > 0.0 ? ns_per_op : 1.0);
            long next = (long)(want * 1.2) + 1;
            if (next > n * 100) next = n * 100;   /* distrust tiny timings */
            if (next <= n) next = n + 1;
            n = next;
            ns_per_op = bench_run_once(cases[c].fn, n);
        }
        iters[c] = n;
        ran++;
        fprintf(stderr, "calibrated %-44s n=%ld\n", cases[c].name, n);
    }

    /* Sample round-robin so slow machine drift (thermals, scheduling)
     * lands inside every benchmark's sample spread instead of shifting
     * whole benchmarks against each other. Results print after the
     * last pass; the pass lines are the progress indicator. */
    double sample_t0 = bench_now_ns();
    for (int s = 0; s < o->count; s++) {
        for (int c = 0; c < n_cases; c++)
            if (iters[c])
                all[(size_t)c * o->count + s] =
                    bench_run_once(cases[c].fn, iters[c]);
        fprintf(stderr, "pass %d/%d (%.0fs elapsed)\n", s + 1, o->count,
                (bench_now_ns() - sample_t0) / 1e9);
    }

    int emitted = 0;
    for (int c = 0; c < n_cases; c++) {
        if (!iters[c])
            continue;
        memcpy(samples, all + (size_t)c * o->count,
               (size_t)o->count * sizeof(double));

        if (jf) {
            fprintf(jf, "%s{\"name\":\"%s\",\"iters\":%ld,\"ns_per_op\":[",
                    emitted ? "," : "", cases[c].name, iters[c]);
            for (int s = 0; s < o->count; s++)
                fprintf(jf, "%s%.3f", s ? "," : "", samples[s]);
            fprintf(jf, "]}");
        }

        double med = bench_median(samples, o->count);
        double spread = o->count > 1
            ? 100.0 * (samples[o->count - 1] - samples[0]) / med : 0.0;
        printf("%-44s %10ld iters %14.1f ns/op  (spread %.1f%%)\n",
               cases[c].name, iters[c], med, spread);
        fflush(stdout);
        emitted++;
    }

    free(all);
    free(iters);
    free(samples);
    if (jf) {
        fprintf(jf, "]}\n");
        fclose(jf);
        fprintf(stderr, "wrote %s\n", o->json_path);
    }
    if (!ran) {
        fprintf(stderr, "no benchmarks matched\n");
        return 1;
    }
    return 0;
}

#endif /* BENCH_H */
