#include "bench.h"
#include "base64.h"
#include "sbuf.h"

#include <stdint.h>

static unsigned char *make_bytes(size_t n) {
    unsigned char *p = malloc(n);
    if (!p)
        return NULL;
    uint32_t x = 0x12345678u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        p[i] = (unsigned char)(x >> 24);
    }
    return p;
}

static void bm_base64_1k(bench_state_t *b) {
    unsigned char *src = make_bytes(1024);
    bench_begin(b);
    for (long i = 0; i < b->n; i++) {
        char *out = base64_encode(src, 1024);
        bench_sink += (float)(unsigned char)out[0];
        free(out);
    }
    free(src);
}

static void bm_base64_16k(bench_state_t *b) {
    unsigned char *src = make_bytes(16 * 1024);
    bench_begin(b);
    for (long i = 0; i < b->n; i++) {
        char *out = base64_encode(src, 16 * 1024);
        bench_sink += (float)(unsigned char)out[0];
        free(out);
    }
    free(src);
}

static void bm_sbuf_putc_1k(bench_state_t *b) {
    sbuf s = {0};
    bench_begin(b);
    for (long i = 0; i < b->n; i++) {
        sbuf_clear(&s);
        for (int k = 0; k < 1024; k++)
            sbuf_putc(&s, (char)('a' + (k % 26)));
        bench_sink += (float)s.len;
    }
    sbuf_free(&s);
}

static void bm_sbuf_printf_small(bench_state_t *b) {
    sbuf s = {0};
    sbuf_reserve(&s, 256);
    bench_begin(b);
    for (long i = 0; i < b->n; i++) {
        sbuf_clear(&s);
        sbuf_printf(&s, "{\"id\":%ld,\"value\":\"%s\",\"n\":%d}", i, "embedding", 1024);
        bench_sink += (float)s.len;
    }
    sbuf_free(&s);
}

static const bench_case_t cases[] = {
    {"server/base64_1k", bm_base64_1k},
    {"server/base64_16k", bm_base64_16k},
    {"server/sbuf_putc_1k", bm_sbuf_putc_1k},
    {"server/sbuf_printf_small", bm_sbuf_printf_small},
};

int main(int argc, char **argv) {
    bench_opts_t opts = {0};
    opts.meta = "suite=server-utils";
    bench_parse_args(&opts, argc, argv);
    return bench_main(&opts, cases, (int)(sizeof(cases) / sizeof(cases[0])));
}
