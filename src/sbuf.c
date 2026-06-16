#include "sbuf.h"
#include "server_util.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sbuf_reserve(sbuf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1)
        die_oom();
    size_t need = b->len + add + 1;
    if (need <= b->cap)
        return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2)
            die_oom();
        cap *= 2;
    }
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

void sbuf_append(sbuf *b, const void *p, size_t n) {
    sbuf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

void sbuf_putc(sbuf *b, char c) { sbuf_append(b, &c, 1); }
void sbuf_puts(sbuf *b, const char *s) { sbuf_append(b, s, strlen(s)); }

void sbuf_clear(sbuf *b) {
    b->len = 0;
    if (b->ptr)
        b->ptr[0] = '\0';
}

void sbuf_printf(sbuf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0)
        die_oom();
    sbuf_reserve(b, (size_t)n);
    int n2 = vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    if (n2 < 0 || n2 != n)
        die_oom();
    b->len += (size_t)n;
}

void sbuf_free(sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}
