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
    size_t cap = b->cap ? b->cap : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) {
            cap = need;
            break;
        }
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

void sbuf_putc(sbuf *b, char c) {
    sbuf_reserve(b, 1);
    b->ptr[b->len++] = c;
    b->ptr[b->len] = '\0';
}

void sbuf_puts(sbuf *b, const char *s) { sbuf_append(b, s, strlen(s)); }

void sbuf_clear(sbuf *b) {
    b->len = 0;
    if (b->ptr)
        b->ptr[0] = '\0';
}

void sbuf_printf(sbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    for (;;) {
        size_t avail = b->cap > b->len ? b->cap - b->len : 0;
        va_list aq;
        va_copy(aq, ap);
        int n = vsnprintf(avail ? b->ptr + b->len : NULL, avail, fmt, aq);
        va_end(aq);
        if (n < 0) {
            va_end(ap);
            die_oom();
        }
        if ((size_t)n < avail) {
            b->len += (size_t)n;
            va_end(ap);
            return;
        }
        sbuf_reserve(b, (size_t)n);
    }
}

void sbuf_free(sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}
