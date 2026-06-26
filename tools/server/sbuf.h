#ifndef FFWD_SBUF_H
#define FFWD_SBUF_H

#include <stddef.h>

/* Growable, always-NUL-terminated byte buffer for HTTP responses and JSON bodies.
 * Growth is die-on-OOM; see server_util.h.
 * Mutators never fail, so callers do not check return values. */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} sbuf;

void sbuf_reserve(sbuf *b, size_t add);
void sbuf_append(sbuf *b, const void *p, size_t n);
void sbuf_putc(sbuf *b, char c);
void sbuf_puts(sbuf *b, const char *s);
void sbuf_clear(sbuf *b);
void sbuf_printf(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
void sbuf_free(sbuf *b);

#endif /* FFWD_SBUF_H */
