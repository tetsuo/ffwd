#ifndef EMBED_SERVER_UTIL_H
#define EMBED_SERVER_UTIL_H

#include <stdint.h>
#include <stddef.h>

/* Die-on-OOM allocation, monotonic clocks, and timestamped logging shared by
 * the embed-server translation units. The x* allocators abort the process on
 * failure rather than returning NULL, so callers never null-check them; the
 * attributes below carry that contract across translation-unit boundaries so
 * the static analyzer does not flag the (correctly) unchecked uses. */

#if defined(__GNUC__) || defined(__clang__)
#define SERVER_NORETURN        __attribute__((noreturn))
#define SERVER_RETURNS_NONNULL __attribute__((returns_nonnull))
#else
#define SERVER_NORETURN
#define SERVER_RETURNS_NONNULL
#endif

SERVER_NORETURN void die_oom(void);

SERVER_RETURNS_NONNULL void *xmalloc(size_t n);
SERVER_RETURNS_NONNULL void *xcalloc(size_t n, size_t sz);
SERVER_RETURNS_NONNULL void *xrealloc(void *p, size_t n);
SERVER_RETURNS_NONNULL char *xstrdup(const char *s);
SERVER_RETURNS_NONNULL char *xstrndup(const char *s, size_t n);

uint64_t mstime(void);
uint64_t nstime(void);
double ns_to_ms(uint64_t ns);

/* printf-style line logger: timestamp prefix, newline appended, to stderr. */
void server_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

#endif /* EMBED_SERVER_UTIL_H */
