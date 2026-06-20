#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

void die_oom(void) {
    fprintf(stderr, "ffwd-server: out of memory\n");
    exit(1);
}

void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p)
        die_oom();
    return p;
}

void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p)
        die_oom();
    return p;
}

void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p)
        die_oom();
    return p;
}

char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

uint64_t mstime(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000u + (uint64_t)tv.tv_usec / 1000u;
}

uint64_t nstime(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000000u + (uint64_t)tv.tv_usec * 1000u;
}

double ns_to_ms(uint64_t ns) { return (double)ns / 1000000.0; }

void server_log(const char *fmt, ...) {
    char ts[32];
    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof(ts), "%m%d %H:%M:%S", &tmv);
    fprintf(stderr, "%s ", ts);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}
