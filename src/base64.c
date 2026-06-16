#include "base64.h"
#include "server_util.h"

#include <stdint.h>

char *base64_encode(const unsigned char *src, size_t n) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t max_groups = (SIZE_MAX - 1) / 4;
    if (n > max_groups * 3)
        die_oom();

    size_t out_len = ((n + 2) / 3) * 4;
    char *out = xmalloc(out_len + 1);
    const unsigned char *p = src;
    const unsigned char *end = src + (n / 3) * 3;
    char *q = out;

    while (p < end) {
        uint32_t v = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
        q[0] = tbl[(v >> 18) & 63];
        q[1] = tbl[(v >> 12) & 63];
        q[2] = tbl[(v >> 6) & 63];
        q[3] = tbl[v & 63];
        p += 3;
        q += 4;
    }

    size_t rem = n - (size_t)(p - src);
    if (rem == 1) {
        uint32_t v = (uint32_t)p[0] << 16;
        q[0] = tbl[(v >> 18) & 63];
        q[1] = tbl[(v >> 12) & 63];
        q[2] = '=';
        q[3] = '=';
        q += 4;
    } else if (rem == 2) {
        uint32_t v = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8);
        q[0] = tbl[(v >> 18) & 63];
        q[1] = tbl[(v >> 12) & 63];
        q[2] = tbl[(v >> 6) & 63];
        q[3] = '=';
        q += 4;
    }

    *q = '\0';
    return out;
}
