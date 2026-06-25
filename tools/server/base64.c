#include "base64.h"
#include "util.h"

#include <stdint.h>

static const char b64_tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t base64_encoded_len(size_t n) {
    size_t max_groups = (SIZE_MAX - 1) / 4;
    if (n > max_groups * 3)
        die_oom();
    return ((n + 2) / 3) * 4;
}

void base64_encode_to(char *dst, const unsigned char *src, size_t n) {
    const unsigned char *p = src;
    const unsigned char *end = src + (n / 3) * 3;
    char *q = dst;

    while (p < end) {
        uint32_t v = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
        q[0] = b64_tbl[(v >> 18) & 63];
        q[1] = b64_tbl[(v >> 12) & 63];
        q[2] = b64_tbl[(v >> 6) & 63];
        q[3] = b64_tbl[v & 63];
        p += 3;
        q += 4;
    }

    size_t rem = n - (size_t)(p - src);
    if (rem == 1) {
        uint32_t v = (uint32_t)p[0] << 16;
        q[0] = b64_tbl[(v >> 18) & 63];
        q[1] = b64_tbl[(v >> 12) & 63];
        q[2] = '=';
        q[3] = '=';
    } else if (rem == 2) {
        uint32_t v = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8);
        q[0] = b64_tbl[(v >> 18) & 63];
        q[1] = b64_tbl[(v >> 12) & 63];
        q[2] = b64_tbl[(v >> 6) & 63];
        q[3] = '=';
    }
}

char *base64_encode(const unsigned char *src, size_t n) {
    size_t out_len = base64_encoded_len(n);
    char *out = xmalloc(out_len + 1);
    base64_encode_to(out, src, n);
    out[out_len] = '\0';
    return out;
}
