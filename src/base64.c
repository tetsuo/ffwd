#include "base64.h"
#include "server_util.h"

char *base64_encode(const unsigned char *src, size_t n) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t out_len = ((n + 2) / 3) * 4;
    char *out = xmalloc(out_len + 1);
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = (unsigned)src[i] << 16;
        int remain = (int)(n - i);
        if (remain > 1)
            v |= (unsigned)src[i + 1] << 8;
        if (remain > 2)
            v |= (unsigned)src[i + 2];
        out[o++] = tbl[(v >> 18) & 63];
        out[o++] = tbl[(v >> 12) & 63];
        out[o++] = remain > 1 ? tbl[(v >> 6) & 63] : '=';
        out[o++] = remain > 2 ? tbl[v & 63] : '=';
    }
    out[o] = '\0';
    return out;
}
