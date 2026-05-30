/* util.h - minimal utilities needed by anet */
#ifndef __UTIL_H
#define __UTIL_H

#include <string.h>

/* Simple error buffer size */
#define ANET_ERR_LEN 256

/* Safe string copy - simplified version */
static inline size_t redis_strlcpy(char *dst, const char *src, size_t dsize) {
    size_t len = strlen(src);
    if (dsize > 0) {
        size_t copy_len = (len >= dsize) ? dsize - 1 : len;
        memcpy(dst, src, copy_len);
        dst[copy_len] = '\0';
    }
    return len;
}

#endif
