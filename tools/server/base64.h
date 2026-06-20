#ifndef FFWD_BASE64_H
#define FFWD_BASE64_H

#include <stddef.h>

/* Standard base64 (with '=' padding) of n bytes. Returns a NUL-terminated heap
 * string the caller frees; die-on-OOM, so it never returns NULL. */
char *base64_encode(const unsigned char *src, size_t n);

#endif /* FFWD_BASE64_H */
