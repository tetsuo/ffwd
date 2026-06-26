#ifndef FFWD_BASE64_H
#define FFWD_BASE64_H

#include <stddef.h>

/* Standard base64 (with '=' padding). */

/* Number of base64 characters for n input bytes, excluding any NUL.
 * Aborts (die-on-OOM) if n is so large the length would overflow. */
size_t base64_encoded_len(size_t n);

/* Encode n bytes of src as base64 into dst, writing exactly base64_encoded_len(n)
 * bytes and no NUL terminator; dst must have room for that many bytes. */
void base64_encode_to(char *dst, const unsigned char *src, size_t n);

/* Convenience wrapper: returns a NUL-terminated heap string the caller frees;
 * die-on-OOM, so it never returns NULL. */
char *base64_encode(const unsigned char *src, size_t n);

#endif /* FFWD_BASE64_H */
