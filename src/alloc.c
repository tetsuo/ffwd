#include "alloc.h"

#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

/* Sequence-capacity growth policy for the forward-pass workspaces: for new
 * growth, never allocate fewer than the minimum, and round up to the
 * granularity so small length changes do not trigger a realloc on every call.
 * Existing larger capacity is preserved. */
#define EMBED_MIN_WORKSPACE_SEQ_CAP     16
#define EMBED_WORKSPACE_SEQ_GRANULARITY 16

int mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a)
        return -1;
    *out = a * b;
    return 0;
}

int add_size(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a)
        return -1;
    *out = a + b;
    return 0;
}

int grow_cap(int current, int needed, int *out) {
    if (current < 0 || needed <= 0)
        return -1;

    int cap = needed;
    if (cap < EMBED_MIN_WORKSPACE_SEQ_CAP)
        cap = EMBED_MIN_WORKSPACE_SEQ_CAP;

    int rem = cap % EMBED_WORKSPACE_SEQ_GRANULARITY;
    if (rem != 0) {
        int pad = EMBED_WORKSPACE_SEQ_GRANULARITY - rem;
        if (cap > INT_MAX - pad)
            return -1;
        cap += pad;
    }

    if (cap < current)
        cap = current;

    *out = cap;
    return 0;
}

int realloc_floats(float **ptr, size_t count) {
    size_t bytes;
    if (mul_size(count, sizeof(float), &bytes) != 0)
        return -1;

    void *p = realloc(*ptr, bytes);
    if (!p && bytes != 0)
        return -1;
    *ptr = (float *)p;
    return 0;
}

float *malloc_floats(size_t count) {
    size_t bytes;
    if (mul_size(count, sizeof(float), &bytes) != 0)
        return NULL;
    return (float *)malloc(bytes);
}

int realloc_floats_2d(float **ptr, int rows, int cols) {
    if (rows < 0 || cols < 0)
        return -1;

    size_t count;
    if (mul_size((size_t)rows, (size_t)cols, &count) != 0)
        return -1;
    return realloc_floats(ptr, count);
}

int realloc_ints(int **ptr, size_t count) {
    size_t bytes;
    if (mul_size(count, sizeof(int), &bytes) != 0)
        return -1;

    void *p = realloc(*ptr, bytes);
    if (!p && bytes != 0)
        return -1;
    *ptr = (int *)p;
    return 0;
}
