#ifndef FFWD_ALLOC_H
#define FFWD_ALLOC_H

#include <stddef.h>

/* Checked size arithmetic and typed allocation helpers.
 * Return -1 or NULL on overflow/allocation failure.
 * realloc_* leave the original pointer unchanged on failure. */

/* out = a * b, or -1 if the product overflows size_t. */
int mul_size(size_t a, size_t b, size_t *out);

/* out = a + b, or -1 if the sum overflows size_t. */
int add_size(size_t a, size_t b, size_t *out);

/* Grow sequence capacity without shrinking. */
int grow_cap(int current, int needed, int *out);

/* realloc *ptr to hold count floats; *ptr unchanged on failure. */
int realloc_floats(float **ptr, size_t count);

/* malloc count floats, NULL on overflow/failure. */
float *malloc_floats(size_t count);

/* realloc *ptr to hold rows*cols floats; -1 on negative dims/overflow/failure. */
int realloc_floats_2d(float **ptr, int rows, int cols);

/* realloc *ptr to hold count ints; *ptr unchanged on failure. */
int realloc_ints(int **ptr, size_t count);

#endif /* FFWD_ALLOC_H */
