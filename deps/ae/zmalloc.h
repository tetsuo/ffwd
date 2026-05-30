/* zmalloc.h - simplified memory allocation wrapper for ae library
 * This is a minimal version for standalone use outside Redis.
 */
#ifndef __ZMALLOC_H
#define __ZMALLOC_H

#include <stdlib.h>

/* Simple wrappers around standard malloc/free */
#define zmalloc malloc
#define zfree free
#define zrealloc realloc
#define zcalloc calloc

#endif
