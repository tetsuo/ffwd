/* redisassert.h - simplified assert for standalone ae library */
#ifndef __REDIS_ASSERT_H__
#define __REDIS_ASSERT_H__

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* config.h provides redis_unreachable for standalone AE builds. */

/* Panic function for error handling */
#define panic(...) do { \
    fprintf(stderr, "PANIC: " __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    abort(); \
} while(0)

#endif
