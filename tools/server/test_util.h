#ifndef FFWD_SERVER_TEST_UTIL_H
#define FFWD_SERVER_TEST_UTIL_H

/* Shared harness for the ffwd-server unit tests. TEST_ASSERT records failures
 * (rather than aborting) so a single run reports every failure, not just the
 * first. Each test file includes this, runs its checks, and ends main with
 * return TEST_REPORT("label");, which prints a summary and yields the exit
 * code (0 when clean, 1 otherwise). */

#include <stdio.h>

static int test_failures = 0;

#define TEST_ASSERT(cond)                                                   \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            test_failures++;                                                \
        }                                                                   \
    } while (0)

#define TEST_REPORT(label)                                                               \
    (test_failures ? (fprintf(stderr, "%s: %d failure(s)\n", (label), test_failures), 1) \
                   : (printf("ok: %s\n", (label)), 0))

#endif /* FFWD_SERVER_TEST_UTIL_H */
