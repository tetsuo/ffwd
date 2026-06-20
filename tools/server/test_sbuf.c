/* test_sbuf.c - growable string buffer growth, formatting, and clearing. */

#include "sbuf.h"
#include "test_util.h"

#include <string.h>

static void test_sbuf_growth_and_formatting(void) {
    sbuf b = {0};
    sbuf_puts(&b, "ab");
    sbuf_putc(&b, 'c');
    TEST_ASSERT(b.len == 3);
    TEST_ASSERT(strcmp(b.ptr, "abc") == 0);

    size_t old_cap = b.cap;
    sbuf_printf(&b, "-%04d-%s", 7, "tail");
    TEST_ASSERT(strcmp(b.ptr, "abc-0007-tail") == 0);
    TEST_ASSERT(b.cap == old_cap);

    char big[600];
    memset(big, 'x', sizeof(big) - 1);
    big[sizeof(big) - 1] = '\0';
    sbuf_puts(&b, big);
    TEST_ASSERT(b.len == strlen("abc-0007-tail") + sizeof(big) - 1);
    TEST_ASSERT(b.ptr[b.len] == '\0');
    sbuf_clear(&b);
    TEST_ASSERT(b.len == 0);
    TEST_ASSERT(b.ptr && b.ptr[0] == '\0');
    sbuf_free(&b);
}

int main(void) {
    test_sbuf_growth_and_formatting();
    return TEST_REPORT("sbuf");
}
