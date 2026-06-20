/* test_base64.c - base64_encode against RFC 4648 vectors and binary inputs. */

#include "base64.h"
#include "test_util.h"

#include <stdlib.h>
#include <string.h>

static void test_base64_rfc4648(void) {
    /* RFC 4648 test vectors. */
    static const struct {
        const char *in, *out;
    } cases[] = {
        {"", ""},
        {"f", "Zg=="},
        {"fo", "Zm8="},
        {"foo", "Zm9v"},
        {"foob", "Zm9vYg=="},
        {"fooba", "Zm9vYmE="},
        {"foobar", "Zm9vYmFy"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *got = base64_encode((const unsigned char *)cases[i].in, strlen(cases[i].in));
        TEST_ASSERT(strcmp(got, cases[i].out) == 0);
        free(got);
    }
}

static void test_base64_binary_vectors(void) {
    static const struct {
        unsigned char bytes[5];
        size_t n;
        const char *out;
    } cases[] = {
        {{0xff}, 1, "/w=="},
        {{0xff, 0xee}, 2, "/+4="},
        {{0xff, 0xee, 0xdd}, 3, "/+7d"},
        {{0x00, 0x01, 0x02, 0x03, 0x04}, 5, "AAECAwQ="},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *got = base64_encode(cases[i].bytes, cases[i].n);
        TEST_ASSERT(strcmp(got, cases[i].out) == 0);
        free(got);
    }
}

int main(void) {
    test_base64_rfc4648();
    test_base64_binary_vectors();
    return TEST_REPORT("base64");
}
