/* tests/test_server.c - unit tests for embed_server.c static helpers.
 * ds4-style: includes the whole server translation unit with the real main
 * compiled out, so statics are directly testable. Runs via `make test`. */

#define EMBED_SERVER_TEST
#include "../embed_server.c"

static int test_failures = 0;

#define TEST_ASSERT(cond) do {                                        \
    if (!(cond)) {                                                    \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        test_failures++;                                              \
    }                                                                 \
} while (0)

static void test_base64_rfc4648(void)
{
    /* RFC 4648 test vectors. */
    static const struct { const char *in, *out; } cases[] = {
        {"", ""}, {"f", "Zg=="}, {"fo", "Zm8="}, {"foo", "Zm9v"},
        {"foob", "Zm9vYg=="}, {"fooba", "Zm9vYmE="}, {"foobar", "Zm9vYmFy"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        char *got = base64_encode((const unsigned char *)cases[i].in,
                                  strlen(cases[i].in));
        TEST_ASSERT(strcmp(got, cases[i].out) == 0);
        free(got);
    }
}

static void test_quantize_int8_tanh(void)
{
    TEST_ASSERT(quantize_int8_tanh(0.0f) == 0);
    /* tanh saturates to +-1, scaled by 127. */
    TEST_ASSERT(quantize_int8_tanh(100.0f) == 127);
    TEST_ASSERT(quantize_int8_tanh(-100.0f) == -127);
    /* Symmetry. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == -quantize_int8_tanh(-0.5f));
    /* round(tanh(0.5) * 127) = round(58.69) = 59. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == 59);
}

static void test_encode_embedding_int8(void)
{
    float emb[4] = {0.0f, 0.5f, -0.5f, 100.0f};
    signed char expect[4] = {0, 59, -59, 127};
    char *got = encode_embedding(emb, 4, "base64_int8");
    char *want = base64_encode((const unsigned char *)expect, 4);
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);
}

static void test_encode_embedding_binary(void)
{
    /* Sign bits pack LSB-first: bit i of byte i/8 is 1 when emb[i] >= 0. */
    float emb[8] = {1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
    unsigned char expect = 0x85; /* bits 0, 2, 7 set */
    char *got = encode_embedding(emb, 8, "base64_binary");
    char *want = base64_encode(&expect, 1);
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);

    /* Zero counts as non-negative (sign bit set). */
    float zeros[8] = {0};
    unsigned char all = 0xff;
    got = encode_embedding(zeros, 8, "base64_binary");
    want = base64_encode(&all, 1);
    TEST_ASSERT(strcmp(got, want) == 0);
    free(got);
    free(want);
}

int main(void)
{
    test_base64_rfc4648();
    test_quantize_int8_tanh();
    test_encode_embedding_int8();
    test_encode_embedding_binary();
    if (test_failures) {
        fprintf(stderr, "server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ok: server unit tests passed");
    return 0;
}
