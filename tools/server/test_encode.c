/* test_encode.c - embedding quantization and the output encodings
 * (base64_int8, base64_binary, base64 float32) plus float rendering. */

#include "base64.h"
#include "server_internal.h"
#include "test_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void test_quantize_int8_tanh(void) {
    TEST_ASSERT(quantize_int8_tanh(0.0f) == 0);
    /* tanh saturates to +-1, scaled by 127. */
    TEST_ASSERT(quantize_int8_tanh(100.0f) == 127);
    TEST_ASSERT(quantize_int8_tanh(-100.0f) == -127);
    /* Symmetry. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == -quantize_int8_tanh(-0.5f));
    /* round(tanh(0.5) * 127) = round(58.69) = 59. */
    TEST_ASSERT(quantize_int8_tanh(0.5f) == 59);
}

/* Assert that rendering emb under encoding puts the base64 of expect in the
 * embedding field of the response object. */
static void assert_base64_field(const float *emb,
                                int dims,
                                const char *encoding,
                                embedding_api_t api,
                                const unsigned char *expect,
                                size_t expect_n) {
    char *b64 = base64_encode(expect, expect_n);
    char field[96];
    snprintf(field, sizeof(field), "\"embedding\":\"%s\"", b64);
    sbuf b = {0};
    append_embedding_value(&b, 0, emb, dims, encoding, api);
    TEST_ASSERT(b.ptr && strstr(b.ptr, field) != NULL);
    sbuf_free(&b);
    free(b64);
}

static void test_append_embedding_value_int8(void) {
    float emb[4] = {0.0f, 0.5f, -0.5f, 100.0f};
    signed char expect[4] = {0, 59, -59, 127};
    assert_base64_field(emb, 4, "base64_int8", FFWD_API_PERPLEXITY, (const unsigned char *)expect, 4);
}

static void test_append_embedding_value_binary(void) {
    /* Sign bits pack LSB-first: bit i of byte i/8 is 1 when emb[i] >= 0. */
    float emb[8] = {1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
    unsigned char expect = 0x85; /* bits 0, 2, 7 set */
    assert_base64_field(emb, 8, "base64_binary", FFWD_API_PERPLEXITY, &expect, 1);

    /* Zero counts as non-negative (sign bit set). */
    float zeros[8] = {0};
    unsigned char all = 0xff;
    assert_base64_field(zeros, 8, "base64_binary", FFWD_API_PERPLEXITY, &all, 1);
}

static void test_append_embedding_value_base64(void) {
    /* OpenAI/DashScope "base64" is base64 of the raw little-endian float32
     * vector - lossless, unlike the int8 formats. */
    float emb[4] = {0.0f, 0.5f, -0.5f, 100.0f};
    /* Reference bytes via memcpy: a float->byte reinterpret cast confuses the
     * static analyzer's uninitialized-read model. */
    unsigned char bytes[sizeof emb];
    memcpy(bytes, emb, sizeof emb);
    assert_base64_field(emb, 4, "base64", FFWD_API_OPENAI, bytes, sizeof emb);
}

static void test_append_embedding_value_float(void) {
    float emb[1] = {0.5f};

    /* OpenAI-compatible models render the true float32 value. */
    sbuf b = {0};
    append_embedding_value(&b, 0, emb, 1, "float", FFWD_API_OPENAI);
    TEST_ASSERT(strstr(b.ptr, "\"embedding\":[0.5]") != NULL);
    sbuf_free(&b);

    /* Perplexity-compatible models render the int8-decoded view. */
    sbuf b2 = {0};
    append_embedding_value(&b2, 0, emb, 1, "float", FFWD_API_PERPLEXITY);
    TEST_ASSERT(strstr(b2.ptr, "0.4609375") != NULL);
    sbuf_free(&b2);
}

static void test_job_embedding_render_payload(void) {
    model_info info;
    memset(&info, 0, sizeof(info));
    info.id = "m";
    info.dim = 2;
    info.api = FFWD_API_OPENAI;

    loaded_model model;
    memset(&model, 0, sizeof(model));
    model.info = &info;

    job j;
    memset(&j, 0, sizeof(j));
    embedding_request r;
    memset(&r, 0, sizeof(r));
    r.j = &j;
    r.model = &model;
    r.dims = 2;
    r.encoding = "float";
    r.n_inputs = 2;
    r.total_tokens = 7;

    float embs[4] = {0.5f, 1.0f, -0.25f, 0.0f};
    job_set_embedding_render(&j, &r, embs);
    TEST_ASSERT(j.render_kind == 1);
    TEST_ASSERT(j.embedding_render.embs != embs);
    TEST_ASSERT(render_job_response(&j) == 0);
    TEST_ASSERT(j.status == 200);
    TEST_ASSERT(j.response != NULL);
    TEST_ASSERT(strstr(j.response, "\"model\":\"m\"") != NULL);
    TEST_ASSERT(strstr(j.response, "\"prompt_tokens\":7") != NULL);
    TEST_ASSERT(strstr(j.response, "\"embedding\":[0.5,1]") != NULL);
    TEST_ASSERT(strstr(j.response, "\"embedding\":[-0.25,0]") != NULL);

    free(j.response);
    j.response = NULL;
    job_render_free(&j);
}

int main(void) {
    test_quantize_int8_tanh();
    test_append_embedding_value_int8();
    test_append_embedding_value_binary();
    test_append_embedding_value_base64();
    test_append_embedding_value_float();
    test_job_embedding_render_payload();
    return TEST_REPORT("encode");
}
