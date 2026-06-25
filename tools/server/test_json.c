/* test_json.c - request validation, JSON error bodies, and 422 detail
 * handling (encoding_format, text_type, error escaping, job_set_422). */

#include "server_internal.h"
#include "test_util.h"
#include "util.h" /* xstrdup */

#include <stdlib.h>
#include <string.h>

static const char *TEST_QWEN3_QUERY_INSTRUCT =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

/* Parse a JSON literal into an immutable document so its root can be passed to
 * the read-only validators (encoding_from_root, text_type_is_query). */
static yyjson_doc *parse_obj(const char *json) { return yyjson_read(json, strlen(json), 0); }

static void test_json_error_body_escaping(void) {
    size_t len = 0;
    char *body = json_error_body("line\nquote\" tab\t ctrl\001 slash\\", "bad\ntype", &len);
    TEST_ASSERT(body != NULL);
    if (!body)
        return;
    TEST_ASSERT(len == strlen(body));
    TEST_ASSERT(strchr(body, '\n') == NULL);
    TEST_ASSERT(strchr(body, '\t') == NULL);
    yyjson_doc *doc = yyjson_read(body, strlen(body), 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    TEST_ASSERT(root != NULL);
    if (!root) {
        yyjson_doc_free(doc);
        free(body);
        return;
    }
    yyjson_val *err = yyjson_obj_get(root, "error");
    yyjson_val *msg = err ? yyjson_obj_get(err, "message") : NULL;
    yyjson_val *type = err ? yyjson_obj_get(err, "type") : NULL;
    if (!msg || !type || !yyjson_is_str(msg) || !yyjson_is_str(type)) {
        TEST_ASSERT(0 && "error body fields must be strings");
        yyjson_doc_free(doc);
        free(body);
        return;
    }
    TEST_ASSERT(strcmp(yyjson_get_str(msg), "line\nquote\" tab\t ctrl\001 slash\\") == 0);
    TEST_ASSERT(strcmp(yyjson_get_str(type), "bad\ntype") == 0);
    yyjson_doc_free(doc);
    free(body);
}

static void test_job_set_422_replaces_response(void) {
    job j;
    memset(&j, 0, sizeof(j));
    j.response = xstrdup("old response");
    j.response_len = strlen(j.response);
    j.status = 500;

    yyjson_mut_doc *detail = ve_new();
    ve_add(detail, "[\"body\",\"field\"]", "bad value", "value_error");
    job_set_422(&j, detail);
    TEST_ASSERT(j.status == 422);
    TEST_ASSERT(j.response != NULL);
    if (j.response) {
        TEST_ASSERT(j.response_len == strlen(j.response));
        TEST_ASSERT(strstr(j.response, "bad value") != NULL);
    }

    yyjson_mut_doc_free(detail);
    free(j.response);
}

static void test_encoding_from_root_family(void) {
    yyjson_mut_doc *detail = ve_new();

    /* Default encoding follows the explicit serving API. */
    yyjson_doc *empty = parse_obj("{}");
    yyjson_val *empty_root = yyjson_doc_get_root(empty);
    TEST_ASSERT(strcmp(encoding_from_root(empty_root, detail, FFWD_API_OPENAI), "float") == 0);
    TEST_ASSERT(strcmp(encoding_from_root(empty_root, detail, FFWD_API_PERPLEXITY), "base64_int8") ==
                0);
    yyjson_doc_free(empty);
    TEST_ASSERT(ve_count(detail) == 0);

    /* OpenAI rejects int8 formats; Perplexity rejects the OpenAI base64. */
    yyjson_doc *o = parse_obj("{\"encoding_format\":\"base64_int8\"}");
    encoding_from_root(yyjson_doc_get_root(o), detail, FFWD_API_OPENAI);
    TEST_ASSERT(ve_count(detail) == 1);
    yyjson_doc_free(o);

    yyjson_doc *p = parse_obj("{\"encoding_format\":\"base64\"}");
    encoding_from_root(yyjson_doc_get_root(p), detail, FFWD_API_PERPLEXITY);
    TEST_ASSERT(ve_count(detail) == 2);
    yyjson_doc_free(p);

    /* An unknown format is rejected. */
    yyjson_doc *bad = parse_obj("{\"encoding_format\":\"base64_fp16\"}");
    encoding_from_root(yyjson_doc_get_root(bad), detail, FFWD_API_OPENAI);
    TEST_ASSERT(ve_count(detail) == 3);
    yyjson_doc_free(bad);

    yyjson_mut_doc_free(detail);
}

static void test_text_type(void) {
    yyjson_mut_doc *detail = ve_new();

    /* Absent: no-op for both model types, no error. */
    yyjson_doc *empty = parse_obj("{}");
    yyjson_val *empty_root = yyjson_doc_get_root(empty);
    TEST_ASSERT(text_type_is_query(empty_root, detail, TEST_QWEN3_QUERY_INSTRUCT) == 0);
    TEST_ASSERT(text_type_is_query(empty_root, detail, NULL) == 0);
    yyjson_doc_free(empty);
    TEST_ASSERT(ve_count(detail) == 0);

    /* Qwen3: query -> 1, document -> 0, both accepted. */
    yyjson_doc *q = parse_obj("{\"text_type\":\"query\"}");
    TEST_ASSERT(text_type_is_query(yyjson_doc_get_root(q), detail, TEST_QWEN3_QUERY_INSTRUCT) == 1);
    yyjson_doc_free(q);
    yyjson_doc *d = parse_obj("{\"text_type\":\"document\"}");
    TEST_ASSERT(text_type_is_query(yyjson_doc_get_root(d), detail, TEST_QWEN3_QUERY_INSTRUCT) == 0);
    yyjson_doc_free(d);
    TEST_ASSERT(ve_count(detail) == 0);

    /* Invalid enum value on Qwen3 is reported. */
    yyjson_doc *bad = parse_obj("{\"text_type\":\"passage\"}");
    TEST_ASSERT(text_type_is_query(yyjson_doc_get_root(bad), detail, TEST_QWEN3_QUERY_INSTRUCT) == 0);
    TEST_ASSERT(ve_count(detail) == 1);
    yyjson_doc_free(bad);

    /* text_type on a model with no query instruction is rejected. */
    yyjson_doc *p = parse_obj("{\"text_type\":\"query\"}");
    TEST_ASSERT(text_type_is_query(yyjson_doc_get_root(p), detail, NULL) == 0);
    TEST_ASSERT(ve_count(detail) == 2);
    yyjson_doc_free(p);

    yyjson_mut_doc_free(detail);
}

int main(void) {
    test_json_error_body_escaping();
    test_job_set_422_replaces_response();
    test_encoding_from_root_family();
    test_text_type();
    return TEST_REPORT("json");
}
