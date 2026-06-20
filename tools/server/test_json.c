/* test_json.c - request validation, JSON error bodies, and 422 detail
 * handling (encoding_format, text_type, error escaping, job_set_422). */

#include "server_internal.h"
#include "test_util.h"
#include "util.h" /* xstrdup */

#include <stdlib.h>
#include <string.h>

static const char *TEST_QWEN3_QUERY_INSTRUCT =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

static void test_json_error_body_escaping(void) {
    size_t len = 0;
    char *body = json_error_body("line\nquote\" tab\t ctrl\001 slash\\", "bad\ntype", &len);
    TEST_ASSERT(body != NULL);
    if (!body)
        return;
    TEST_ASSERT(len == strlen(body));
    TEST_ASSERT(strchr(body, '\n') == NULL);
    TEST_ASSERT(strchr(body, '\t') == NULL);
    cJSON *root = cJSON_Parse(body);
    TEST_ASSERT(root != NULL);
    if (!root) {
        free(body);
        return;
    }
    cJSON *err = cJSON_GetObjectItemCaseSensitive(root, "error");
    cJSON *msg = err ? cJSON_GetObjectItemCaseSensitive(err, "message") : NULL;
    cJSON *type = err ? cJSON_GetObjectItemCaseSensitive(err, "type") : NULL;
    if (!msg || !type || !cJSON_IsString(msg) || !cJSON_IsString(type)) {
        TEST_ASSERT(0 && "error body fields must be strings");
        cJSON_Delete(root);
        free(body);
        return;
    }
    TEST_ASSERT(strcmp(msg->valuestring, "line\nquote\" tab\t ctrl\001 slash\\") == 0);
    TEST_ASSERT(strcmp(type->valuestring, "bad\ntype") == 0);
    cJSON_Delete(root);
    free(body);
}

static void test_job_set_422_replaces_response(void) {
    job j;
    memset(&j, 0, sizeof(j));
    j.response = xstrdup("old response");
    j.response_len = strlen(j.response);
    j.status = 500;

    cJSON *detail = cJSON_CreateArray();
    ve_add(detail, "[\"body\",\"field\"]", "bad value", "value_error");
    job_set_422(&j, detail);
    TEST_ASSERT(j.status == 422);
    TEST_ASSERT(j.response != NULL);
    if (j.response) {
        TEST_ASSERT(j.response_len == strlen(j.response));
        TEST_ASSERT(strstr(j.response, "bad value") != NULL);
    }

    cJSON_Delete(detail);
    free(j.response);
}

static void test_encoding_from_root_family(void) {
    cJSON *detail = cJSON_CreateArray();

    /* Default encoding follows the explicit serving API. */
    cJSON *empty = cJSON_CreateObject();
    TEST_ASSERT(strcmp(encoding_from_root(empty, detail, FFWD_API_OPENAI), "float") == 0);
    TEST_ASSERT(strcmp(encoding_from_root(empty, detail, FFWD_API_PERPLEXITY), "base64_int8") ==
                0);
    cJSON_Delete(empty);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 0);

    /* OpenAI rejects int8 formats; Perplexity rejects the OpenAI base64. */
    cJSON *o = cJSON_CreateObject();
    cJSON_AddStringToObject(o, "encoding_format", "base64_int8");
    encoding_from_root(o, detail, FFWD_API_OPENAI);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 1);
    cJSON_Delete(o);

    cJSON *p = cJSON_CreateObject();
    cJSON_AddStringToObject(p, "encoding_format", "base64");
    encoding_from_root(p, detail, FFWD_API_PERPLEXITY);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 2);
    cJSON_Delete(p);

    /* An unknown format is rejected. */
    cJSON *bad = cJSON_CreateObject();
    cJSON_AddStringToObject(bad, "encoding_format", "base64_fp16");
    encoding_from_root(bad, detail, FFWD_API_OPENAI);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 3);
    cJSON_Delete(bad);

    cJSON_Delete(detail);
}

static void test_text_type(void) {
    cJSON *detail = cJSON_CreateArray();

    /* Absent: no-op for both model types, no error. */
    cJSON *empty = cJSON_CreateObject();
    TEST_ASSERT(text_type_is_query(empty, detail, TEST_QWEN3_QUERY_INSTRUCT) == 0);
    TEST_ASSERT(text_type_is_query(empty, detail, NULL) == 0);
    cJSON_Delete(empty);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 0);

    /* Qwen3: query -> 1, document -> 0, both accepted. */
    cJSON *q = cJSON_CreateObject();
    cJSON_AddStringToObject(q, "text_type", "query");
    TEST_ASSERT(text_type_is_query(q, detail, TEST_QWEN3_QUERY_INSTRUCT) == 1);
    cJSON_Delete(q);
    cJSON *d = cJSON_CreateObject();
    cJSON_AddStringToObject(d, "text_type", "document");
    TEST_ASSERT(text_type_is_query(d, detail, TEST_QWEN3_QUERY_INSTRUCT) == 0);
    cJSON_Delete(d);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 0);

    /* Invalid enum value on Qwen3 is reported. */
    cJSON *bad = cJSON_CreateObject();
    cJSON_AddStringToObject(bad, "text_type", "passage");
    TEST_ASSERT(text_type_is_query(bad, detail, TEST_QWEN3_QUERY_INSTRUCT) == 0);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 1);
    cJSON_Delete(bad);

    /* text_type on a model with no query instruction is rejected. */
    cJSON *p = cJSON_CreateObject();
    cJSON_AddStringToObject(p, "text_type", "query");
    TEST_ASSERT(text_type_is_query(p, detail, NULL) == 0);
    TEST_ASSERT(cJSON_GetArraySize(detail) == 2);
    cJSON_Delete(p);

    cJSON_Delete(detail);
}

int main(void) {
    test_json_error_body_escaping();
    test_job_set_422_replaces_response();
    test_encoding_from_root_family();
    test_text_type();
    return TEST_REPORT("json");
}
