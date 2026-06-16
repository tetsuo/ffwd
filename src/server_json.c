#include "server_internal.h"
#include "server_util.h"

#include <cjson/cJSON.h>
#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Request validation and error bodies
 *
 * Two JSON shapes are produced here: the OpenAI-style {"error":{message,type}}
 * body (append_json_error / json_error_body / job_set_error) for transport and
 * auth failures, and the Perplexity-style {"detail":[...]} 422 body built from
 * ve_add entries for field validation. The parse_* / *_from_root helpers read
 * and validate the request object; append_json_string escapes a value for
 * inclusion in a response. */

static void append_json_error(sbuf *b, const char *message, const char *type) {
    sbuf_puts(b, "{\"error\":{\"message\":");
    append_json_string(b, message ? message : "");
    sbuf_puts(b, ",\"type\":");
    append_json_string(b, type ? type : "server_error");
    sbuf_puts(b, "}}");
}

void append_json_string(sbuf *b, const char *s) {
    sbuf_putc(b, '"');
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
        case '"':
            sbuf_puts(b, "\\\"");
            break;
        case '\\':
            sbuf_puts(b, "\\\\");
            break;
        case '\b':
            sbuf_puts(b, "\\b");
            break;
        case '\f':
            sbuf_puts(b, "\\f");
            break;
        case '\n':
            sbuf_puts(b, "\\n");
            break;
        case '\r':
            sbuf_puts(b, "\\r");
            break;
        case '\t':
            sbuf_puts(b, "\\t");
            break;
        default:
            if (*p < 0x20)
                sbuf_printf(b, "\\u%04x", (unsigned)*p);
            else
                sbuf_putc(b, (char)*p);
        }
    }
    sbuf_putc(b, '"');
}

char *json_error_body(const char *message, const char *type, size_t *len) {
    sbuf b = {0};
    append_json_error(&b, message, type);
    *len = b.len;
    return b.ptr;
}

void job_set_error(job *j, int status, const char *message, const char *type) {
    free(j->response);
    j->status = status;
    j->content_type = "application/json";
    j->response = json_error_body(message, type, &j->response_len);
}

bool ve_add(cJSON *detail, const char *loc_json, const char *msg, const char *type) {
    cJSON *obj = cJSON_CreateObject();
    cJSON *loc = cJSON_Parse(loc_json);
    if (!obj || !loc) {
        cJSON_Delete(obj);
        cJSON_Delete(loc);
        return false;
    }
    cJSON_AddItemToObject(obj, "loc", loc);
    cJSON_AddStringToObject(obj, "msg", msg ? msg : "");
    cJSON_AddStringToObject(obj, "type", type ? type : "");
    cJSON_AddItemToArray(detail, obj);
    return true;
}

void job_set_422(job *j, cJSON *detail) {
    cJSON *root = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "detail", cJSON_Duplicate(detail, 1));
    char *s = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!s) {
        job_set_error(j, 500, "failed to render validation error", "server_error");
        return;
    }
    free(j->response);
    j->status = 422;
    j->content_type = "application/json";
    j->response = xstrdup(s);
    j->response_len = strlen(j->response);
    cJSON_free(s);
}

cJSON *parse_json_body(job *j, cJSON *detail) {
    if (!j->body || j->body_len == 0) {
        ve_add(detail, "[\"body\"]", "body required", "missing");
        return NULL;
    }
    const char *end = NULL;
    cJSON *root = cJSON_ParseWithLengthOpts(j->body, j->body_len, &end, false);
    if (!root) {
        ve_add(detail, "[\"body\"]", "invalid JSON", "json_invalid");
        return NULL;
    }
    const char *body_end = j->body + j->body_len;
    while (end && end < body_end && isspace((unsigned char)*end))
        end++;
    if (!end || end != body_end) {
        ve_add(detail, "[\"body\"]", "invalid JSON", "json_invalid");
        cJSON_Delete(root);
        return NULL;
    }
    if (!cJSON_IsObject(root)) {
        ve_add(detail, "[\"body\"]", "body must be a JSON object", "type_error.object");
        cJSON_Delete(root);
        return NULL;
    }
    return root;
}

bool cjson_is_integer(cJSON *item) {
    if (!cJSON_IsNumber(item))
        return false;
    double d = item->valuedouble;
    int i = item->valueint;
    return d >= (double)INT_MIN && d <= (double)INT_MAX && d == (double)i;
}

const char *encoding_from_root(cJSON *root, cJSON *detail, embedding_api_t api) {
    /* Default and accepted encodings follow the model's API family: Perplexity
     * defaults to base64_int8 and accepts {base64_int8, base64_binary, float};
     * OpenAI/DashScope (Qwen3) defaults to float and accepts {float, base64}. */
    const char *dflt = api == EMBED_API_OPENAI ? "float" : "base64_int8";
    cJSON *encoding = cJSON_GetObjectItemCaseSensitive(root, "encoding_format");
    if (!encoding)
        return dflt;
    if (!cJSON_IsString(encoding) || !encoding->valuestring) {
        ve_add(detail, "[\"body\",\"encoding_format\"]", "encoding_format must be a string",
               "type_error.string");
        return dflt;
    }
    const char *v = encoding->valuestring;
    int ok = api == EMBED_API_OPENAI ? (!strcmp(v, "float") || !strcmp(v, "base64"))
                                     : (!strcmp(v, "base64_int8") || !strcmp(v, "base64_binary") ||
                                        !strcmp(v, "float"));
    if (!ok) {
        ve_add(detail, "[\"body\",\"encoding_format\"]",
               api == EMBED_API_OPENAI
                   ? "value is not a valid enum member; permitted: 'float', 'base64'"
                   : "value is not a valid enum member; permitted: "
                     "'base64_int8', 'base64_binary', 'float'",
               "enum");
        return dflt;
    }
    return v;
}

/* Parse the `text_type` hint. Returns 1 when the model's query instruction
 * should be prepended, 0 otherwise (document / absent). Models without a
 * published query instruction reject the field. */
int text_type_is_query(cJSON *root, cJSON *detail, const char *query_instruct) {
    cJSON *tt = cJSON_GetObjectItemCaseSensitive(root, "text_type");
    if (!tt)
        return 0;
    if (!query_instruct) {
        ve_add(detail, "[\"body\",\"text_type\"]",
               "text_type is only supported for instruction embedding models", "extra_fields");
        return 0;
    }
    if (!cJSON_IsString(tt) || !tt->valuestring) {
        ve_add(detail, "[\"body\",\"text_type\"]", "text_type must be a string",
               "type_error.string");
        return 0;
    }
    const char *v = tt->valuestring;
    if (!strcmp(v, "query"))
        return 1;
    if (!strcmp(v, "document"))
        return 0;
    ve_add(detail, "[\"body\",\"text_type\"]",
           "value is not a valid enum member; permitted: 'query', 'document'", "enum");
    return 0;
}

int dimensions_from_root(
    cJSON *root, cJSON *detail, int min_dim, int max_dim, const char *encoding) {
    int dims = max_dim;
    cJSON *dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    if (dimensions) {
        if (!cjson_is_integer(dimensions)) {
            ve_add(detail, "[\"body\",\"dimensions\"]", "dimensions must be an integer",
                   "type_error.integer");
        } else {
            dims = dimensions->valueint;
            if (dims < min_dim || dims > max_dim) {
                char msg[96];
                snprintf(msg, sizeof(msg), "dimensions must be between %d and %d", min_dim,
                         max_dim);
                ve_add(detail, "[\"body\",\"dimensions\"]", msg, "value_error.range");
            }
        }
    }
    if (!strcmp(encoding, "base64_binary") && dims % 8 != 0)
        ve_add(detail, "[\"body\",\"dimensions\"]",
               "dimensions must be divisible by 8 for base64_binary", "value_error.divisible");
    return dims;
}
