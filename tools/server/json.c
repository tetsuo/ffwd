/* Request validation and error bodies.
 *
 * This code builds two JSON error shapes:
 *  - OpenAI-style {"error":{message,type}} for transport/auth failures, via
 *    append_json_error, json_error_body, and job_set_error.
 *  - Perplexity-style {"detail":[...]} 422 responses for field validation, built
 *    from ve_add entries.
 *
 * parse_* and *_from_root helpers read and validate the request object.
 * append_json_string escapes a value for use in a response.
 */

#include "server_internal.h"
#include "util.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

yyjson_mut_doc *ve_new(void) {
    yyjson_mut_doc *detail = yyjson_mut_doc_new(NULL);
    yyjson_mut_doc_set_root(detail, yyjson_mut_arr(detail));
    return detail;
}

bool ve_add(yyjson_mut_doc *detail, const char *loc_json, const char *msg, const char *type) {
    yyjson_mut_val *arr = yyjson_mut_doc_get_root(detail);
    yyjson_mut_val *obj = yyjson_mut_obj(detail);
    /* loc is a JSON array literal (mix of strings and integer indices) built by
     * the caller; parse it and deep-copy it into the detail document so it owns
     * the values once the temporary parse is freed. */
    yyjson_doc *loc_doc = yyjson_read(loc_json, strlen(loc_json), 0);
    yyjson_mut_val *loc = loc_doc ? yyjson_val_mut_copy(detail, yyjson_doc_get_root(loc_doc)) : NULL;
    yyjson_doc_free(loc_doc);
    if (!arr || !obj || !loc)
        return false;
    yyjson_mut_obj_add_val(detail, obj, "loc", loc);
    /* msg may point at a caller stack buffer, so copy the value strings. */
    yyjson_mut_obj_add_strcpy(detail, obj, "msg", msg ? msg : "");
    yyjson_mut_obj_add_strcpy(detail, obj, "type", type ? type : "");
    yyjson_mut_arr_add_val(arr, obj);
    return true;
}

size_t ve_count(yyjson_mut_doc *detail) {
    return yyjson_mut_arr_size(yyjson_mut_doc_get_root(detail));
}

void job_set_422(job *j, yyjson_mut_doc *detail) {
    /* Reparent the detail array under a fresh {"detail":[...]} root in the same
     * document, then serialize. The caller frees the document right after, so
     * mutating its root here is safe and avoids a deep copy. */
    yyjson_mut_val *arr = yyjson_mut_doc_get_root(detail);
    yyjson_mut_val *root = yyjson_mut_obj(detail);
    size_t len = 0;
    char *s = NULL;
    if (root && arr && yyjson_mut_obj_add_val(detail, root, "detail", arr)) {
        yyjson_mut_doc_set_root(detail, root);
        s = yyjson_mut_write(detail, 0, &len);
    }
    if (!s) {
        job_set_error(j, 500, "failed to render validation error", "server_error");
        return;
    }
    free(j->response);
    j->status = 422;
    j->content_type = "application/json";
    j->response = s;
    j->response_len = len;
}

yyjson_doc *parse_json_body(job *j, yyjson_mut_doc *detail) {
    if (!j->body || j->body_len == 0) {
        ve_add(detail, "[\"body\"]", "body required", "missing");
        return NULL;
    }
    /* yyjson_read with default flags requires the whole buffer to be one JSON
     * value (trailing whitespace is allowed, trailing content is an error), so
     * it already enforces "the entire body must be consumed". */
    yyjson_doc *doc = yyjson_read(j->body, j->body_len, 0);
    if (!doc) {
        ve_add(detail, "[\"body\"]", "invalid JSON", "json_invalid");
        return NULL;
    }
    if (!yyjson_is_obj(yyjson_doc_get_root(doc))) {
        ve_add(detail, "[\"body\"]", "body must be a JSON object", "type_error.object");
        yyjson_doc_free(doc);
        return NULL;
    }
    return doc;
}

bool json_is_integer(yyjson_val *item) {
    if (!yyjson_is_num(item))
        return false;
    double d = yyjson_get_num(item);
    if (!(d >= (double)INT_MIN && d <= (double)INT_MAX))
        return false;
    /* A real such as 5.0 counts as an integer; 5.5 does not. */
    return d == (double)(int)d;
}

const char *encoding_from_root(yyjson_val *root, yyjson_mut_doc *detail, embedding_api_t api) {
    const char *dflt = api == FFWD_API_PERPLEXITY ? "base64_int8" : "float";
    yyjson_val *encoding = yyjson_obj_get(root, "encoding_format");
    if (!encoding)
        return dflt;
    const char *v = yyjson_get_str(encoding);
    if (!v) {
        ve_add(detail, "[\"body\",\"encoding_format\"]", "encoding_format must be a string",
               "type_error.string");
        return dflt;
    }
    int ok = api == FFWD_API_PERPLEXITY
                 ? (!strcmp(v, "base64_int8") || !strcmp(v, "base64_binary") || !strcmp(v, "float"))
                 : (!strcmp(v, "float") || !strcmp(v, "base64"));
    if (!ok) {
        ve_add(detail, "[\"body\",\"encoding_format\"]",
               api == FFWD_API_PERPLEXITY
                   ? "value is not a valid enum member; permitted: "
                     "'base64_int8', 'base64_binary', 'float'"
                   : "value is not a valid enum member; permitted: 'float', 'base64'",
               "enum");
        return dflt;
    }
    return v;
}

/* Parse the text_type hint. Returns 1 when the model's query instruction
 * should be prepended, 0 otherwise (document / absent). Models without a
 * published query instruction reject the field. */
int text_type_is_query(yyjson_val *root, yyjson_mut_doc *detail, const char *query_instruct) {
    yyjson_val *tt = yyjson_obj_get(root, "text_type");
    if (!tt)
        return 0;
    if (!query_instruct) {
        ve_add(detail, "[\"body\",\"text_type\"]",
               "text_type is only supported for instruction embedding models", "extra_fields");
        return 0;
    }
    const char *v = yyjson_get_str(tt);
    if (!v) {
        ve_add(detail, "[\"body\",\"text_type\"]", "text_type must be a string", "type_error.string");
        return 0;
    }
    if (!strcmp(v, "query"))
        return 1;
    if (!strcmp(v, "document"))
        return 0;
    ve_add(detail, "[\"body\",\"text_type\"]",
           "value is not a valid enum member; permitted: 'query', 'document'", "enum");
    return 0;
}

int dimensions_from_root(
    yyjson_val *root, yyjson_mut_doc *detail, int min_dim, int max_dim, const char *encoding) {
    int dims = max_dim;
    yyjson_val *dimensions = yyjson_obj_get(root, "dimensions");
    if (dimensions) {
        if (!json_is_integer(dimensions)) {
            ve_add(detail, "[\"body\",\"dimensions\"]", "dimensions must be an integer",
                   "type_error.integer");
        } else {
            dims = (int)yyjson_get_num(dimensions);
            if (dims < min_dim || dims > max_dim) {
                char msg[96];
                snprintf(msg, sizeof(msg), "dimensions must be between %d and %d", min_dim, max_dim);
                ve_add(detail, "[\"body\",\"dimensions\"]", msg, "value_error.range");
            }
        }
    }
    if (!strcmp(encoding, "base64_binary") && dims % 8 != 0)
        ve_add(detail, "[\"body\",\"dimensions\"]",
               "dimensions must be divisible by 8 for base64_binary", "value_error.divisible");
    return dims;
}
