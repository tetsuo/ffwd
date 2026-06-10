/* tests/test_safetensors.c - safetensors parser limit and rejection tests.
 * Hermetic: synthesizes every fixture in a temp directory. Runs via
 * `make test`. */

#include "qwen_safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures = 0;
static char tmproot[1024];

/* Write u64-LE header length + JSON header + payload (safetensors layout). */
static void write_st(const char *path, const char *header,
                     const void *payload, size_t payload_len)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(2); }
    uint64_t n = strlen(header);
    unsigned char len8[8];
    for (int i = 0; i < 8; i++) len8[i] = (unsigned char)(n >> (8 * i));
    fwrite(len8, 1, 8, f);
    fwrite(header, 1, n, f);
    if (payload_len) fwrite(payload, 1, payload_len, f);
    fclose(f);
}

static const char *path_in(const char *rel)
{
    static char buf[2048];
    snprintf(buf, sizeof(buf), "%s/%s", tmproot, rel);
    return buf;
}

static void check(const char *mode, const char *path, int want_ok)
{
    int got_ok;
    if (strcmp(mode, "file") == 0) {
        safetensors_file_t *sf = safetensors_open(path);
        got_ok = sf != NULL;
        safetensors_close(sf);
    } else {
        multi_safetensors_t *ms = multi_safetensors_open(path);
        got_ok = ms != NULL;
        multi_safetensors_close(ms);
    }
    if (got_ok != want_ok) {
        fprintf(stderr, "%s %s: got %s want %s\n", mode, path,
                got_ok ? "ok" : "fail", want_ok ? "ok" : "fail");
        failures++;
    }
}

static const char ONE_TENSOR_A[] =
    "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
static const char ONE_TENSOR_B[] =
    "{\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
static const char ONE_TENSOR_X[] =
    "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";

int main(void)
{
    snprintf(tmproot, sizeof(tmproot), "%s/pplx-st-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(tmproot)) { perror("mkdtemp"); return 2; }

    static const char zeros[4 * 1025] = {0};

    write_st(path_in("valid.safetensors"), ONE_TENSOR_X, zeros, 4);
    check("file", path_in("valid.safetensors"), 1);

    /* 1025 tensors exceeds the parser's tensor-count limit. */
    {
        size_t cap = 1025 * 64 + 16;
        char *hdr = malloc(cap);
        if (!hdr) return 2;
        size_t off = 0;
        off += (size_t)snprintf(hdr + off, cap - off, "{");
        for (int i = 0; i < 1025; i++) {
            off += (size_t)snprintf(hdr + off, cap - off,
                "%s\"t%d\":{\"dtype\":\"F32\",\"shape\":[1],"
                "\"data_offsets\":[%d,%d]}",
                i ? "," : "", i, i * 4, i * 4 + 4);
        }
        snprintf(hdr + off, cap - off, "}");
        write_st(path_in("too_many.safetensors"), hdr, zeros, sizeof(zeros));
        free(hdr);
        check("file", path_in("too_many.safetensors"), 0);
    }

    write_st(path_in("too_many_dims.safetensors"),
             "{\"x\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,1,1,1,1,1],"
             "\"data_offsets\":[0,4]}}", zeros, 4);
    check("file", path_in("too_many_dims.safetensors"), 0);

    write_st(path_in("bad_offsets.safetensors"),
             "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
             "\"data_offsets\":[4,0]}}", zeros, 4);
    check("file", path_in("bad_offsets.safetensors"), 0);

    mkdir(path_in("valid_multi"), 0755);
    write_st(path_in("valid_multi/model-00001-of-00002.safetensors"),
             ONE_TENSOR_A, zeros, 4);
    write_st(path_in("valid_multi/model-00002-of-00002.safetensors"),
             ONE_TENSOR_B, zeros, 4);
    check("multi", path_in("valid_multi"), 1);

    /* Payload accounting across shards: two 4-byte F32 tensors. */
    {
        multi_safetensors_t *ms = multi_safetensors_open(path_in("valid_multi"));
        size_t nbytes = 0;
        if (!ms || multi_safetensors_data_nbytes(ms, &nbytes) != 0 ||
            nbytes != 8) {
            fprintf(stderr, "multi data_nbytes: got %zu want 8\n", nbytes);
            failures++;
        }
        multi_safetensors_close(ms);
    }

    mkdir(path_in("missing"), 0755);
    write_st(path_in("missing/model-00001-of-00002.safetensors"),
             ONE_TENSOR_X, zeros, 4);
    check("multi", path_in("missing"), 0);

    mkdir(path_in("too_many_shards"), 0755);
    write_st(path_in("too_many_shards/model-00001-of-00009.safetensors"),
             ONE_TENSOR_X, zeros, 4);
    check("multi", path_in("too_many_shards"), 0);

    mkdir(path_in("bad_name"), 0755);
    write_st(path_in("bad_name/model-foo.safetensors"),
             ONE_TENSOR_X, zeros, 4);
    check("multi", path_in("bad_name"), 0);

    if (failures) {
        fprintf(stderr, "safetensors tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: safetensors parser limit checks passed");
    return 0;
}
