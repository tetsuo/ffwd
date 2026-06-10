/* tests/test_safetensors.c - safetensors parser limit and rejection tests.
 * Runs via `make test`. Fixtures are synthesized in a temp directory. */

#include "qwen_safetensors.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s file|multi path ok|fail\n", argv[0]);
        return 2;
    }

    int want_ok = strcmp(argv[3], "ok") == 0;
    int got_ok = 0;
    if (strcmp(argv[1], "file") == 0) {
        safetensors_file_t *sf = safetensors_open(argv[2]);
        got_ok = sf != NULL;
        safetensors_close(sf);
    } else if (strcmp(argv[1], "multi") == 0) {
        multi_safetensors_t *ms = multi_safetensors_open(argv[2]);
        got_ok = ms != NULL;
        multi_safetensors_close(ms);
    } else {
        return 2;
    }

    if (got_ok != want_ok) {
        fprintf(stderr, "%s %s: got %s want %s\n",
                argv[1], argv[2], got_ok ? "ok" : "fail",
                want_ok ? "ok" : "fail");
        return 1;
    }
    return 0;
}
