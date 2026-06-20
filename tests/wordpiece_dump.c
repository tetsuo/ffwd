/* Parity driver for the WordPiece tokenizer. Loads a BERT vocab dir, encodes
 * each argument through both encode entry points (and cross-checks they agree),
 * then prints "<count> <id> <id> ..." per line. check_wordpiece_parity.py builds
 * this with `make parity-wordpiece-driver` and compares the ids against the
 * stored reference vectors in tests/test-vectors/wordpiece/. Links libtok. */
#include "wordpiece.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL_DIR [TEXT...]\n", argv[0]);
        return 2;
    }
    tok_wp_t *tok = tok_wp_load(argv[1]);
    if (!tok) {
        fprintf(stderr, "failed to load tokenizer: %s\n", argv[1]);
        return 1;
    }
    tok_wp_workspace_t *ws = tok_wp_workspace_new();
    for (int a = 2; a < argc; a++) {
        int n = 0;
        int *ids = tok_wp_encode(tok, argv[a], &n);
        if (!ids && n != 0) {
            fprintf(stderr, "encode failed for argument %d\n", a - 2);
            return 1;
        }
        int n_ws = 0;
        int *ids_ws = tok_wp_encode_with_workspace(tok, ws, argv[a], &n_ws);
        if (n_ws != n) {
            fprintf(stderr, "workspace count mismatch for argument %d\n", a - 2);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (ids[i] != ids_ws[i]) {
                fprintf(stderr, "workspace id mismatch for argument %d\n", a - 2);
                return 1;
            }
        }
        printf("%d", n);
        for (int i = 0; i < n; i++)
            printf(" %d", ids[i]);
        printf("\n");
        free(ids);
        free(ids_ws);
    }
    tok_wp_workspace_free(ws);
    tok_wp_free(tok);
    return 0;
}
