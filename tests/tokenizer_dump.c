/* BPE tokenizer parity driver.
 * Loads vocab.json; merges.txt is read from the same directory.
 * Encodes each argument through the three encode entry points and checks that
 * they agree.
 *
 * Prints one line per argument: "<count> <id> <id> ...".
 *
 * check_tokenizer_parity.py builds this with `make parity-tokenizer-driver` and
 * compares ids against reference vectors in tests/test-vectors/bpe/.
 */

#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s VOCAB_JSON [TEXT...]\n", argv[0]);
        return 2;
    }

    tok_bpe_t *tok = tok_bpe_load(argv[1]);
    if (!tok) {
        fprintf(stderr, "failed to load tokenizer: %s\n", argv[1]);
        return 1;
    }
    tok_bpe_workspace_t *ws = tok_bpe_workspace_new();
    if (!ws) {
        fprintf(stderr, "failed to allocate tokenizer workspace\n");
        tok_bpe_free(tok);
        return 1;
    }

    for (int a = 2; a < argc; a++) {
        int n = 0;
        int *ids = tok_bpe_encode(tok, argv[a], &n);
        if (!ids && n != 0) {
            fprintf(stderr, "failed to encode argument %d\n", a - 2);
            tok_bpe_free(tok);
            return 1;
        }

        int n_ws = 0;
        int *ids_ws = tok_bpe_encode_with_workspace(tok, ws, argv[a], &n_ws);
        if (n_ws != n || (n_ws > 0 && !ids_ws)) {
            fprintf(stderr, "workspace encode count mismatch for argument %d\n", a - 2);
            free(ids);
            free(ids_ws);
            tok_bpe_workspace_free(ws);
            tok_bpe_free(tok);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (ids[i] != ids_ws[i]) {
                fprintf(stderr, "workspace encode mismatch for argument %d token %d\n", a - 2, i);
                free(ids);
                free(ids_ws);
                tok_bpe_workspace_free(ws);
                tok_bpe_free(tok);
                return 1;
            }
        }

        int cap = (int)(strlen(argv[a]) * 2 + 8);
        if (cap < 1)
            cap = 1;
        int *into = (int *)malloc((size_t)cap * sizeof(int));
        int n_into = 0;
        int rc = tok_bpe_encode_into(tok, ws, argv[a], into, cap, &n_into);
        if (rc == -2) {
            int *tmp = (int *)realloc(into, (size_t)n_into * sizeof(int));
            if (!tmp) {
                free(into);
                free(ids);
                free(ids_ws);
                tok_bpe_workspace_free(ws);
                tok_bpe_free(tok);
                return 1;
            }
            into = tmp;
            cap = n_into;
            rc = tok_bpe_encode_into(tok, ws, argv[a], into, cap, &n_into);
        }
        if (rc != 0 || n_into != n) {
            fprintf(stderr, "encode_into mismatch for argument %d\n", a - 2);
            free(into);
            free(ids);
            free(ids_ws);
            tok_bpe_workspace_free(ws);
            tok_bpe_free(tok);
            return 1;
        }
        for (int i = 0; i < n; i++) {
            if (ids[i] != into[i]) {
                fprintf(stderr, "encode_into mismatch for argument %d token %d\n", a - 2, i);
                free(into);
                free(ids);
                free(ids_ws);
                tok_bpe_workspace_free(ws);
                tok_bpe_free(tok);
                return 1;
            }
        }

        printf("%d", n);
        for (int i = 0; i < n; i++)
            printf(" %d", ids[i]);
        printf("\n");
        free(into);
        free(ids_ws);
        free(ids);
    }

    tok_bpe_workspace_free(ws);
    tok_bpe_free(tok);
    return 0;
}
