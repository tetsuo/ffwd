/* tests/test_wordpiece.c - hermetic WordPiece tokenizer checks. A synthetic
 * vocab exercises greedy ## matching, [UNK], casing, accent stripping,
 * punctuation splitting, CJK char-splitting, special-token resolution, and the
 * workspace encode paths. No model files or network needed. */

#include "wordpiece.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* id order = line order. */
static const char *VOCAB[] = {
    "[PAD]",  "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the",   "capital",   "of",
    "france", "is",    "paris", ",",     ".",      "un",    "##aff",     "##able",
    "cafe",   "hello", "world", "!",     "2023",   "token", "##ization", "\xe4\xb8\xad",
};
enum { VOCAB_N = (int)(sizeof(VOCAB) / sizeof(VOCAB[0])) };

static int write_fixture(const char *dir, int lower_case) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/vocab.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    for (int i = 0; i < VOCAB_N; i++)
        fprintf(f, "%s\n", VOCAB[i]);
    fclose(f);

    snprintf(path, sizeof(path), "%s/tokenizer_config.json", dir);
    f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f, "{\"do_lower_case\":%s}", lower_case ? "true" : "false");
    fclose(f);
    return 0;
}

static int
check(const tok_wp_t *tok, const char *text, const int *want, int want_n, const char *name) {
    int n = 0;
    int *ids = tok_wp_encode(tok, text, &n);
    if (!ids) {
        fprintf(stderr, "%s: encode returned NULL\n", name);
        return -1;
    }
    int ok = (n == want_n);
    for (int i = 0; ok && i < n; i++)
        ok = ids[i] == want[i];
    if (!ok) {
        fprintf(stderr, "%s: got [", name);
        for (int i = 0; i < n; i++)
            fprintf(stderr, "%d%s", ids[i], i + 1 < n ? "," : "");
        fprintf(stderr, "], want [");
        for (int i = 0; i < want_n; i++)
            fprintf(stderr, "%d%s", want[i], i + 1 < want_n ? "," : "");
        fprintf(stderr, "]\n");
    }
    free(ids);
    return ok ? 0 : -1;
}

int main(void) {
    const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/ffwd-wp-test-XXXXXX", tmp);
    if (!mkdtemp(dir) || write_fixture(dir, 1) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    tok_wp_t *tok = tok_wp_load(dir);
    if (!tok) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    if (tok->vocab_size != VOCAB_N || !tok->do_lower_case || tok->unk_id != 1) {
        fprintf(stderr, "load metadata wrong: vocab=%d lower=%d unk=%d\n", tok->vocab_size,
                tok->do_lower_case, tok->unk_id);
        return 1;
    }

    int rc = 0;
    /* whole words */
    rc |= check(tok, "the capital of france is paris", (int[]){5, 6, 7, 8, 9, 10}, 6, "sentence");
    /* greedy ## subword matching */
    rc |= check(tok, "unaffable", (int[]){13, 14, 15}, 3, "unaffable");
    rc |= check(tok, "tokenization", (int[]){21, 22}, 2, "tokenization");
    /* lowercasing + punctuation split */
    rc |= check(tok, "The Capital.", (int[]){5, 6, 12}, 3, "casing+punct");
    rc |= check(tok, "hello, world!", (int[]){17, 11, 18, 19}, 4, "punct");
    /* accent stripping: café (combining) and café (precomposed) */
    rc |= check(tok, "caf\xc3\xa9", (int[]){16}, 1, "precomposed-accent");
    rc |= check(tok, "cafe\xcc\x81", (int[]){16}, 1, "combining-accent");
    /* digits as a whole token */
    rc |= check(tok, "2023", (int[]){20}, 1, "digits");
    /* unknown word -> single [UNK] */
    rc |= check(tok, "xyzzy", (int[]){1}, 1, "unknown");
    /* CJK ideographs split per character; the second is unknown */
    rc |= check(tok, "\xe4\xb8\xad\xe6\x96\x87", (int[]){23, 1}, 2, "cjk-split");
    if (rc)
        return 1;

    /* named special-token resolution */
    if (tok_wp_token_id(tok, "[CLS]") != 2 || tok_wp_token_id(tok, "[SEP]") != 3 ||
        tok_wp_token_id(tok, "[UNK]") != 1 || tok_wp_token_id(tok, "[MISSING]") != -1) {
        fprintf(stderr, "special-token resolution wrong\n");
        return 1;
    }
    if (strcmp(tok_wp_decode(tok, 5), "the") != 0 || tok_wp_decode(tok, -1) != NULL) {
        fprintf(stderr, "decode wrong\n");
        return 1;
    }

    /* the three encode entry points must agree */
    tok_wp_workspace_t *ws = tok_wp_workspace_new();
    int n_alloc = 0;
    int *alloc = tok_wp_encode_with_workspace(tok, ws, "the unaffable cafe", &n_alloc);
    int into[16], n_into = 0;
    int rc2 = tok_wp_encode_into(tok, ws, "the unaffable cafe", into, 16, &n_into);
    if (!alloc || rc2 != 0 || n_alloc != n_into || n_alloc <= 0) {
        fprintf(stderr, "encode-path counts disagree (%d vs %d, rc=%d)\n", n_alloc, n_into, rc2);
        return 1;
    }
    for (int i = 0; i < n_alloc; i++) {
        if (alloc[i] != into[i]) {
            fprintf(stderr, "encode-path ids disagree at %d\n", i);
            return 1;
        }
    }
    /* a too-small buffer reports the needed count and -2 */
    int tiny[1], n_need = 0;
    if (tok_wp_encode_into(tok, ws, "the unaffable cafe", tiny, 1, &n_need) != -2 ||
        n_need != n_alloc) {
        fprintf(stderr, "small-buffer handling wrong\n");
        return 1;
    }
    free(alloc);
    tok_wp_workspace_free(ws);

    /* cased mode keeps capitals distinct (here -> [UNK] since vocab is lower). */
    char cdir[1024];
    snprintf(cdir, sizeof(cdir), "%s/ffwd-wp-cased-XXXXXX", tmp);
    if (!mkdtemp(cdir) || write_fixture(cdir, 0) != 0)
        return 2;
    tok_wp_t *cased = tok_wp_load(cdir);
    if (!cased || cased->do_lower_case)
        return 1;
    rc |= check(cased, "The", (int[]){1}, 1, "cased-unknown");
    rc |= check(cased, "the", (int[]){5}, 1, "cased-known");
    tok_wp_free(cased);
    if (rc)
        return 1;

    tok_wp_free(tok);
    puts("ok: WordPiece basic+subword tokenization, casing, accents, CJK, specials");
    return 0;
}
