/* tests/test_tokenizer.c - hermetic byte-level BPE tokenizer tests.
 * Synthesizes a complete GPT-2-style vocab.json (all 256 byte tokens plus
 * merged tokens) and merges.txt, then checks exact token ids through the
 * integer-pair fast path, and that dropping the integer tables (forcing the
 * string-based fallback) produces identical ids. Runs via `make test`. */

#include "qwen_tokenizer.h"
#include "tok_fixture.h"

#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
int qwen_verbose = 0;

#define TEST_ASSERT(cond) do {                                          \
    if (!(cond)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++;                                                     \
    }                                                                   \
} while (0)

static void check_ids(const qwen_tokenizer_t *tok,
                      qwen_tokenizer_workspace_t *ws, const char *text,
                      const int *want, int n_want, const char *label)
{
    int n = 0;
    int *ids = qwen_tokenizer_encode_with_workspace(
        (qwen_tokenizer_t *)tok, ws, text, &n);
    TEST_ASSERT(ids != NULL);
    if (!ids) return;
    if (n != n_want) {
        fprintf(stderr, "FAIL %s \"%s\": got %d ids want %d\n",
                label, text, n, n_want);
        failures++;
    } else {
        for (int i = 0; i < n; i++) {
            if (ids[i] != want[i]) {
                fprintf(stderr, "FAIL %s \"%s\": ids[%d]=%d want %d\n",
                        label, text, i, ids[i], want[i]);
                failures++;
            }
        }
    }
    free(ids);
}

/* Concatenated decode of an encoding must reproduce the (NFC-normalized)
 * input - byte-level BPE is lossless. */
static void check_roundtrip(const qwen_tokenizer_t *tok,
                            qwen_tokenizer_workspace_t *ws, const char *text,
                            const char *want, const char *label)
{
    int n = 0;
    int *ids = qwen_tokenizer_encode_with_workspace(
        (qwen_tokenizer_t *)tok, ws, text, &n);
    TEST_ASSERT(ids != NULL && n > 0);
    if (!ids) return;
    size_t cap = 1;
    for (int i = 0; i < n; i++)
        cap += strlen(qwen_tokenizer_decode(tok, ids[i]));
    char *got = (char *)malloc(cap);
    if (!got) { free(ids); TEST_ASSERT(0); return; }
    got[0] = '\0';
    for (int i = 0; i < n; i++)
        strcat(got, qwen_tokenizer_decode(tok, ids[i]));
    if (strcmp(got, want) != 0) {
        fprintf(stderr, "FAIL %s roundtrip \"%s\": got \"%s\" want \"%s\"\n",
                label, text, got, want);
        failures++;
    }
    free(got);
    free(ids);
}

/* Two spellings of the same text must encode identically. */
static void check_same_ids(const qwen_tokenizer_t *tok,
                           qwen_tokenizer_workspace_t *ws,
                           const char *a, const char *b, const char *label)
{
    int na = 0, nb = 0;
    int *ia = qwen_tokenizer_encode_with_workspace(
        (qwen_tokenizer_t *)tok, ws, a, &na);
    if (!ia) { TEST_ASSERT(0); return; }
    int *ib = qwen_tokenizer_encode_with_workspace(
        (qwen_tokenizer_t *)tok, ws, b, &nb);
    if (!ib) { free(ia); TEST_ASSERT(0); return; }
    if (na != nb || memcmp(ia, ib, (size_t)na * sizeof(int)) != 0) {
        fprintf(stderr, "FAIL %s same-ids \"%s\" vs \"%s\"\n", label, a, b);
        failures++;
    }
    free(ib);
    free(ia);
}

static void run_cases(const qwen_tokenizer_t *tok,
                      qwen_tokenizer_workspace_t *ws, const char *label)
{
    /* h e l l o -> he -> he,ll -> hell -> hello */
    const int hello[] = {259};
    check_ids(tok, ws, "hello", hello, 1, label);

    /* "hello world": piece 2 = "Ġworld" -> Gw, or, ld -> Gwor, ld */
    const int hello_world[] = {259, 263, 262};
    check_ids(tok, ws, "hello world", hello_world, 3, label);

    /* h e l d -> he, l, d -> he, ld (no "he ld" merge) */
    const int held[] = {256, 262};
    check_ids(tok, ws, "held", held, 2, label);

    /* No applicable merges: raw byte tokens. */
    const int dr[] = {'d', 'r'};
    check_ids(tok, ws, "dr", dr, 2, label);

    /* Pretokenizer codepoint classes (the ASCII class-table paths). */

    /* A letter run stops at a digit; digits split one per piece. */
    const int x123[] = {'x', '1', '2', '3'};
    check_ids(tok, ws, "x123", x123, 4, label);

    /* Punctuation is its own piece; " world" joins as one piece
     * (Gworld -> Gwor, ld); trailing "!" is a symbol piece. */
    const int punct[] = {259, ',', 263, 262, '!'};
    check_ids(tok, ws, "hello, world!", punct, 5, label);

    /* Tab and newline are spaces; each stays a piece between letters. */
    const int tab[] = {'a', '\t', 'b'};
    check_ids(tok, ws, "a\tb", tab, 3, label);
    const int nl[] = {'a', '\n', 'b'};
    check_ids(tok, ws, "a\nb", nl, 3, label);

    /* A space is not absorbed into a digit piece (digits are not letters). */
    const int sp_digit[] = {'7', ' ', '8'};
    check_ids(tok, ws, "7 8", sp_digit, 3, label);

    /* Contraction suffix splits off. */
    const int its[] = {'i', 't', '\'', 's'};
    check_ids(tok, ws, "it's", its, 4, label);

    /* Non-ASCII falls through the table: U+00E9 is a letter (byte pair);
     * U+2026 is symbolish, so it prefixes the following letter run. */
    const int eacute[] = {0xC3, 0xA9};
    check_ids(tok, ws, "\xC3\xA9", eacute, 2, label);

    /* NFC: decomposed base+combining-mark spellings must match the
     * precomposed codepoint (one pair per mark in the composition table). */
    check_same_ids(tok, ws, "e\xCC\x81", "\xC3\xA9", label);   /* acute */
    check_same_ids(tok, ws, "a\xCC\x80", "\xC3\xA0", label);   /* grave */
    check_same_ids(tok, ws, "e\xCC\x82", "\xC3\xAA", label);   /* circumflex */
    check_same_ids(tok, ws, "n\xCC\x83", "\xC3\xB1", label);   /* tilde */
    check_same_ids(tok, ws, "u\xCC\x88", "\xC3\xBC", label);   /* diaeresis */
    check_same_ids(tok, ws, "a\xCC\x8A", "\xC3\xA5", label);   /* ring */
    check_same_ids(tok, ws, "c\xCC\xA7", "\xC3\xA7", label);   /* cedilla */
    check_same_ids(tok, ws, "A\xCC\x80", "\xC3\x80", label);   /* upper */
    check_roundtrip(tok, ws, "cafe\xCC\x81", "caf\xC3\xA9", label);
    /* 'x' + acute has no composition: bytes pass through untouched. */
    check_roundtrip(tok, ws, "x\xCC\x81", "x\xCC\x81", label);

    /* Contraction splitting (ASCII apostrophe forms, both cases). */
    check_roundtrip(tok, ws, "it's", "it's", label);
    check_roundtrip(tok, ws, "don't", "don't", label);
    check_roundtrip(tok, ws, "I'm", "I'm", label);
    check_roundtrip(tok, ws, "he'd", "he'd", label);
    check_roundtrip(tok, ws, "we're", "we're", label);
    check_roundtrip(tok, ws, "you've", "you've", label);
    check_roundtrip(tok, ws, "she'll", "she'll", label);
    check_roundtrip(tok, ws, "IT'S", "IT'S", label);
    check_roundtrip(tok, ws, "rock '", "rock '", label);

    /* Arbitrary and broken byte sequences survive byte-level fallback. */
    check_roundtrip(tok, ws, "\x80", "\x80", label);
    check_roundtrip(tok, ws, "\xFF\xFE", "\xFF\xFE", label);
    check_roundtrip(tok, ws, "a\xE2\x82", "a\xE2\x82", label);
    check_roundtrip(tok, ws, "\xF0\x9F\x98\x80", "\xF0\x9F\x98\x80", label);
    check_roundtrip(tok, ws, "tab\tand\nnl 42!", "tab\tand\nnl 42!", label);
    const int ellipsis[] = {'a', 0xE2, 0x80, 0xA6, 'b'};
    check_ids(tok, ws, "a\xE2\x80\xA6" "b", ellipsis, 5, label);
}

static void write_file(const char *dir, const char *name, const char *body)
{
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fputs(body, f);
    fclose(f);
}

/* Loader rejection paths and vocab.json escape handling, in throwaway
 * subdirectories of the main fixture dir. */
static void test_loader_edges(const char *root)
{
    char dir[1152], path[2048];

    TEST_ASSERT(qwen_tokenizer_load("/nonexistent/vocab.json") == NULL);

    /* Not a JSON object. */
    snprintf(dir, sizeof(dir), "%s/badjson", root);
    if (mkdir(dir, 0755) != 0) { perror(dir); exit(2); }
    write_file(dir, "vocab.json", "[1, 2]\n");
    write_file(dir, "merges.txt", "a b\n");
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    TEST_ASSERT(qwen_tokenizer_load(path) == NULL);

    /* Missing or pair-less merges.txt is tolerated: the tokenizer loads
     * and encoding falls back to byte/vocab-level (no merges applied). */
    snprintf(dir, sizeof(dir), "%s/nomerges", root);
    if (mkdir(dir, 0755) != 0) { perror(dir); exit(2); }
    write_file(dir, "vocab.json", "{\"a\":0,\"b\":1}\n");
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    qwen_tokenizer_t *nom = qwen_tokenizer_load(path);
    TEST_ASSERT(nom != NULL);
    if (nom) {
        qwen_tokenizer_workspace_t *nws = qwen_tokenizer_workspace_new();
        int n = 0;
        int *ids = qwen_tokenizer_encode_with_workspace(nom, nws, "ab", &n);
        TEST_ASSERT(ids && n == 2 && ids[0] == 0 && ids[1] == 1);
        free(ids);
        qwen_tokenizer_workspace_free(nws);
        qwen_tokenizer_free(nom);
    }

    snprintf(dir, sizeof(dir), "%s/emptymerges", root);
    if (mkdir(dir, 0755) != 0) { perror(dir); exit(2); }
    write_file(dir, "vocab.json", "{\"a\":0,\"b\":1}\n");
    write_file(dir, "merges.txt", "#version: 0.2\n\n   \n");
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    qwen_tokenizer_t *emp = qwen_tokenizer_load(path);
    TEST_ASSERT(emp != NULL);
    qwen_tokenizer_free(emp);

    /* Escape forms in vocab keys; a malformed merge line is skipped. */
    snprintf(dir, sizeof(dir), "%s/escapes", root);
    if (mkdir(dir, 0755) != 0) { perror(dir); exit(2); }
    write_file(dir, "vocab.json",
               "{\"a\":0, \"b\":1, \"ab\":2,\n"
               " \"\\u20AC\":3, \"x\\ny\":4, \"q\\\"r\":5,\n"
               " \"s\\\\t\":6, \"p\\/q\":7, \"u\\tv\":8, \"\\u0041Z\":9}\n");
    write_file(dir, "merges.txt", "#version: 0.2\nmalformed-line\na b\n");
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    qwen_tokenizer_t *tok = qwen_tokenizer_load(path);
    TEST_ASSERT(tok != NULL);
    if (tok) {
        TEST_ASSERT(qwen_tokenizer_token_id(tok, "\xE2\x82\xAC") == 3);
        TEST_ASSERT(qwen_tokenizer_token_id(tok, "x\ny") == 4);
        TEST_ASSERT(qwen_tokenizer_token_id(tok, "q\"r") == 5);
        TEST_ASSERT(qwen_tokenizer_token_id(tok, "s\\t") == 6);
        TEST_ASSERT(qwen_tokenizer_token_id(tok, "p/q") == 7);
        TEST_ASSERT(qwen_tokenizer_token_id(tok, "u\tv") == 8);
        TEST_ASSERT(qwen_tokenizer_token_id(tok, "AZ") == 9);
        TEST_ASSERT(strcmp(qwen_tokenizer_decode(tok, 2), "ab") == 0);
        /* Keys outside the GPT-2 byte alphabet decode to replacement. */
        TEST_ASSERT(strcmp(qwen_tokenizer_decode(tok, 3), "?") == 0);
        qwen_tokenizer_free(tok);
    }

    /* A bare filename exercises the no-directory merges-path branch. */
    char saved_cwd[1024];
    if (getcwd(saved_cwd, sizeof(saved_cwd)) && chdir(dir) == 0) {
        qwen_tokenizer_t *rel = qwen_tokenizer_load("vocab.json");
        TEST_ASSERT(rel != NULL);
        qwen_tokenizer_free(rel);
        if (chdir(saved_cwd) != 0) { perror("chdir back"); exit(2); }
    }
}

int main(void)
{
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/pplx-tok-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(dir) || tf_write_vocab(dir) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    char vocab_path[2048];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", dir);
    qwen_tokenizer_t *tok = qwen_tokenizer_load(vocab_path);
    TEST_ASSERT(tok != NULL);
    if (!tok) return 1;

    /* The complete byte alphabet means the integer-pair tables must exist. */
    TEST_ASSERT(tok->int_merges != NULL);
    TEST_ASSERT(tok->cp_to_id != NULL);

    qwen_tokenizer_workspace_t *ws = qwen_tokenizer_workspace_new();
    TEST_ASSERT(ws != NULL);

    run_cases(tok, ws, "fast");

    /* decode: byte tokens, merged tokens, and out-of-range ids. */
    TEST_ASSERT(strcmp(qwen_tokenizer_decode(tok, 'a'), "a") == 0);
    TEST_ASSERT(strcmp(qwen_tokenizer_decode(tok, 259), "hello") == 0);
    TEST_ASSERT(strcmp(qwen_tokenizer_decode(tok, -1), "") == 0);
    TEST_ASSERT(strcmp(qwen_tokenizer_decode(tok, TF_VOCAB_SIZE), "") == 0);
    TEST_ASSERT(strcmp(qwen_tokenizer_decode(NULL, 0), "") == 0);

    /* token_id lookups. */
    TEST_ASSERT(qwen_tokenizer_token_id(tok, "hello") == 259);
    TEST_ASSERT(qwen_tokenizer_token_id(tok, "<|endoftext|>") == TF_EOT_ID);
    TEST_ASSERT(qwen_tokenizer_token_id(tok, "no-such-token") == -1);
    TEST_ASSERT(qwen_tokenizer_token_id(NULL, "a") == -1);
    TEST_ASSERT(qwen_tokenizer_token_id(tok, NULL) == -1);

    /* encode_into: exact fit, truncation (-2 plus the untruncated prefix),
     * zero capacity, empty text, and argument guards. */
    {
        int out[8];
        int n = -1;
        TEST_ASSERT(qwen_tokenizer_encode_into(tok, ws, "hello world", out, 8,
                                               &n) == 0 && n == 3);
        TEST_ASSERT(out[0] == 259 && out[1] == 263 && out[2] == 262);
        /* On overflow the prefix is written and n reports the total
         * required count, so a caller can re-size and retry. */
        n = -1;
        TEST_ASSERT(qwen_tokenizer_encode_into(tok, ws, "hello world", out, 2,
                                               &n) == -2 && n == 3);
        TEST_ASSERT(out[0] == 259 && out[1] == 263);
        n = -1;
        TEST_ASSERT(qwen_tokenizer_encode_into(tok, ws, "hello world", out, 0,
                                               &n) == -2 && n == 3);
        n = -1;
        TEST_ASSERT(qwen_tokenizer_encode_into(tok, ws, "", out, 8, &n) == 0 &&
                    n == 0);
        TEST_ASSERT(qwen_tokenizer_encode_into(tok, ws, NULL, out, 8, &n) == -1);
        TEST_ASSERT(qwen_tokenizer_encode_into(NULL, ws, "x", out, 8, &n) == -1);
    }
    TEST_ASSERT(qwen_tokenizer_encode(tok, "", NULL) == NULL);
    TEST_ASSERT(qwen_tokenizer_encode(NULL, "x", NULL) == NULL);
    qwen_tokenizer_workspace_free(NULL);
    qwen_tokenizer_free(NULL);

    /* Drop the integer tables: the string-based fallback must produce the
     * exact same ids. */
    free(tok->int_merges);
    tok->int_merges = NULL;
    tok->int_merges_cap = 0;
    free(tok->cp_to_id);
    tok->cp_to_id = NULL;
    run_cases(tok, ws, "slow");

    qwen_tokenizer_workspace_free(ws);
    qwen_tokenizer_free(tok);

    test_loader_edges(dir);

    if (failures) {
        fprintf(stderr, "tokenizer tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: tokenizer BPE tests passed");
    return 0;
}
