/* tests/test_tokenizer.c - hermetic byte-level BPE tokenizer tests.
 * Synthesizes a complete GPT-2-style vocab.json (all 256 byte tokens plus
 * merged tokens) and merges.txt, then checks exact token ids through the
 * integer-pair fast path, and that dropping the integer tables (forcing the
 * string-based fallback) produces identical ids. Runs via `make test`. */

#include "qwen_tokenizer.h"

#include <stdio.h>
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

/* GPT-2 byte -> unicode codepoint mapping (mirrors the tokenizer's rules). */
static int byte_to_cp(int b)
{
    static int table[256];
    static int init = 0;
    if (!init) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            int normal = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) ||
                         (i >= 174 && i <= 255);
            table[i] = normal ? i : 256 + n++;
        }
        init = 1;
    }
    return table[b];
}

static int cp_to_utf8(int cp, char *out)
{
    if (cp < 0x80) { out[0] = (char)cp; out[1] = '\0'; return 1; }
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    out[2] = '\0';
    return 2;
}

/* JSON-escape a byte-unicode token (the byte alphabet needs only \" and \\). */
static const char *json_escape(const char *s, char *buf, size_t cap)
{
    size_t o = 0;
    for (; *s && o + 2 < cap; s++) {
        if (*s == '"' || *s == '\\') buf[o++] = '\\';
        buf[o++] = *s;
    }
    buf[o] = '\0';
    return buf;
}

/* Merged tokens (ids 256+) and the merge list that produces them. */
static const char *MERGED[] = {
    "he", "ll", "hell", "hello", "\xC4\xA0w", "or", "ld", "\xC4\xA0wor",
};
static const char *MERGES[] = {
    "h e", "l l", "he ll", "hell o", "\xC4\xA0 w", "o r", "l d",
    "\xC4\xA0w or",
};
enum { N_MERGED = sizeof(MERGED) / sizeof(MERGED[0]) };

static int write_fixture(const char *dir)
{
    char path[2048], tok[4], esc[16];
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fputs("{", f);
    for (int b = 0; b < 256; b++) {
        cp_to_utf8(byte_to_cp(b), tok);
        fprintf(f, "%s\"%s\": %d", b ? ", " : "",
                json_escape(tok, esc, sizeof(esc)), b);
    }
    for (int i = 0; i < N_MERGED; i++)
        fprintf(f, ", \"%s\": %d", MERGED[i], 256 + i);
    fputs("}", f);
    fclose(f);

    snprintf(path, sizeof(path), "%s/merges.txt", dir);
    f = fopen(path, "w");
    if (!f) return -1;
    for (size_t i = 0; i < sizeof(MERGES) / sizeof(MERGES[0]); i++)
        fprintf(f, "%s\n", MERGES[i]);
    fclose(f);
    return 0;
}

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
    const int ellipsis[] = {'a', 0xE2, 0x80, 0xA6, 'b'};
    check_ids(tok, ws, "a\xE2\x80\xA6" "b", ellipsis, 5, label);
}

int main(void)
{
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/pplx-tok-test-XXXXXX",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(dir) || write_fixture(dir) != 0) {
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

    if (failures) {
        fprintf(stderr, "tokenizer tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("ok: tokenizer BPE tests passed");
    return 0;
}
