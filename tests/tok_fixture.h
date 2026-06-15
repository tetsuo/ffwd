/* tests/tok_fixture.h - synthesize a complete GPT-2-style byte-level BPE
 * tokenizer fixture into a directory: vocab.json with all 256 byte tokens
 * (ids 0-255, so any input tokenizes) plus a few merged tokens (ids 256+),
 * and the matching merges.txt. Shared by the tokenizer and server tests. */

#ifndef TOK_FIXTURE_H
#define TOK_FIXTURE_H

#include <stdio.h>
#include <string.h>

/* GPT-2 byte -> unicode codepoint mapping (mirrors the tokenizer's rules). */
static int tf_byte_to_cp(int b) {
    static int table[256];
    static int init = 0;
    if (!init) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            int normal = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255);
            table[i] = normal ? i : 256 + n++;
        }
        init = 1;
    }
    return table[b];
}

static int tf_cp_to_utf8(int cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    }
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    out[2] = '\0';
    return 2;
}

/* JSON-escape a byte-unicode token (the byte alphabet needs only \" and \\). */
static const char *tf_json_escape(const char *s, char *buf, size_t cap) {
    size_t o = 0;
    for (; *s && o + 2 < cap; s++) {
        if (*s == '"' || *s == '\\')
            buf[o++] = '\\';
        buf[o++] = *s;
    }
    buf[o] = '\0';
    return buf;
}

/* Merged tokens (ids 256+) and the merge list that produces them. */
static const char *TF_MERGED[] = {
    "he", "ll", "hell", "hello", "\xC4\xA0w", "or", "ld", "\xC4\xA0wor",
};
static const char *TF_MERGES[] = {
    "h e", "l l", "he ll", "hell o", "\xC4\xA0 w", "o r", "l d", "\xC4\xA0w or",
};
enum { TF_N_MERGED = sizeof(TF_MERGED) / sizeof(TF_MERGED[0]) };
/* Special tokens go last so contextual and late-interaction ids resolve
 * in-range for tiny server models. */
enum { TF_EOT_ID = 256 + TF_N_MERGED };
enum { TF_MASK_ID = TF_EOT_ID + 1 };
enum { TF_QUERY_ID = TF_MASK_ID + 1 };
enum { TF_DOCUMENT_ID = TF_QUERY_ID + 1 };
enum { TF_VOCAB_SIZE = TF_DOCUMENT_ID + 1 };

/* Write vocab.json + merges.txt into dir. Returns 0 on success. */
static int tf_write_vocab(const char *dir) {
    char path[2048], tok[4], esc[16];
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fputs("{", f);
    for (int b = 0; b < 256; b++) {
        tf_cp_to_utf8(tf_byte_to_cp(b), tok);
        fprintf(f, "%s\"%s\": %d", b ? ", " : "", tf_json_escape(tok, esc, sizeof(esc)), b);
    }
    for (int i = 0; i < TF_N_MERGED; i++)
        fprintf(f, ", \"%s\": %d", TF_MERGED[i], 256 + i);
    fprintf(f, ", \"<|endoftext|>\": %d", TF_EOT_ID);
    fprintf(f, ", \"[MASK]\": %d", TF_MASK_ID);
    fprintf(f, ", \"[Q]\": %d", TF_QUERY_ID);
    fprintf(f, ", \"[D]\": %d", TF_DOCUMENT_ID);
    fputs("}", f);
    fclose(f);

    snprintf(path, sizeof(path), "%s/merges.txt", dir);
    f = fopen(path, "w");
    if (!f)
        return -1;
    for (size_t i = 0; i < sizeof(TF_MERGES) / sizeof(TF_MERGES[0]); i++)
        fprintf(f, "%s\n", TF_MERGES[i]);
    fclose(f);
    return 0;
}

#endif /* TOK_FIXTURE_H */
