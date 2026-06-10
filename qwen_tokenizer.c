#include "qwen_tokenizer.h"
#include "qwen_kernels.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * GPT-2 Bytes-to-Unicode Mapping
 * ======================================================================== */

static int gpt2_byte_to_unicode[256];
static int gpt2_unicode_to_byte[512]; /* codepoints up to ~384 */
static int gpt2_mapping_initialized = 0;

static void init_gpt2_mapping(void) {
    if (gpt2_mapping_initialized) return;

    memset(gpt2_unicode_to_byte, -1, sizeof(gpt2_unicode_to_byte));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        int is_normal = 0;
        if (b >= 33 && b <= 126) is_normal = 1;   /* '!'..'~' */
        if (b >= 161 && b <= 172) is_normal = 1;  /* '¡'..'¬' */
        if (b >= 174 && b <= 255) is_normal = 1;  /* '®'..'ÿ' */

        if (is_normal) gpt2_byte_to_unicode[b] = b;
        else gpt2_byte_to_unicode[b] = 256 + n++;
    }

    for (int b = 0; b < 256; b++) {
        int cp = gpt2_byte_to_unicode[b];
        if (cp < 512) gpt2_unicode_to_byte[cp] = b;
    }

    gpt2_mapping_initialized = 1;
}

/* Convert one Unicode codepoint to UTF-8 bytes. Returns bytes written. */
static int utf8_encode_cp(int cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

static int utf8_encode_full_cp(int cp, char out[4]) {
    if (cp < 0x80) return utf8_encode_cp(cp, out);
    if (cp < 0x800) return utf8_encode_cp(cp, out);
    if (cp < 0x10000) return utf8_encode_cp(cp, out);
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/*
 * Decode a GPT-2 encoded token string (vocab key) to raw bytes/UTF-8 text.
 * Returns allocated string (caller must free).
 */
static char *decode_gpt2_token(const char *token_str) {
    init_gpt2_mapping();

    size_t len = strlen(token_str);
    unsigned char *bytes = (unsigned char *)malloc(len + 1);
    if (!bytes) return NULL;

    int byte_count = 0;
    const unsigned char *p = (const unsigned char *)token_str;
    const unsigned char *end = p + len;

    while (p < end) {
        int cp = 0;
        int nbytes = 1;

        if ((*p & 0x80) == 0) {
            cp = *p;
            nbytes = 1;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            if (p + 1 < end) cp |= (p[1] & 0x3F);
            nbytes = 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 6;
            if (p + 2 < end) cp |= (p[2] & 0x3F);
            nbytes = 3;
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07) << 18;
            if (p + 1 < end) cp |= (p[1] & 0x3F) << 12;
            if (p + 2 < end) cp |= (p[2] & 0x3F) << 6;
            if (p + 3 < end) cp |= (p[3] & 0x3F);
            nbytes = 4;
        } else {
            cp = *p;
            nbytes = 1;
        }
        p += nbytes;

        if (cp < 512 && gpt2_unicode_to_byte[cp] >= 0) {
            bytes[byte_count++] = (unsigned char)gpt2_unicode_to_byte[cp];
        } else {
            bytes[byte_count++] = '?';
        }
    }
    bytes[byte_count] = '\0';
    return (char *)bytes;
}

/* ========================================================================
 * Simple JSON parser for vocab.json
 * ======================================================================== */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_json_string(const char **p, char *out, size_t max_len) {
    skip_ws(p);
    if (**p != '"') return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else if (**p == '/') out[i++] = '/';
            else if (**p == 'u') {
                (*p)++;
                unsigned int cp = 0;
                for (int j = 0; j < 4 && **p; j++, (*p)++) {
                    cp <<= 4;
                    char c = **p;
                    if (c >= '0' && c <= '9') cp |= (unsigned)(c - '0');
                    else if (c >= 'a' && c <= 'f') cp |= (unsigned)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') cp |= (unsigned)(c - 'A' + 10);
                }
                if (cp < 0x80 && i + 1 < max_len) {
                    out[i++] = (char)cp;
                } else if (cp < 0x800 && i + 2 < max_len) {
                    out[i++] = (char)(0xC0 | (cp >> 6));
                    out[i++] = (char)(0x80 | (cp & 0x3F));
                } else if (i + 3 < max_len) {
                    out[i++] = (char)(0xE0 | (cp >> 12));
                    out[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    out[i++] = (char)(0x80 | (cp & 0x3F));
                }
                continue;
            } else {
                out[i++] = **p;
            }
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p != '"') return -1;
    (*p)++;
    return 0;
}

static int64_t parse_json_int(const char **p) {
    skip_ws(p);
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    int64_t val = 0;
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

/* ========================================================================
 * String->int hash map (open addressing)
 * ======================================================================== */

typedef struct {
    char *key; /* NULL means empty slot */
    int value;
} str_int_entry_t;

static uint64_t fnv1a_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv1a_hash_pair(const char *a, const char *b) {
    uint64_t h = 1469598103934665603ULL;
    while (*a) {
        h ^= (unsigned char)*a++;
        h *= 1099511628211ULL;
    }
    h ^= (unsigned char)' ';
    h *= 1099511628211ULL;
    while (*b) {
        h ^= (unsigned char)*b++;
        h *= 1099511628211ULL;
    }
    return h;
}

static int pair_key_eq(const char *key, const char *a, const char *b) {
    while (*a) {
        if (*key++ != *a++) return 0;
    }
    if (*key++ != ' ') return 0;
    while (*b) {
        if (*key++ != *b++) return 0;
    }
    return *key == '\0';
}

static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

static int map_insert(str_int_entry_t *map, int cap, char *key, int value) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) {
            map[idx].key = key;
            map[idx].value = value;
            return 0;
        }
        if (strcmp(map[idx].key, key) == 0) {
            map[idx].value = value;
            return 0;
        }
    }
    return -1;
}

static int map_insert_owned(str_int_entry_t *map, int cap, char **keyp, int value) {
    if (!keyp) return -1;
    char *key = *keyp;
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) {
            map[idx].key = key;
            map[idx].value = value;
            *keyp = NULL;
            return 0;
        }
        if (strcmp(map[idx].key, key) == 0) {
            map[idx].value = value;
            return 0;
        }
    }
    return -1;
}

static int map_get(const str_int_entry_t *map, int cap, const char *key) {
    if (!map || cap <= 0 || !key) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash(key) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) return -1;
        if (strcmp(map[idx].key, key) == 0) return map[idx].value;
    }
    return -1;
}

static int map_get_pair(const str_int_entry_t *map, int cap,
                        const char *a, const char *b) {
    if (!map || cap <= 0 || !a || !b) return -1;
    int mask = cap - 1;
    int pos = (int)(fnv1a_hash_pair(a, b) & (uint64_t)mask);
    for (int i = 0; i < cap; i++) {
        int idx = (pos + i) & mask;
        if (!map[idx].key) return -1;
        if (pair_key_eq(map[idx].key, a, b)) return map[idx].value;
    }
    return -1;
}

/* ========================================================================
 * BPE helpers
 * ======================================================================== */

static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static int split_utf8_symbols(const char *s, char ***out_syms, int *out_n) {
    *out_syms = NULL;
    *out_n = 0;
    if (!s || !*s) return 0;

    int n = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        p += l;
        n++;
    }
    if (n <= 0) return 0;

    char **syms = (char **)calloc((size_t)n, sizeof(char *));
    if (!syms) return -1;

    int i = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; ) {
        int l = utf8_char_len(*p);
        syms[i] = (char *)malloc((size_t)l + 1);
        if (!syms[i]) {
            for (int j = 0; j < i; j++) free(syms[j]);
            free(syms);
            return -1;
        }
        memcpy(syms[i], p, (size_t)l);
        syms[i][l] = '\0';
        p += l;
        i++;
    }

    *out_syms = syms;
    *out_n = n;
    return 0;
}

static char *str_concat2(const char *a, const char *b) {
    size_t la = strlen(a), lb = strlen(b);
    char *r = (char *)malloc(la + lb + 1);
    if (!r) return NULL;
    memcpy(r, a, la);
    memcpy(r + la, b, lb);
    r[la + lb] = '\0';
    return r;
}

/* ========================================================================
 * Integer-pair BPE merges: (left_id, right_id) -> (rank, merged_id)
 * ======================================================================== */

typedef struct {
    uint64_t key;       /* (left_id << 32) | right_id; UINT64_MAX = empty */
    int rank;
    int merged_id;
} pair_merge_entry_t;

static uint64_t pair_hash(uint64_t k) {
    /* splitmix64 finalizer */
    k ^= k >> 30; k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27; k *= 0x94d049bb133111ebULL;
    k ^= k >> 31;
    return k;
}

static void pair_map_insert(pair_merge_entry_t *map, int cap, uint64_t key,
                            int rank, int merged_id) {
    int mask = cap - 1;
    int idx = (int)(pair_hash(key) & (uint64_t)mask);
    for (int probe = 0; probe < cap; probe++, idx = (idx + 1) & mask) {
        if (map[idx].key == UINT64_MAX) {
            map[idx].key = key;
            map[idx].rank = rank;
            map[idx].merged_id = merged_id;
            return;
        }
        if (map[idx].key == key) return; /* first (lowest) rank wins */
    }
}

/* Returns the rank and sets *merged_id, or INT_MAX when the pair never
 * merges. */
static int pair_merge_lookup(const qwen_tokenizer_t *tok, int left, int right,
                             int *merged_id) {
    const pair_merge_entry_t *map =
        (const pair_merge_entry_t *)tok->int_merges;
    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    int mask = tok->int_merges_cap - 1;
    int idx = (int)(pair_hash(key) & (uint64_t)mask);
    for (int probe = 0; probe < tok->int_merges_cap;
         probe++, idx = (idx + 1) & mask) {
        if (map[idx].key == UINT64_MAX) return INT_MAX;
        if (map[idx].key == key) {
            *merged_id = map[idx].merged_id;
            return map[idx].rank;
        }
    }
    return INT_MAX;
}

static int merge_rank(const qwen_tokenizer_t *tok, const char *a, const char *b) {
    if (!tok->merge_map || tok->merge_map_cap <= 0) return INT_MAX;

    int rank = map_get_pair((const str_int_entry_t *)tok->merge_map,
                            tok->merge_map_cap, a, b);
    return rank >= 0 ? rank : INT_MAX;
}

static int append_id(int **arr, int *n, int *cap, int id) {
    if (*n >= *cap) {
        int new_cap = (*cap == 0) ? 16 : (*cap * 2);
        int *tmp = (int *)realloc(*arr, (size_t)new_cap * sizeof(int));
        if (!tmp) return -1;
        *arr = tmp;
        *cap = new_cap;
    }
    (*arr)[(*n)++] = id;
    return 0;
}

/* Convert UTF-8 bytes to GPT-2 byte-level unicode string. */
static char *text_to_bpe_unicode(const char *text) {
    init_gpt2_mapping();
    size_t len = strlen(text);
    /* Every input byte becomes one codepoint up to 3 UTF-8 bytes. */
    char *out = (char *)malloc(len * 3 + 1);
    if (!out) return NULL;

    size_t w = 0;
    for (size_t i = 0; i < len; i++) {
        int cp = gpt2_byte_to_unicode[(unsigned char)text[i]];
        char tmp[4];
        int n = utf8_encode_cp(cp, tmp);
        for (int j = 0; j < n; j++) out[w++] = tmp[j];
    }
    out[w] = '\0';
    return out;
}

static int encode_bpe_word(const qwen_tokenizer_t *tok, const char *mapped,
                           int **out_ids, int *out_n);

static int utf8_decode_cp_len(const unsigned char *p, const unsigned char *end,
                              int *cp_out, int *len_out) {
    if (p >= end || *p == '\0') return 0;

    unsigned char c = *p;
    if ((c & 0x80) == 0) {
        *cp_out = c;
        *len_out = 1;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && p + 1 < end &&
        (p[1] & 0xC0) == 0x80) {
        *cp_out = ((c & 0x1F) << 6) | (p[1] & 0x3F);
        *len_out = 2;
        return 1;
    }
    if ((c & 0xF0) == 0xE0 && p + 2 < end &&
        (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
        *cp_out = ((c & 0x0F) << 12) |
                  ((p[1] & 0x3F) << 6) |
                  (p[2] & 0x3F);
        *len_out = 3;
        return 1;
    }
    if ((c & 0xF8) == 0xF0 && p + 3 < end &&
        (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 &&
        (p[3] & 0xC0) == 0x80) {
        *cp_out = ((c & 0x07) << 18) |
                  ((p[1] & 0x3F) << 12) |
                  ((p[2] & 0x3F) << 6) |
                  (p[3] & 0x3F);
        *len_out = 4;
        return 1;
    }

    *cp_out = c;
    *len_out = 1;
    return 1;
}

static int is_ascii_alpha_cp(int cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}

static int is_number_cp(int cp) {
    if (cp >= '0' && cp <= '9') return 1;
    if (cp >= 0x0660 && cp <= 0x0669) return 1;
    if (cp >= 0x06F0 && cp <= 0x06F9) return 1;
    if (cp >= 0x0966 && cp <= 0x096F) return 1;
    if (cp >= 0xFF10 && cp <= 0xFF19) return 1;
    return 0;
}

static int is_space_cp(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x85 || cp == 0xA0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

static int is_symbolish_cp(int cp) {
    if (cp < 0x80) return 0;
    if (cp >= 0x0300 && cp <= 0x036F) return 1;
    if (cp >= 0x00A1 && cp <= 0x00BF &&
        cp != 0x00AA && cp != 0x00B5 && cp != 0x00BA)
        return 1;
    if (cp == 0x00D7 || cp == 0x00F7) return 1;
    if (cp >= 0x064B && cp <= 0x065F) return 1;
    if ((cp >= 0x0900 && cp <= 0x0903) ||
        (cp >= 0x093A && cp <= 0x094F) ||
        (cp >= 0x0951 && cp <= 0x0957) ||
        (cp >= 0x0962 && cp <= 0x0963))
        return 1;
    if (cp >= 0x2000 && cp <= 0x206F) return 1;
    if (cp >= 0x2190 && cp <= 0x27BF) return 1;
    if (cp >= 0x2E00 && cp <= 0x2E7F) return 1;
    if (cp >= 0x3000 && cp <= 0x303F) return 1;
    if (cp >= 0x1F000 && cp <= 0x1FAFF) return 1;
    return 0;
}

static int compose_latin_mark(int base, int mark) {
    switch (mark) {
    case 0x0300:
        switch (base) {
        case 'A': return 0x00C0; case 'E': return 0x00C8;
        case 'I': return 0x00CC; case 'O': return 0x00D2;
        case 'U': return 0x00D9; case 'a': return 0x00E0;
        case 'e': return 0x00E8; case 'i': return 0x00EC;
        case 'o': return 0x00F2; case 'u': return 0x00F9;
        }
        break;
    case 0x0301:
        switch (base) {
        case 'A': return 0x00C1; case 'C': return 0x0106;
        case 'E': return 0x00C9; case 'I': return 0x00CD;
        case 'L': return 0x0139; case 'N': return 0x0143;
        case 'O': return 0x00D3; case 'R': return 0x0154;
        case 'S': return 0x015A; case 'U': return 0x00DA;
        case 'Y': return 0x00DD; case 'Z': return 0x0179;
        case 'a': return 0x00E1; case 'c': return 0x0107;
        case 'e': return 0x00E9; case 'i': return 0x00ED;
        case 'l': return 0x013A; case 'n': return 0x0144;
        case 'o': return 0x00F3; case 'r': return 0x0155;
        case 's': return 0x015B; case 'u': return 0x00FA;
        case 'y': return 0x00FD; case 'z': return 0x017A;
        }
        break;
    case 0x0302:
        switch (base) {
        case 'A': return 0x00C2; case 'E': return 0x00CA;
        case 'I': return 0x00CE; case 'O': return 0x00D4;
        case 'U': return 0x00DB; case 'a': return 0x00E2;
        case 'e': return 0x00EA; case 'i': return 0x00EE;
        case 'o': return 0x00F4; case 'u': return 0x00FB;
        }
        break;
    case 0x0303:
        switch (base) {
        case 'A': return 0x00C3; case 'N': return 0x00D1;
        case 'O': return 0x00D5; case 'a': return 0x00E3;
        case 'n': return 0x00F1; case 'o': return 0x00F5;
        }
        break;
    case 0x0308:
        switch (base) {
        case 'A': return 0x00C4; case 'E': return 0x00CB;
        case 'I': return 0x00CF; case 'O': return 0x00D6;
        case 'U': return 0x00DC; case 'Y': return 0x0178;
        case 'a': return 0x00E4; case 'e': return 0x00EB;
        case 'i': return 0x00EF; case 'o': return 0x00F6;
        case 'u': return 0x00FC; case 'y': return 0x00FF;
        }
        break;
    case 0x030A:
        switch (base) {
        case 'A': return 0x00C5; case 'a': return 0x00E5;
        }
        break;
    case 0x0327:
        switch (base) {
        case 'C': return 0x00C7; case 'c': return 0x00E7;
        }
        break;
    }
    return 0;
}

static char *normalize_nfc_latin(const char *text) {
    const unsigned char *p = (const unsigned char *)text;
    const unsigned char *end = p + strlen(text);
    char *out = (char *)malloc((size_t)(end - p) * 2 + 1);
    if (!out) return NULL;

    size_t w = 0;
    while (p < end) {
        const unsigned char *cur = p;
        int cp = 0, len = 0;
        if (!utf8_decode_cp_len(p, end, &cp, &len)) break;
        p += len;

        if (p < end) {
            int mark = 0, mark_len = 0;
            utf8_decode_cp_len(p, end, &mark, &mark_len);
            int composed = compose_latin_mark(cp, mark);
            if (composed) {
                char tmp[4];
                int n = utf8_encode_full_cp(composed, tmp);
                for (int i = 0; i < n; i++) out[w++] = tmp[i];
                p += mark_len;
                continue;
            }
        }

        memcpy(out + w, cur, (size_t)len);
        w += (size_t)len;
    }

    out[w] = '\0';
    return out;
}

static int is_letter_cp(int cp) {
    if (is_ascii_alpha_cp(cp)) return 1;
    if (cp < 0x80 || is_space_cp(cp) || is_number_cp(cp) ||
        is_symbolish_cp(cp))
        return 0;
    return 1;
}

static int is_word_cp(int cp) {
    return is_letter_cp(cp) || is_number_cp(cp);
}

static int is_newline_cp(int cp) {
    return cp == '\n' || cp == '\r';
}

static int ascii_lower(int c) {
    return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

static int match_contraction(const unsigned char *p, const unsigned char *end) {
    if (p >= end || *p != '\'') return 0;
    int remain = (int)(end - p);
    if (remain >= 2) {
        int c1 = ascii_lower(p[1]);
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
            return 2;
    }
    if (remain >= 3) {
        int c1 = ascii_lower(p[1]);
        int c2 = ascii_lower(p[2]);
        if ((c1 == 'r' && c2 == 'e') ||
            (c1 == 'v' && c2 == 'e') ||
            (c1 == 'l' && c2 == 'l'))
            return 3;
    }
    return 0;
}

static int append_encoded_piece(const qwen_tokenizer_t *tok,
                                const char *piece, size_t piece_len,
                                int **ids, int *n_ids, int *cap) {
    if (piece_len == 0) return 0;

    char *tmp = (char *)malloc(piece_len + 1);
    if (!tmp) return -1;
    memcpy(tmp, piece, piece_len);
    tmp[piece_len] = '\0';

    char *mapped = text_to_bpe_unicode(tmp);
    free(tmp);
    if (!mapped) return -1;

    int *piece_ids = NULL;
    int n_piece = 0;
    if (encode_bpe_word(tok, mapped, &piece_ids, &n_piece) != 0) {
        free(mapped);
        free(piece_ids);
        return -1;
    }
    free(mapped);

    for (int i = 0; i < n_piece; i++) {
        if (append_id(ids, n_ids, cap, piece_ids[i]) != 0) {
            free(piece_ids);
            return -1;
        }
    }
    free(piece_ids);
    return 0;
}

static int encode_pretokenized(const qwen_tokenizer_t *tok, const char *text,
                               int **out_ids, int *out_n) {
    const unsigned char *start = (const unsigned char *)text;
    const unsigned char *p = start;
    const unsigned char *end = start + strlen(text);
    int *ids = NULL;
    int n_ids = 0, cap = 0;

    while (p < end) {
        const unsigned char *piece = p;
        int cp = 0, len = 0;
        if (!utf8_decode_cp_len(p, end, &cp, &len)) break;

        int clen = match_contraction(p, end);
        if (clen > 0) {
            p += clen;
        } else {
            const unsigned char *next = p + len;
            int next_cp = 0, next_len = 0;
            int has_next = utf8_decode_cp_len(next, end, &next_cp, &next_len);

            if (!is_newline_cp(cp) && !is_word_cp(cp) &&
                has_next && is_letter_cp(next_cp)) {
                p = next + next_len;
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (!is_letter_cp(c)) break;
                    p += l;
                }
            } else if (is_letter_cp(cp)) {
                p += len;
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (!is_letter_cp(c)) break;
                    p += l;
                }
            } else if (is_number_cp(cp)) {
                p += len;
            } else if (is_space_cp(cp)) {
                const unsigned char *q = p;
                int saw_newline = 0;
                while (q < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(q, end, &c, &l);
                    if (!is_space_cp(c)) break;
                    if (is_newline_cp(c)) saw_newline = 1;
                    q += l;
                    if (saw_newline) {
                        while (q < end) {
                            int c2 = 0, l2 = 0;
                            utf8_decode_cp_len(q, end, &c2, &l2);
                            if (!is_newline_cp(c2)) break;
                            q += l2;
                        }
                        break;
                    }
                }

                if (saw_newline) {
                    p = q;
                } else if (cp == ' ' && has_next && !is_space_cp(next_cp) &&
                           !is_word_cp(next_cp)) {
                    p = next + next_len;
                    while (p < end) {
                        int c = 0, l = 0;
                        utf8_decode_cp_len(p, end, &c, &l);
                        if (is_space_cp(c) || is_word_cp(c)) break;
                        p += l;
                    }
                    while (p < end) {
                        int c = 0, l = 0;
                        utf8_decode_cp_len(p, end, &c, &l);
                        if (!is_newline_cp(c)) break;
                        p += l;
                    }
                } else {
                    const unsigned char *last = p;
                    const unsigned char *q2 = p;
                    int count = 0;
                    while (q2 < end) {
                        int c = 0, l = 0;
                        utf8_decode_cp_len(q2, end, &c, &l);
                        if (!is_space_cp(c)) break;
                        last = q2;
                        q2 += l;
                        count++;
                    }
                    if (q2 < end && count > 1)
                        p = last;
                    else
                        p = q2;
                }
            } else {
                p += len;
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (is_space_cp(c) || is_word_cp(c)) break;
                    p += l;
                }
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (!is_newline_cp(c)) break;
                    p += l;
                }
            }
        }

        if (append_encoded_piece(tok, (const char *)piece, (size_t)(p - piece),
                                 &ids, &n_ids, &cap) != 0) {
            free(ids);
            return -1;
        }
    }

    *out_ids = ids;
    *out_n = n_ids;
    return 0;
}

/* Encode one mapped BPE unicode string to token IDs. */
static int encode_bpe_word(const qwen_tokenizer_t *tok, const char *mapped, int **out_ids, int *out_n) {
    *out_ids = NULL;
    *out_n = 0;
    if (!mapped || !*mapped) return 0;

    char **syms = NULL;
    int n_syms = 0;
    if (split_utf8_symbols(mapped, &syms, &n_syms) != 0) return -1;
    if (n_syms <= 0) {
        free(syms);
        return 0;
    }

    while (n_syms > 1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < n_syms - 1; i++) {
            int r = merge_rank(tok, syms[i], syms[i + 1]);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i < 0 || best_rank == INT_MAX) break;

        char *merged = str_concat2(syms[best_i], syms[best_i + 1]);
        if (!merged) {
            for (int i = 0; i < n_syms; i++) free(syms[i]);
            free(syms);
            return -1;
        }
        free(syms[best_i]);
        free(syms[best_i + 1]);
        syms[best_i] = merged;
        for (int j = best_i + 1; j < n_syms - 1; j++) syms[j] = syms[j + 1];
        n_syms--;
    }

    int *ids = NULL;
    int n_ids = 0, cap = 0;
    for (int i = 0; i < n_syms; i++) {
        int id = map_get((const str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, syms[i]);
        if (id < 0) {
            /* Should not happen with valid vocab + merges + byte-level mapping. */
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
        if (append_id(&ids, &n_ids, &cap, id) != 0) {
            for (int k = 0; k < n_syms; k++) free(syms[k]);
            free(syms);
            free(ids);
            return -1;
        }
    }

    for (int i = 0; i < n_syms; i++) free(syms[i]);
    free(syms);

    *out_ids = ids;
    *out_n = n_ids;
    return 0;
}

typedef struct {
    size_t off;
} bpe_sym_ref_t;

struct qwen_tokenizer_workspace {
    char *normalized;
    size_t normalized_cap;
    char *mapped;
    size_t mapped_cap;
    char *arena;
    size_t arena_cap;
    size_t arena_len;
    bpe_sym_ref_t *syms;
    int syms_cap;
    int *ids;
    int ids_cap;
    /* Integer-pair BPE merge scratch: token id, doubly-linked live order,
     * and the cached (rank, merged id) of the pair starting at each live
     * position. */
    int *bpe_ids;
    int *bpe_prv;
    int *bpe_nxt;
    int *bpe_rnk;
    int *bpe_mrg;
    int bpe_cap;
};

static int reserve_bpe(qwen_tokenizer_workspace_t *ws, int need) {
    if (ws->bpe_cap >= need) return 0;
    int new_cap = ws->bpe_cap ? ws->bpe_cap : 64;
    while (new_cap < need) {
        if (new_cap > INT_MAX / 2) return -1;
        new_cap *= 2;
    }
    int *p;
#define R(field) \
    p = (int *)realloc(ws->field, (size_t)new_cap * sizeof(int)); \
    if (!p) return -1; \
    ws->field = p;
    R(bpe_ids) R(bpe_prv) R(bpe_nxt) R(bpe_rnk) R(bpe_mrg)
#undef R
    ws->bpe_cap = new_cap;
    return 0;
}

static int reserve_bytes(char **buf, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    size_t new_cap = *cap ? *cap : 256;
    while (new_cap < need) {
        if (new_cap > SIZE_MAX / 2) return -1;
        new_cap *= 2;
    }
    char *p = (char *)realloc(*buf, new_cap);
    if (!p) return -1;
    *buf = p;
    *cap = new_cap;
    return 0;
}

static int reserve_ints(int **buf, int *cap, int need) {
    if (*cap >= need) return 0;
    int new_cap = *cap ? *cap : 64;
    while (new_cap < need) {
        if (new_cap > INT_MAX / 2) return -1;
        new_cap *= 2;
    }
    int *p = (int *)realloc(*buf, (size_t)new_cap * sizeof(int));
    if (!p) return -1;
    *buf = p;
    *cap = new_cap;
    return 0;
}

static int reserve_syms(qwen_tokenizer_workspace_t *ws, int need) {
    if (ws->syms_cap >= need) return 0;
    int new_cap = ws->syms_cap ? ws->syms_cap : 64;
    while (new_cap < need) {
        if (new_cap > INT_MAX / 2) return -1;
        new_cap *= 2;
    }
    bpe_sym_ref_t *p =
        (bpe_sym_ref_t *)realloc(ws->syms,
                                 (size_t)new_cap * sizeof(*ws->syms));
    if (!p) return -1;
    ws->syms = p;
    ws->syms_cap = new_cap;
    return 0;
}

static int arena_append(qwen_tokenizer_workspace_t *ws, const char *s,
                        size_t len, size_t *off) {
    if (ws->arena_len > SIZE_MAX - len - 1) return -1;
    size_t need = ws->arena_len + len + 1;
    if (reserve_bytes(&ws->arena, &ws->arena_cap, need) != 0) return -1;
    *off = ws->arena_len;
    memcpy(ws->arena + ws->arena_len, s, len);
    ws->arena_len += len;
    ws->arena[ws->arena_len++] = '\0';
    return 0;
}

static const char *arena_str(const qwen_tokenizer_workspace_t *ws,
                             bpe_sym_ref_t sym) {
    return ws->arena + sym.off;
}

static int normalize_nfc_latin_into(qwen_tokenizer_workspace_t *ws,
                                    const char *text, const char **out) {
    const unsigned char *p = (const unsigned char *)text;
    const unsigned char *end = p + strlen(text);
    size_t in_len = (size_t)(end - p);
    if (in_len > (SIZE_MAX - 1) / 2) return -1;
    if (reserve_bytes(&ws->normalized, &ws->normalized_cap,
                      in_len * 2 + 1) != 0)
        return -1;

    size_t w = 0;
    while (p < end) {
        const unsigned char *cur = p;
        int cp = 0, len = 0;
        if (!utf8_decode_cp_len(p, end, &cp, &len)) break;
        p += len;

        if (p < end) {
            int mark = 0, mark_len = 0;
            utf8_decode_cp_len(p, end, &mark, &mark_len);
            int composed = compose_latin_mark(cp, mark);
            if (composed) {
                char tmp[4];
                int n = utf8_encode_full_cp(composed, tmp);
                memcpy(ws->normalized + w, tmp, (size_t)n);
                w += (size_t)n;
                p += mark_len;
                continue;
            }
        }

        memcpy(ws->normalized + w, cur, (size_t)len);
        w += (size_t)len;
    }

    ws->normalized[w] = '\0';
    *out = ws->normalized;
    return 0;
}

static int text_to_bpe_unicode_into(qwen_tokenizer_workspace_t *ws,
                                    const char *text, size_t len,
                                    const char **out) {
    init_gpt2_mapping();
    if (len > (SIZE_MAX - 1) / 3) return -1;
    if (reserve_bytes(&ws->mapped, &ws->mapped_cap, len * 3 + 1) != 0)
        return -1;

    size_t w = 0;
    for (size_t i = 0; i < len; i++) {
        int cp = gpt2_byte_to_unicode[(unsigned char)text[i]];
        char tmp[4];
        int n = utf8_encode_cp(cp, tmp);
        memcpy(ws->mapped + w, tmp, (size_t)n);
        w += (size_t)n;
    }
    ws->mapped[w] = '\0';
    *out = ws->mapped;
    return 0;
}

static int output_append_id(int *out_ids, int out_cap, int *n_ids,
                            int *overflow, int id) {
    if (*n_ids < out_cap && out_ids)
        out_ids[*n_ids] = id;
    else
        *overflow = 1;
    (*n_ids)++;
    return 0;
}

static int encode_bpe_word_into_slow(const qwen_tokenizer_t *tok,
                                     qwen_tokenizer_workspace_t *ws,
                                     const char *mapped,
                                     int *out_ids, int out_cap,
                                     int *n_ids, int *overflow) {
    if (!mapped || !*mapped) return 0;

    int n_syms = 0;
    for (const unsigned char *p = (const unsigned char *)mapped; *p; ) {
        int l = utf8_char_len(*p);
        p += l;
        n_syms++;
    }
    if (n_syms <= 0) return 0;
    if (reserve_syms(ws, n_syms) != 0) return -1;

    ws->arena_len = 0;
    int i = 0;
    for (const unsigned char *p = (const unsigned char *)mapped; *p; ) {
        int l = utf8_char_len(*p);
        size_t off = 0;
        if (arena_append(ws, (const char *)p, (size_t)l, &off) != 0)
            return -1;
        ws->syms[i++].off = off;
        p += l;
    }

    while (n_syms > 1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int j = 0; j < n_syms - 1; j++) {
            int r = merge_rank(tok, arena_str(ws, ws->syms[j]),
                               arena_str(ws, ws->syms[j + 1]));
            if (r < best_rank) {
                best_rank = r;
                best_i = j;
            }
        }
        if (best_i < 0 || best_rank == INT_MAX) break;

        const char *a = arena_str(ws, ws->syms[best_i]);
        const char *b = arena_str(ws, ws->syms[best_i + 1]);
        size_t la = strlen(a), lb = strlen(b);
        if (la > SIZE_MAX - lb) return -1;
        if (reserve_bytes(&ws->arena, &ws->arena_cap,
                          ws->arena_len + la + lb + 1) != 0)
            return -1;

        a = arena_str(ws, ws->syms[best_i]);
        b = arena_str(ws, ws->syms[best_i + 1]);
        size_t off = ws->arena_len;
        memcpy(ws->arena + ws->arena_len, a, la);
        ws->arena_len += la;
        memcpy(ws->arena + ws->arena_len, b, lb);
        ws->arena_len += lb;
        ws->arena[ws->arena_len++] = '\0';
        ws->syms[best_i].off = off;
        for (int j = best_i + 1; j < n_syms - 1; j++)
            ws->syms[j] = ws->syms[j + 1];
        n_syms--;
    }

    for (int j = 0; j < n_syms; j++) {
        int id = map_get((const str_int_entry_t *)tok->vocab_map,
                         tok->vocab_map_cap, arena_str(ws, ws->syms[j]));
        if (id < 0) return -1;
        output_append_id(out_ids, out_cap, n_ids, overflow, id);
    }
    return 0;
}

/* Integer-pair BPE merge: token ids in a doubly-linked array with cached
 * pair ranks; each merge updates only the two adjacent pairs. Falls back to
 * the string-based loop when the int tables are unavailable. Identical
 * output: initial ids are the byte-level single-char token ids and every
 * merge result is the vocab id of the concatenated pair. */
static int encode_bpe_word_into(const qwen_tokenizer_t *tok,
                                qwen_tokenizer_workspace_t *ws,
                                const char *mapped,
                                int *out_ids, int out_cap,
                                int *n_ids, int *overflow) {
    if (!mapped || !*mapped) return 0;
    if (!tok->int_merges || !tok->cp_to_id)
        return encode_bpe_word_into_slow(tok, ws, mapped, out_ids, out_cap,
                                         n_ids, overflow);

    int n = 0;
    for (const unsigned char *p = (const unsigned char *)mapped; *p; ) {
        p += utf8_char_len(*p);
        n++;
    }
    if (n <= 0) return 0;
    if (reserve_bpe(ws, n) != 0) return -1;

    int *ids = ws->bpe_ids, *prv = ws->bpe_prv, *nxt = ws->bpe_nxt;
    int *rnk = ws->bpe_rnk, *mrg = ws->bpe_mrg;

    int i = 0;
    for (const unsigned char *p = (const unsigned char *)mapped; *p; ) {
        int len = utf8_char_len(*p);
        int cp;
        if (len == 1) cp = *p;
        else if (len == 2) cp = ((*p & 0x1F) << 6) | (p[1] & 0x3F);
        else cp = 512; /* byte-level alphabet never exceeds two bytes */
        int id = cp < 512 ? tok->cp_to_id[cp] : -1;
        if (id < 0)
            return encode_bpe_word_into_slow(tok, ws, mapped, out_ids,
                                             out_cap, n_ids, overflow);
        ids[i] = id;
        prv[i] = i - 1;
        nxt[i] = i + 1 < n ? i + 1 : -1;
        i++;
        p += len;
    }

    for (int j = 0; j < n - 1; j++)
        rnk[j] = pair_merge_lookup(tok, ids[j], ids[j + 1], &mrg[j]);
    if (n > 0) rnk[n - 1] = INT_MAX;

    for (;;) {
        int best = -1, best_rank = INT_MAX;
        for (int j = 0; j != -1 && nxt[j] != -1; j = nxt[j]) {
            if (rnk[j] < best_rank) {
                best_rank = rnk[j];
                best = j;
            }
        }
        if (best < 0) break;

        ids[best] = mrg[best];
        int dead = nxt[best];
        nxt[best] = nxt[dead];
        if (nxt[dead] != -1) prv[nxt[dead]] = best;

        if (prv[best] != -1)
            rnk[prv[best]] = pair_merge_lookup(tok, ids[prv[best]], ids[best],
                                               &mrg[prv[best]]);
        rnk[best] = nxt[best] != -1
            ? pair_merge_lookup(tok, ids[best], ids[nxt[best]], &mrg[best])
            : INT_MAX;
    }

    for (int j = 0; j != -1; j = nxt[j])
        output_append_id(out_ids, out_cap, n_ids, overflow, ids[j]);
    return 0;
}

static int append_encoded_piece_into(const qwen_tokenizer_t *tok,
                                     qwen_tokenizer_workspace_t *ws,
                                     const char *piece, size_t piece_len,
                                     int *out_ids, int out_cap,
                                     int *n_ids, int *overflow) {
    if (piece_len == 0) return 0;

    const char *mapped = NULL;
    if (text_to_bpe_unicode_into(ws, piece, piece_len, &mapped) != 0)
        return -1;
    return encode_bpe_word_into(tok, ws, mapped, out_ids, out_cap, n_ids,
                                overflow);
}

static int encode_pretokenized_into(const qwen_tokenizer_t *tok,
                                    qwen_tokenizer_workspace_t *ws,
                                    const char *text,
                                    int *out_ids, int out_cap,
                                    int *out_n) {
    const unsigned char *start = (const unsigned char *)text;
    const unsigned char *p = start;
    const unsigned char *end = start + strlen(text);
    int n_ids = 0;
    int overflow = 0;

    while (p < end) {
        const unsigned char *piece = p;
        int cp = 0, len = 0;
        if (!utf8_decode_cp_len(p, end, &cp, &len)) break;

        int clen = match_contraction(p, end);
        if (clen > 0) {
            p += clen;
        } else {
            const unsigned char *next = p + len;
            int next_cp = 0, next_len = 0;
            int has_next = utf8_decode_cp_len(next, end, &next_cp, &next_len);

            if (!is_newline_cp(cp) && !is_word_cp(cp) &&
                has_next && is_letter_cp(next_cp)) {
                p = next + next_len;
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (!is_letter_cp(c)) break;
                    p += l;
                }
            } else if (is_letter_cp(cp)) {
                p += len;
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (!is_letter_cp(c)) break;
                    p += l;
                }
            } else if (is_number_cp(cp)) {
                p += len;
            } else if (is_space_cp(cp)) {
                const unsigned char *q = p;
                int saw_newline = 0;
                while (q < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(q, end, &c, &l);
                    if (!is_space_cp(c)) break;
                    if (is_newline_cp(c)) saw_newline = 1;
                    q += l;
                    if (saw_newline) {
                        while (q < end) {
                            int c2 = 0, l2 = 0;
                            utf8_decode_cp_len(q, end, &c2, &l2);
                            if (!is_newline_cp(c2)) break;
                            q += l2;
                        }
                        break;
                    }
                }

                if (saw_newline) {
                    p = q;
                } else if (cp == ' ' && has_next && !is_space_cp(next_cp) &&
                           !is_word_cp(next_cp)) {
                    p = next + next_len;
                    while (p < end) {
                        int c = 0, l = 0;
                        utf8_decode_cp_len(p, end, &c, &l);
                        if (is_space_cp(c) || is_word_cp(c)) break;
                        p += l;
                    }
                    while (p < end) {
                        int c = 0, l = 0;
                        utf8_decode_cp_len(p, end, &c, &l);
                        if (!is_newline_cp(c)) break;
                        p += l;
                    }
                } else {
                    const unsigned char *last = p;
                    const unsigned char *q2 = p;
                    int count = 0;
                    while (q2 < end) {
                        int c = 0, l = 0;
                        utf8_decode_cp_len(q2, end, &c, &l);
                        if (!is_space_cp(c)) break;
                        last = q2;
                        q2 += l;
                        count++;
                    }
                    if (q2 < end && count > 1)
                        p = last;
                    else
                        p = q2;
                }
            } else {
                p += len;
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (is_space_cp(c) || is_word_cp(c)) break;
                    p += l;
                }
                while (p < end) {
                    int c = 0, l = 0;
                    utf8_decode_cp_len(p, end, &c, &l);
                    if (!is_newline_cp(c)) break;
                    p += l;
                }
            }
        }

        if (append_encoded_piece_into(tok, ws, (const char *)piece,
                                      (size_t)(p - piece), out_ids, out_cap,
                                      &n_ids, &overflow) != 0)
            return -1;
    }

    *out_n = n_ids;
    return overflow ? -2 : 0;
}

static int derive_merges_path(const char *vocab_path, char *out_path, size_t out_cap) {
    const char *slash = strrchr(vocab_path, '/');
    if (!slash) {
        if (snprintf(out_path, out_cap, "merges.txt") >= (int)out_cap) return -1;
        return 0;
    }
    size_t dir_len = (size_t)(slash - vocab_path);
    if (dir_len + 12 >= out_cap) return -1;
    memcpy(out_path, vocab_path, dir_len);
    out_path[dir_len] = '\0';
    snprintf(out_path + dir_len, out_cap - dir_len, "/merges.txt");
    return 0;
}

static void trim_newline(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
}

static int parse_merge_pair(char *line, char **a, char **b) {
    *a = NULL;
    *b = NULL;
    trim_newline(line);
    if (line[0] == '\0') return -1;
    if (line[0] == '#') return -1;

    char *sp = strchr(line, ' ');
    if (!sp) return -1;
    *sp = '\0';
    char *p2 = sp + 1;
    while (*p2 == ' ') p2++;
    if (*line == '\0' || *p2 == '\0') return -1;
    *a = line;
    *b = p2;
    return 0;
}

static int load_merges_map(qwen_tokenizer_t *tok, const char *merges_path) {
    FILE *f = fopen(merges_path, "rb");
    if (!f) return -1;

    int n_pairs = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        char *a = NULL, *b = NULL;
        if (parse_merge_pair(line, &a, &b) == 0) n_pairs++;
    }
    if (n_pairs <= 0) {
        fclose(f);
        return -1;
    }

    tok->merge_map_cap = next_pow2(n_pairs * 2);
    tok->merge_map = calloc((size_t)tok->merge_map_cap, sizeof(str_int_entry_t));
    if (!tok->merge_map) {
        fclose(f);
        tok->merge_map_cap = 0;
        return -1;
    }

    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    tok->int_merges_cap = next_pow2(n_pairs * 2);
    pair_merge_entry_t *pm = (pair_merge_entry_t *)
        calloc((size_t)tok->int_merges_cap, sizeof(pair_merge_entry_t));
    if (pm) {
        for (int i = 0; i < tok->int_merges_cap; i++) {
            pm[i].key = UINT64_MAX;
            pm[i].rank = INT_MAX;
            pm[i].merged_id = -1;
        }
    }
    tok->int_merges = pm;

    int rank = 0;
    while (fgets(line, sizeof(line), f)) {
        char *a = NULL, *b = NULL;
        if (parse_merge_pair(line, &a, &b) != 0) continue;

        size_t la = strlen(a), lb = strlen(b);
        char *key = (char *)malloc(la + 1 + lb + 1);
        if (!key) continue;
        memcpy(key, a, la);
        key[la] = ' ';
        memcpy(key + la + 1, b, lb);
        key[la + 1 + lb] = '\0';

        if (tok->int_merges) {
            /* Resolve (a, b) and the merged token a+b to vocab ids; if any
             * merge is unresolvable, drop the whole int table and keep the
             * string-based loop as the only path. */
            const str_int_entry_t *vm = (const str_int_entry_t *)tok->vocab_map;
            int left = map_get(vm, tok->vocab_map_cap, a);
            int right = map_get(vm, tok->vocab_map_cap, b);
            char ab[8192];
            int merged = -1;
            if (la + lb < sizeof(ab)) {
                memcpy(ab, a, la);
                memcpy(ab + la, b, lb);
                ab[la + lb] = '\0';
                merged = map_get(vm, tok->vocab_map_cap, ab);
            }
            if (left < 0 || right < 0 || merged < 0) {
                free(tok->int_merges);
                tok->int_merges = NULL;
                tok->int_merges_cap = 0;
            } else {
                pair_map_insert((pair_merge_entry_t *)tok->int_merges,
                                tok->int_merges_cap,
                                ((uint64_t)(uint32_t)left << 32) |
                                    (uint32_t)right,
                                rank, merged);
            }
        }
        (void)map_insert_owned((str_int_entry_t *)tok->merge_map,
                               tok->merge_map_cap, &key, rank);
        free(key);
        rank++;
    }
    fclose(f);

    /* Byte-level unicode codepoint -> single-char token id table. */
    if (tok->int_merges) {
        init_gpt2_mapping();
        tok->cp_to_id = (int *)malloc(512 * sizeof(int));
        if (tok->cp_to_id) {
            for (int i = 0; i < 512; i++) tok->cp_to_id[i] = -1;
            for (int byte = 0; byte < 256 && tok->cp_to_id; byte++) {
                int cp = gpt2_byte_to_unicode[byte];
                char buf[4];
                int n;
                if (cp < 0x80) { buf[0] = (char)cp; n = 1; }
                else { buf[0] = (char)(0xC0 | (cp >> 6));
                       buf[1] = (char)(0x80 | (cp & 0x3F)); n = 2; }
                buf[n] = '\0';
                int id = map_get((const str_int_entry_t *)tok->vocab_map,
                                 tok->vocab_map_cap, buf);
                if (id < 0) {
                    free(tok->cp_to_id);
                    tok->cp_to_id = NULL;
                } else {
                    tok->cp_to_id[cp] = id;
                }
            }
        }
    }
    return 0;
}

/* ========================================================================
 * Public API
 * ======================================================================== */

qwen_tokenizer_t *qwen_tokenizer_load(const char *vocab_json_path) {
    FILE *f = fopen(vocab_json_path, "rb");
    if (!f) {
        fprintf(stderr, "qwen_tokenizer_load: cannot open %s\n", vocab_json_path);
        return NULL;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long file_size = ftell(f);
    if (file_size < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    char *json = (char *)malloc((size_t)file_size + 1);
    if (!json || fread(json, 1, (size_t)file_size, f) != (size_t)file_size) {
        fclose(f);
        free(json);
        return NULL;
    }
    fclose(f);
    json[file_size] = '\0';

    int max_id = 0;
    const char *p = json;
    skip_ws(&p);
    if (*p != '{') {
        free(json);
        return NULL;
    }
    p++;

    const char *saved = p;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char key[4096] = {0};
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);
        if (id > max_id) max_id = (int)id;
    }

    int vocab_size = max_id + 1;
    qwen_tokenizer_t *tok = (qwen_tokenizer_t *)calloc(1, sizeof(qwen_tokenizer_t));
    if (!tok) {
        free(json);
        return NULL;
    }
    tok->vocab_size = vocab_size;
    tok->id_to_text = (char **)calloc((size_t)vocab_size, sizeof(char *));
    tok->id_to_bpe = (char **)calloc((size_t)vocab_size, sizeof(char *));
    if (!tok->id_to_text || !tok->id_to_bpe) {
        qwen_tokenizer_free(tok);
        free(json);
        return NULL;
    }

    p = saved;
    while (*p && *p != '}') {
        skip_ws(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char key[4096] = {0};
        if (parse_json_string(&p, key, sizeof(key)) != 0) break;
        skip_ws(&p);
        if (*p != ':') break;
        p++;
        int64_t id = parse_json_int(&p);

        if (id >= 0 && id < vocab_size) {
            free(tok->id_to_bpe[id]);
            free(tok->id_to_text[id]);
            tok->id_to_bpe[id] = strdup(key);
            tok->id_to_text[id] = decode_gpt2_token(key);
        }
    }
    free(json);

    int n_vocab_entries = 0;
    for (int i = 0; i < vocab_size; i++) if (tok->id_to_bpe[i]) n_vocab_entries++;
    tok->vocab_map_cap = next_pow2(n_vocab_entries * 2 + 1);
    tok->vocab_map = calloc((size_t)tok->vocab_map_cap, sizeof(str_int_entry_t));
    if (!tok->vocab_map) {
        qwen_tokenizer_free(tok);
        return NULL;
    }
    for (int i = 0; i < vocab_size; i++) {
        if (tok->id_to_bpe[i]) {
            map_insert((str_int_entry_t *)tok->vocab_map, tok->vocab_map_cap, tok->id_to_bpe[i], i);
        }
    }

    char merges_path[1024];
    if (derive_merges_path(vocab_json_path, merges_path, sizeof(merges_path)) == 0) {
        if (load_merges_map(tok, merges_path) != 0 && qwen_verbose >= 2) {
            fprintf(stderr, "Tokenizer: merges not loaded from %s (encoding falls back to byte-level)\n",
                    merges_path);
        }
    }

    return tok;
}

const char *qwen_tokenizer_decode(const qwen_tokenizer_t *tok, int token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return "";
    return tok->id_to_text[token_id] ? tok->id_to_text[token_id] : "";
}

int *qwen_tokenizer_encode(const qwen_tokenizer_t *tok, const char *text, int *out_n_tokens) {
    if (out_n_tokens) *out_n_tokens = 0;
    if (!tok || !text || text[0] == '\0') return NULL;

    char *normalized = normalize_nfc_latin(text);
    if (!normalized) return NULL;

    int *ids = NULL;
    int n_ids = 0;
    if (encode_pretokenized(tok, normalized, &ids, &n_ids) != 0) {
        free(normalized);
        free(ids);
        return NULL;
    }
    free(normalized);

    if (out_n_tokens) *out_n_tokens = n_ids;
    return ids;
}

qwen_tokenizer_workspace_t *qwen_tokenizer_workspace_new(void) {
    return (qwen_tokenizer_workspace_t *)calloc(1,
                                               sizeof(qwen_tokenizer_workspace_t));
}

void qwen_tokenizer_workspace_free(qwen_tokenizer_workspace_t *ws) {
    if (!ws) return;
    free(ws->normalized);
    free(ws->mapped);
    free(ws->arena);
    free(ws->syms);
    free(ws->ids);
    free(ws->bpe_ids);
    free(ws->bpe_prv);
    free(ws->bpe_nxt);
    free(ws->bpe_rnk);
    free(ws->bpe_mrg);
    free(ws);
}

int qwen_tokenizer_encode_into(const qwen_tokenizer_t *tok,
                               qwen_tokenizer_workspace_t *ws,
                               const char *text,
                               int *out_ids, int out_cap,
                               int *out_n_tokens) {
    if (out_n_tokens) *out_n_tokens = 0;
    if (!tok || !ws || !text || out_cap < 0) return -1;
    if (text[0] == '\0') return 0;

    const char *normalized = NULL;
    if (normalize_nfc_latin_into(ws, text, &normalized) != 0)
        return -1;

    int n_ids = 0;
    int rc = encode_pretokenized_into(tok, ws, normalized, out_ids, out_cap,
                                      &n_ids);
    if (out_n_tokens) *out_n_tokens = n_ids;
    return rc;
}

int *qwen_tokenizer_encode_with_workspace(const qwen_tokenizer_t *tok,
                                          qwen_tokenizer_workspace_t *ws,
                                          const char *text,
                                          int *out_n_tokens) {
    if (out_n_tokens) *out_n_tokens = 0;
    if (!tok || !ws || !text || text[0] == '\0') return NULL;

    size_t len = strlen(text);
    size_t cap_sz = len > (SIZE_MAX - 8) / 2 ? SIZE_MAX : len * 2 + 8;
    if (cap_sz > (size_t)INT_MAX) return NULL;
    int cap = (int)cap_sz;

    if (reserve_ints(&ws->ids, &ws->ids_cap, cap) != 0)
        return NULL;

    int n_ids = 0;
    int rc = qwen_tokenizer_encode_into(tok, ws, text, ws->ids, ws->ids_cap,
                                        &n_ids);
    if (rc == -2) {
        if (reserve_ints(&ws->ids, &ws->ids_cap, n_ids) != 0)
            return NULL;
        rc = qwen_tokenizer_encode_into(tok, ws, text, ws->ids, ws->ids_cap,
                                        &n_ids);
    }
    if (rc != 0 || n_ids <= 0) return NULL;

    int *ids = (int *)malloc((size_t)n_ids * sizeof(int));
    if (!ids) return NULL;
    memcpy(ids, ws->ids, (size_t)n_ids * sizeof(int));
    if (out_n_tokens) *out_n_tokens = n_ids;
    return ids;
}

void qwen_tokenizer_free(qwen_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->id_to_text) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_text[i]);
        free(tok->id_to_text);
    }
    if (tok->id_to_bpe) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->id_to_bpe[i]);
        free(tok->id_to_bpe);
    }

    if (tok->merge_map) {
        str_int_entry_t *m = (str_int_entry_t *)tok->merge_map;
        for (int i = 0; i < tok->merge_map_cap; i++) free(m[i].key);
        free(tok->merge_map);
    }
    free(tok->vocab_map);
    free(tok->int_merges);
    free(tok->cp_to_id);
    free(tok);
}
