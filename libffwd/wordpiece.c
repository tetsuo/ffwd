/*
 * tok_wp_tokenizer.c - BERT-family WordPiece tokenizer.
 *
 * Pipeline (matching transformers' BertTokenizer):
 *   BasicTokenizer: clean control/whitespace -> space CJK ideographs ->
 *     whitespace split -> (lowercase + strip accents) -> split on punctuation
 *   WordPiece: greedy longest-match-first against vocab.txt, continuation
 *     pieces marked "##", whole token -> [UNK] when any piece fails to match.
 *
 * Unicode handling is full and exact against transformers, driven by generated
 * tables (tok_wp_unicode.h): punctuation and control by Unicode category,
 * whitespace by the complete Zs set, the CJK split by the BERT ideograph ranges,
 * and the do_lower_case fold by a lowercase+NFD+strip-marks table (with Hangul
 * decomposed by formula). Cased and uncased multilingual encoders (e.g.
 * bert-base-multilingual-cased / -uncased) match Hugging Face exactly, including
 * Greek, Cyrillic, Vietnamese and Korean. The fold is per-codepoint, matching
 * transformers, which lowercases character by character (no context-sensitive
 * Greek final sigma).
 */

#include "wordpiece.h"

#include "unicode.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ===================== string -> int open-addressing map ===================== */

typedef struct {
    const char *key; /* points into tok->id_to_token; not owned */
    int val;
} wp_entry_t;

static uint64_t fnv1a_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= (uint64_t)*p;
        h *= 1099511628211ULL;
    }
    return h;
}

static int next_pow2(int x) {
    int p = 1;
    while (p < x)
        p <<= 1;
    return p;
}

static void map_put(wp_entry_t *map, int cap, const char *key, int val) {
    uint64_t mask = (uint64_t)cap - 1;
    int pos = (int)(fnv1a_hash(key) & mask);
    while (map[pos].key) {
        if (!strcmp(map[pos].key, key)) {
            map[pos].val = val; /* keep the first (lowest) id, like a dict insert */
            return;
        }
        pos = (int)((pos + 1) & (int)mask);
    }
    map[pos].key = key;
    map[pos].val = val;
}

static int map_get(const wp_entry_t *map, int cap, const char *key) {
    if (!map)
        return -1;
    uint64_t mask = (uint64_t)cap - 1;
    int pos = (int)(fnv1a_hash(key) & mask);
    while (map[pos].key) {
        if (!strcmp(map[pos].key, key))
            return map[pos].val;
        pos = (int)((pos + 1) & (int)mask);
    }
    return -1;
}

/* ============================== UTF-8 helpers ================================ */

/* Decode one codepoint. Returns bytes consumed (>=1); on a malformed lead byte
 * emits U+FFFD and consumes one byte so iteration always advances. */
static int utf8_decode(const unsigned char *p, const unsigned char *end, uint32_t *cp) {
    unsigned char c = p[0];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    int n;
    uint32_t v;
    if ((c & 0xE0) == 0xC0) {
        n = 2;
        v = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
        n = 3;
        v = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
        n = 4;
        v = c & 0x07;
    } else {
        *cp = 0xFFFD;
        return 1;
    }
    if (p + n > end) {
        *cp = 0xFFFD;
        return 1;
    }
    for (int i = 1; i < n; i++) {
        if ((p[i] & 0xC0) != 0x80) {
            *cp = 0xFFFD;
            return 1;
        }
        v = (v << 6) | (p[i] & 0x3F);
    }
    *cp = v;
    return n;
}

static int utf8_encode(uint32_t cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* ========================== Unicode classification ========================== */

static int wp_is_whitespace(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r')
        return 1;
    return cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

/* Binary search a sorted, non-overlapping [lo, hi] range table. */
static int wp_in_ranges(uint32_t cp, const wp_range_t *r, int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < r[mid].lo)
            hi = mid - 1;
        else if (cp > r[mid].hi)
            lo = mid + 1;
        else
            return 1;
    }
    return 0;
}

/* transformers' _is_control: tab/newline/carriage-return are whitespace, not
 * control (they fall through to wp_is_whitespace); everything else is Unicode
 * category Cc, Cf, or Co. */
static int wp_is_control(uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r')
        return 0;
    return wp_in_ranges(cp, WP_CONTROL_RANGES, WP_CONTROL_RANGES_N);
}

/* transformers' _is_punctuation: the ASCII non-alphanumeric set plus every
 * Unicode category P* codepoint, both folded into the generated table. */
static int wp_is_punct(uint32_t cp) { return wp_in_ranges(cp, WP_PUNCT_RANGES, WP_PUNCT_RANGES_N); }

static int wp_is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

/* Hangul syllable decomposition constants (the standard algorithm). Hangul
 * syllables are not cased, so their do_lower_case fold is the canonical NFD into
 * leading (L), vowel (V), and optional trailing (T) jamo - none combining marks
 * - which keeps them out of the generated WP_FOLD table. */
enum {
    WP_HANGUL_S = 0xAC00,
    WP_HANGUL_L = 0x1100,
    WP_HANGUL_V = 0x1161,
    WP_HANGUL_T = 0x11A7,
    WP_HANGUL_TCOUNT = 28,
    WP_HANGUL_NCOUNT = 588,
    WP_HANGUL_SCOUNT = 11172
};

/* Apply the do_lower_case fold (Python str.lower then NFD then drop combining
 * marks) to one codepoint, writing 0..3 output codepoints to out and the count
 * to *n_out. 0 outputs means the codepoint is a dropped mark. Hangul decomposes
 * by formula; everything else is the generated WP_FOLD table or an identity. */
static void wp_fold(uint32_t cp, uint32_t out[3], int *n_out) {
    if (cp >= WP_HANGUL_S && cp < WP_HANGUL_S + WP_HANGUL_SCOUNT) {
        int s = (int)(cp - WP_HANGUL_S);
        out[0] = (uint32_t)(WP_HANGUL_L + s / WP_HANGUL_NCOUNT);
        out[1] = (uint32_t)(WP_HANGUL_V + (s % WP_HANGUL_NCOUNT) / WP_HANGUL_TCOUNT);
        int t = s % WP_HANGUL_TCOUNT;
        if (t == 0) {
            *n_out = 2;
        } else {
            out[2] = (uint32_t)(WP_HANGUL_T + t);
            *n_out = 3;
        }
        return;
    }
    int lo = 0, hi = WP_FOLD_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < WP_FOLD[mid].cp) {
            hi = mid - 1;
        } else if (cp > WP_FOLD[mid].cp) {
            lo = mid + 1;
        } else {
            int n = WP_FOLD[mid].n;
            for (int i = 0; i < n; i++)
                out[i] = WP_FOLD[mid].out[i];
            *n_out = n;
            return;
        }
    }
    out[0] = cp;
    *n_out = 1;
}

/* =============================== workspace ================================== */

struct tok_wp_workspace {
    uint32_t *cps; /* cleaned + CJK-spaced codepoints of the whole input */
    int cps_cap;
    uint32_t *norm; /* one word's normalized codepoints */
    int norm_cap;
    char *wbuf; /* one sub-word's UTF-8 bytes */
    int wbuf_cap;
    int *boff; /* byte offset of each codepoint within wbuf (+1 sentinel) */
    int boff_cap;
    char *look; /* "##" + candidate substring, for vocab lookup */
    int look_cap;
    int *ids; /* output ids */
    int ids_cap;
};

static int grow_u32(uint32_t **buf, int *cap, int need) {
    if (*cap >= need)
        return 0;
    int n = *cap ? *cap : 64;
    while (n < need)
        n *= 2;
    uint32_t *p = (uint32_t *)realloc(*buf, (size_t)n * sizeof(uint32_t));
    if (!p)
        return -1;
    *buf = p;
    *cap = n;
    return 0;
}

static int grow_int(int **buf, int *cap, int need) {
    if (*cap >= need)
        return 0;
    int n = *cap ? *cap : 64;
    while (n < need)
        n *= 2;
    int *p = (int *)realloc(*buf, (size_t)n * sizeof(int));
    if (!p)
        return -1;
    *buf = p;
    *cap = n;
    return 0;
}

static int grow_char(char **buf, int *cap, int need) {
    if (*cap >= need)
        return 0;
    int n = *cap ? *cap : 256;
    while (n < need)
        n *= 2;
    char *p = (char *)realloc(*buf, (size_t)n);
    if (!p)
        return -1;
    *buf = p;
    *cap = n;
    return 0;
}

tok_wp_workspace_t *tok_wp_workspace_new(void) {
    return (tok_wp_workspace_t *)calloc(1, sizeof(tok_wp_workspace_t));
}

void tok_wp_workspace_free(tok_wp_workspace_t *ws) {
    if (!ws)
        return;
    free(ws->cps);
    free(ws->norm);
    free(ws->wbuf);
    free(ws->boff);
    free(ws->look);
    free(ws->ids);
    free(ws);
}

/* ============================== WordPiece core ============================== */

/* Append the WordPiece ids of one BasicTokenizer sub-word (codepoints
 * word[0..n)) to ws->ids[*n_out]. The whole sub-word maps to [UNK] when it is
 * too long or any greedy piece misses the vocab. Returns 0 / -1 on OOM. */
static int
tok_wp_subword(const tok_wp_t *tok, tok_wp_workspace_t *ws, const uint32_t *word, int n, int *n_out) {
    if (n == 0)
        return 0;
    if (n > tok->max_chars_per_word) {
        if (grow_int(&ws->ids, &ws->ids_cap, *n_out + 1) != 0)
            return -1;
        ws->ids[(*n_out)++] = tok->unk_id;
        return 0;
    }

    /* Materialize the sub-word as UTF-8 with per-codepoint byte offsets. */
    if (grow_int(&ws->boff, &ws->boff_cap, n + 1) != 0 ||
        grow_char(&ws->wbuf, &ws->wbuf_cap, n * 4 + 1) != 0)
        return -1;
    int bytes = 0;
    for (int i = 0; i < n; i++) {
        ws->boff[i] = bytes;
        bytes += utf8_encode(word[i], ws->wbuf + bytes);
    }
    ws->boff[n] = bytes;

    int start_count = *n_out;
    int start = 0;
    int bad = 0;
    while (start < n) {
        int end = n;
        int matched = -1;
        int matched_end = start;
        while (start < end) {
            int seg = ws->boff[end] - ws->boff[start];
            int lp = 0;
            if (grow_char(&ws->look, &ws->look_cap, seg + 3) != 0)
                return -1;
            if (start > 0) {
                ws->look[lp++] = '#';
                ws->look[lp++] = '#';
            }
            memcpy(ws->look + lp, ws->wbuf + ws->boff[start], (size_t)seg);
            ws->look[lp + seg] = '\0';
            int id = map_get((const wp_entry_t *)tok->vocab_map, tok->vocab_map_cap, ws->look);
            if (id >= 0) {
                matched = id;
                matched_end = end;
                break;
            }
            end--;
        }
        if (matched < 0) {
            bad = 1;
            break;
        }
        if (grow_int(&ws->ids, &ws->ids_cap, *n_out + 1) != 0)
            return -1;
        ws->ids[(*n_out)++] = matched;
        start = matched_end;
    }
    if (bad) {
        *n_out = start_count; /* discard partial pieces; emit one [UNK] */
        if (grow_int(&ws->ids, &ws->ids_cap, *n_out + 1) != 0)
            return -1;
        ws->ids[(*n_out)++] = tok->unk_id;
    }
    return 0;
}

/* Run the full BasicTokenizer + WordPiece pipeline into ws->ids. Returns the
 * token count, or -1 on allocation failure. */
static int tok_wp_run(const tok_wp_t *tok, tok_wp_workspace_t *ws, const char *text) {
    /* clean_text + tokenize_chinese_chars, in one pass over the input. */
    const unsigned char *p = (const unsigned char *)text;
    const unsigned char *end = p + strlen(text);
    int n_cps = 0;
    while (p < end) {
        uint32_t cp;
        p += utf8_decode(p, end, &cp);
        if (cp == 0 || cp == 0xFFFD || wp_is_control(cp))
            continue;
        if (wp_is_whitespace(cp)) {
            if (grow_u32(&ws->cps, &ws->cps_cap, n_cps + 1) != 0)
                return -1;
            ws->cps[n_cps++] = ' ';
            continue;
        }
        int extra = wp_is_cjk(cp) ? 3 : 1;
        if (grow_u32(&ws->cps, &ws->cps_cap, n_cps + extra) != 0)
            return -1;
        if (extra == 3) {
            ws->cps[n_cps++] = ' ';
            ws->cps[n_cps++] = cp;
            ws->cps[n_cps++] = ' ';
        } else {
            ws->cps[n_cps++] = cp;
        }
    }

    int n_out = 0;
    int i = 0;
    while (i < n_cps) {
        if (ws->cps[i] == ' ') {
            i++;
            continue;
        }
        int wstart = i;
        while (i < n_cps && ws->cps[i] != ' ')
            i++;
        int wlen = i - wstart;

        /* lowercase + strip accents; the fold can drop a codepoint (combining
         * mark) or expand it to up to 3 (Hangul, a few ligatures). */
        int m = 0;
        if (grow_u32(&ws->norm, &ws->norm_cap, tok->do_lower_case ? wlen * 3 : wlen) != 0)
            return -1;
        for (int k = 0; k < wlen; k++) {
            uint32_t cp = ws->cps[wstart + k];
            if (tok->do_lower_case) {
                uint32_t out[3];
                int no = 0;
                wp_fold(cp, out, &no);
                for (int j = 0; j < no; j++)
                    ws->norm[m++] = out[j];
            } else {
                ws->norm[m++] = cp;
            }
        }

        /* split_on_punc, then WordPiece each piece. */
        int run = 0;
        for (int k = 0; k < m; k++) {
            if (wp_is_punct(ws->norm[k])) {
                if (run > 0 && tok_wp_subword(tok, ws, ws->norm + (k - run), run, &n_out) != 0)
                    return -1;
                run = 0;
                if (tok_wp_subword(tok, ws, ws->norm + k, 1, &n_out) != 0)
                    return -1;
            } else {
                run++;
            }
        }
        if (run > 0 && tok_wp_subword(tok, ws, ws->norm + (m - run), run, &n_out) != 0)
            return -1;
    }
    return n_out;
}

/* ================================ loading =================================== */

static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long len = ftell(f);
    if (len < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t rd = fread(buf, 1, (size_t)len, f);
    fclose(f);
    if (rd != (size_t)len) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    if (out_len)
        *out_len = len;
    return buf;
}

/* do_lower_case from config.json: default true (BERT uncased). A
 * lightweight scan is enough - we only need one boolean and avoid a JSON dep
 * here, mirroring how the config layer reads small flags. */
static int read_do_lower_case(const char *model_dir) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/tokenizer_config.json", model_dir);
    char *buf = read_file(path, NULL);
    if (!buf)
        return 1;
    int result = 1;
    const char *k = strstr(buf, "\"do_lower_case\"");
    if (k) {
        const char *c = strchr(k, ':');
        if (c) {
            c++;
            while (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r')
                c++;
            if (!strncmp(c, "false", 5))
                result = 0;
        }
    }
    free(buf);
    return result;
}

tok_wp_t *tok_wp_load(const char *model_dir) {
    if (!model_dir)
        return NULL;
    char path[2048];
    snprintf(path, sizeof(path), "%s/vocab.txt", model_dir);
    long len = 0;
    char *buf = read_file(path, &len);
    if (!buf) {
        fprintf(stderr, "wordpiece: cannot read %s\n", path);
        return NULL;
    }

    /* Count lines (one token each); a single trailing newline adds no token. */
    int count = 0;
    for (long i = 0; i < len; i++)
        if (buf[i] == '\n')
            count++;
    if (len > 0 && buf[len - 1] != '\n')
        count++;

    tok_wp_t *tok = (tok_wp_t *)calloc(1, sizeof(*tok));
    char **id_to_token = (char **)calloc((size_t)(count > 0 ? count : 1), sizeof(char *));
    if (!tok || !id_to_token) {
        free(buf);
        free(tok);
        free(id_to_token);
        return NULL;
    }

    int n = 0;
    char *line = buf;
    for (long i = 0; i <= len && n < count; i++) {
        if (i == len || buf[i] == '\n') {
            char *e = buf + i;
            if (e > line && e[-1] == '\r')
                e[-1] = '\0';
            else
                *e = '\0';
            id_to_token[n] = strdup(line);
            if (!id_to_token[n]) {
                for (int j = 0; j < n; j++)
                    free(id_to_token[j]);
                free(id_to_token);
                free(tok);
                free(buf);
                return NULL;
            }
            n++;
            line = buf + i + 1;
        }
    }
    free(buf);

    tok->id_to_token = id_to_token;
    tok->vocab_size = n;
    tok->do_lower_case = read_do_lower_case(model_dir);
    tok->max_chars_per_word = 100;

    int cap = next_pow2(n * 2 < 16 ? 16 : n * 2);
    wp_entry_t *map = (wp_entry_t *)calloc((size_t)cap, sizeof(wp_entry_t));
    if (!map) {
        tok_wp_free(tok);
        return NULL;
    }
    for (int i = 0; i < n; i++)
        map_put(map, cap, id_to_token[i], i);
    tok->vocab_map = map;
    tok->vocab_map_cap = cap;

    tok->unk_id = map_get(map, cap, "[UNK]");
    if (tok->unk_id < 0) {
        fprintf(stderr, "wordpiece: vocab.txt has no [UNK] token\n");
        tok_wp_free(tok);
        return NULL;
    }
    return tok;
}

/* ============================== public API ================================= */

const char *tok_wp_decode(const tok_wp_t *tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size)
        return NULL;
    return tok->id_to_token[id];
}

int tok_wp_token_id(const tok_wp_t *tok, const char *token) {
    if (!tok || !token)
        return -1;
    return map_get((const wp_entry_t *)tok->vocab_map, tok->vocab_map_cap, token);
}

int tok_wp_encode_into(const tok_wp_t *tok,
                       tok_wp_workspace_t *ws,
                       const char *text,
                       int *out_ids,
                       int out_cap,
                       int *out_n_tokens) {
    if (out_n_tokens)
        *out_n_tokens = 0;
    if (!tok || !ws || !text)
        return -1;
    int n = tok_wp_run(tok, ws, text);
    if (n < 0)
        return -1;
    if (out_n_tokens)
        *out_n_tokens = n;
    if (!out_ids || out_cap < n)
        return out_ids ? -2 : -1;
    memcpy(out_ids, ws->ids, (size_t)n * sizeof(int));
    return 0;
}

int *tok_wp_encode_with_workspace(const tok_wp_t *tok,
                                  tok_wp_workspace_t *ws,
                                  const char *text,
                                  int *out_n_tokens) {
    if (out_n_tokens)
        *out_n_tokens = 0;
    if (!tok || !ws || !text)
        return NULL;
    int n = tok_wp_run(tok, ws, text);
    if (n < 0)
        return NULL;
    int *ids = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!ids)
        return NULL;
    memcpy(ids, ws->ids, (size_t)n * sizeof(int));
    if (out_n_tokens)
        *out_n_tokens = n;
    return ids;
}

int *tok_wp_encode(const tok_wp_t *tok, const char *text, int *out_n) {
    if (out_n)
        *out_n = 0;
    tok_wp_workspace_t *ws = tok_wp_workspace_new();
    if (!ws)
        return NULL;
    int *ids = tok_wp_encode_with_workspace(tok, ws, text, out_n);
    tok_wp_workspace_free(ws);
    return ids;
}

void tok_wp_free(tok_wp_t *tok) {
    if (!tok)
        return;
    if (tok->id_to_token) {
        for (int i = 0; i < tok->vocab_size; i++)
            free(tok->id_to_token[i]);
        free(tok->id_to_token);
    }
    free(tok->vocab_map);
    free(tok);
}
