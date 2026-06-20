/*
 * sentencepiece.c - SentencePiece Unigram tokenizer.
 *
 * The implementation reads the small subset of ModelProto needed at runtime:
 * pieces, scores, piece types, trainer model_type, and normalizer options. The
 * normalizer supports SentencePiece's precompiled charsmap format used by
 * nmt_nfkc, then the encoder runs the standard Unigram Viterbi dynamic program
 * over UTF-8 byte offsets.
 */

#include "sentencepiece.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SP_MODEL_UNIGRAM = 1,
    SP_PIECE_NORMAL = 1,
    SP_PIECE_UNKNOWN = 2,
    SP_PIECE_CONTROL = 3,
    SP_PIECE_USER_DEFINED = 4,
    SP_PIECE_UNUSED = 5,
    SP_PIECE_BYTE = 6,
};

typedef struct {
    const char *key;
    uint16_t len;
    int val;
} sp_entry_t;

typedef struct {
    int id;
    int prev;
    double score;
} sp_best_t;

struct tok_spm_workspace {
    char *norm;
    size_t norm_len;
    size_t norm_cap;
    sp_best_t *best;
    int best_cap;
    int *ids;
    int ids_cap;
};

static uint64_t fnv1a_bytes(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int next_pow2(int x) {
    int p = 1;
    while (p < x && p <= (1 << 29))
        p <<= 1;
    return p;
}

static int map_put(sp_entry_t *map, int cap, const char *key, int len, int val) {
    if (len <= 0 || len > UINT16_MAX)
        return -1;
    uint64_t mask = (uint64_t)cap - 1;
    int pos = (int)(fnv1a_bytes(key, (size_t)len) & mask);
    while (map[pos].key) {
        if (map[pos].len == (uint16_t)len && memcmp(map[pos].key, key, (size_t)len) == 0) {
            map[pos].val = val;
            return 0;
        }
        pos = (int)((pos + 1) & (int)mask);
    }
    map[pos].key = key;
    map[pos].len = (uint16_t)len;
    map[pos].val = val;
    return 0;
}

static int map_get(const sp_entry_t *map, int cap, const char *key, int len) {
    if (!map || len <= 0)
        return -1;
    uint64_t mask = (uint64_t)cap - 1;
    int pos = (int)(fnv1a_bytes(key, (size_t)len) & mask);
    while (map[pos].key) {
        if (map[pos].len == (uint16_t)len && memcmp(map[pos].key, key, (size_t)len) == 0)
            return map[pos].val;
        pos = (int)((pos + 1) & (int)mask);
    }
    return -1;
}

static int is_tokenizable_type(int type) {
    return type == SP_PIECE_NORMAL || type == SP_PIECE_USER_DEFINED || type == SP_PIECE_UNUSED ||
           type == SP_PIECE_BYTE;
}

static int reserve_chars(char **buf, size_t *cap, size_t need) {
    if (need <= *cap)
        return 0;
    size_t nc = *cap ? *cap : 128;
    while (nc < need) {
        if (nc > SIZE_MAX / 2)
            return -1;
        nc *= 2;
    }
    char *p = (char *)realloc(*buf, nc);
    if (!p)
        return -1;
    *buf = p;
    *cap = nc;
    return 0;
}

static int append_bytes(char **buf, size_t *len, size_t *cap, const char *s, size_t n) {
    if (n > SIZE_MAX - *len - 1 || reserve_chars(buf, cap, *len + n + 1) != 0)
        return -1;
    memcpy(*buf + *len, s, n);
    *len += n;
    (*buf)[*len] = '\0';
    return 0;
}

static int reserve_best(tok_spm_workspace_t *ws, int need) {
    if (need <= ws->best_cap)
        return 0;
    int nc = ws->best_cap ? ws->best_cap : 128;
    while (nc < need) {
        if (nc > INT_MAX / 2)
            return -1;
        nc *= 2;
    }
    sp_best_t *p = (sp_best_t *)realloc(ws->best, (size_t)nc * sizeof(*p));
    if (!p)
        return -1;
    ws->best = p;
    ws->best_cap = nc;
    return 0;
}

static int reserve_ids(tok_spm_workspace_t *ws, int need) {
    if (need <= ws->ids_cap)
        return 0;
    int nc = ws->ids_cap ? ws->ids_cap : 64;
    while (nc < need) {
        if (nc > INT_MAX / 2)
            return -1;
        nc *= 2;
    }
    int *p = (int *)realloc(ws->ids, (size_t)nc * sizeof(*p));
    if (!p)
        return -1;
    ws->ids = p;
    ws->ids_cap = nc;
    return 0;
}

static int utf8_len(unsigned char c) {
    if (c < 0x80)
        return 1;
    if ((c & 0xE0) == 0xC0)
        return 2;
    if ((c & 0xF0) == 0xE0)
        return 3;
    if ((c & 0xF8) == 0xF0)
        return 4;
    return 1;
}

static int valid_utf8_one(const char *s, size_t n, size_t off, size_t *len_out) {
    if (off >= n)
        return 0;
    const unsigned char *p = (const unsigned char *)s + off;
    int len = utf8_len(p[0]);
    if (off + (size_t)len > n) {
        *len_out = 1;
        return 0;
    }
    if (len > 1) {
        for (int i = 1; i < len; i++) {
            if ((p[i] & 0xC0) != 0x80) {
                *len_out = 1;
                return 0;
            }
        }
    }
    *len_out = (size_t)len;
    return 1;
}

static uint32_t read_le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int pb_read_varint(const unsigned char **p, const unsigned char *end, uint64_t *out) {
    uint64_t v = 0;
    int shift = 0;
    while (*p < end && shift <= 63) {
        unsigned char c = *(*p)++;
        v |= (uint64_t)(c & 0x7f) << shift;
        if (!(c & 0x80)) {
            *out = v;
            return 0;
        }
        shift += 7;
    }
    return -1;
}

static int pb_read_key(const unsigned char **p, const unsigned char *end, int *field, int *wire) {
    uint64_t key = 0;
    if (pb_read_varint(p, end, &key) != 0)
        return -1;
    *field = (int)(key >> 3);
    *wire = (int)(key & 7);
    return *field > 0 ? 0 : -1;
}

static int pb_read_len(const unsigned char **p,
                       const unsigned char *end,
                       const unsigned char **data,
                       size_t *len) {
    uint64_t n = 0;
    if (pb_read_varint(p, end, &n) != 0 || n > (uint64_t)(end - *p))
        return -1;
    *data = *p;
    *len = (size_t)n;
    *p += n;
    return 0;
}

static int pb_skip(const unsigned char **p, const unsigned char *end, int wire) {
    uint64_t v = 0;
    const unsigned char *data = NULL;
    size_t len = 0;
    switch (wire) {
    case 0:
        return pb_read_varint(p, end, &v);
    case 1:
        if ((size_t)(end - *p) < 8)
            return -1;
        *p += 8;
        return 0;
    case 2:
        return pb_read_len(p, end, &data, &len);
    case 5:
        if ((size_t)(end - *p) < 4)
            return -1;
        *p += 4;
        return 0;
    default:
        return -1;
    }
}

static char *memdup0(const unsigned char *data, size_t len) {
    char *s = (char *)malloc(len + 1);
    if (!s)
        return NULL;
    memcpy(s, data, len);
    s[len] = '\0';
    return s;
}

static int parse_piece(tok_spm_t *tok, int idx, const unsigned char *data, size_t len) {
    const unsigned char *p = data, *end = data + len;
    char *piece = NULL;
    float score = 0.0f;
    int type = SP_PIECE_NORMAL;
    while (p < end) {
        int field = 0, wire = 0;
        if (pb_read_key(&p, end, &field, &wire) != 0)
            goto fail;
        if (field == 1 && wire == 2) {
            const unsigned char *s = NULL;
            size_t n = 0;
            if (pb_read_len(&p, end, &s, &n) != 0 || n > UINT16_MAX)
                goto fail;
            free(piece);
            piece = memdup0(s, n);
            if (!piece)
                goto fail;
            tok->piece_lens[idx] = (uint16_t)n;
        } else if (field == 2 && wire == 5) {
            if ((size_t)(end - p) < 4)
                goto fail;
            uint32_t bits = read_le32(p);
            memcpy(&score, &bits, sizeof(score));
            p += 4;
        } else if (field == 3 && wire == 0) {
            uint64_t v = 0;
            if (pb_read_varint(&p, end, &v) != 0 || v > UINT8_MAX)
                goto fail;
            type = (int)v;
        } else if (pb_skip(&p, end, wire) != 0) {
            goto fail;
        }
    }
    if (!piece)
        goto fail;
    tok->id_to_piece[idx] = piece;
    tok->scores[idx] = score;
    tok->types[idx] = (uint8_t)type;
    if (is_tokenizable_type(type) && tok->piece_lens[idx] > tok->max_piece_len)
        tok->max_piece_len = tok->piece_lens[idx];
    return 0;
fail:
    free(piece);
    return -1;
}

static int parse_trainer(tok_spm_t *tok, const unsigned char *data, size_t len) {
    const unsigned char *p = data, *end = data + len;
    int model_type = SP_MODEL_UNIGRAM;
    while (p < end) {
        int field = 0, wire = 0;
        if (pb_read_key(&p, end, &field, &wire) != 0)
            return -1;
        if (field == 3 && wire == 0) {
            uint64_t v = 0;
            if (pb_read_varint(&p, end, &v) != 0 || v > INT_MAX)
                return -1;
            model_type = (int)v;
        } else if (pb_skip(&p, end, wire) != 0) {
            return -1;
        }
    }
    if (model_type != SP_MODEL_UNIGRAM) {
        fprintf(stderr, "sentencepiece: unsupported model_type %d (only Unigram is implemented)\n",
                model_type);
        return -1;
    }
    (void)tok;
    return 0;
}

static int parse_normalizer(tok_spm_t *tok, const unsigned char *data, size_t len) {
    const unsigned char *p = data, *end = data + len;
    while (p < end) {
        int field = 0, wire = 0;
        if (pb_read_key(&p, end, &field, &wire) != 0)
            return -1;
        if ((field == 3 || field == 4 || field == 5) && wire == 0) {
            uint64_t v = 0;
            if (pb_read_varint(&p, end, &v) != 0)
                return -1;
            if (field == 3)
                tok->add_dummy_prefix = v != 0;
            else if (field == 4)
                tok->remove_extra_whitespaces = v != 0;
            else
                tok->escape_whitespaces = v != 0;
        } else if (field == 2 && wire == 2) {
            const unsigned char *blob = NULL;
            size_t n = 0;
            if (pb_read_len(&p, end, &blob, &n) != 0)
                return -1;
            if (n >= 4) {
                uint32_t xcda_bytes = read_le32(blob);
                if ((size_t)xcda_bytes + 4 <= n && (xcda_bytes % 4) == 0) {
                    tok->xcda_count = xcda_bytes / 4;
                    tok->xcda = (uint32_t *)malloc(tok->xcda_count * sizeof(*tok->xcda));
                    if (!tok->xcda)
                        return -1;
                    for (size_t i = 0; i < tok->xcda_count; i++)
                        tok->xcda[i] = read_le32(blob + 4 + i * 4);
                    tok->prefix_replacements_len = n - 4 - xcda_bytes;
                    tok->prefix_replacements = (char *)malloc(tok->prefix_replacements_len + 1);
                    if (!tok->prefix_replacements)
                        return -1;
                    memcpy(tok->prefix_replacements, blob + 4 + xcda_bytes,
                           tok->prefix_replacements_len);
                    tok->prefix_replacements[tok->prefix_replacements_len] = '\0';
                }
            }
        } else if (pb_skip(&p, end, wire) != 0) {
            return -1;
        }
    }
    return 0;
}

static int count_pieces(const unsigned char *data, size_t len) {
    const unsigned char *p = data, *end = data + len;
    int n = 0;
    while (p < end) {
        int field = 0, wire = 0;
        if (pb_read_key(&p, end, &field, &wire) != 0)
            return -1;
        if (field == 1 && wire == 2) {
            const unsigned char *msg = NULL;
            size_t msg_len = 0;
            if (pb_read_len(&p, end, &msg, &msg_len) != 0)
                return -1;
            n++;
        } else if (pb_skip(&p, end, wire) != 0) {
            return -1;
        }
    }
    return n;
}

static int parse_model(tok_spm_t *tok, const unsigned char *data, size_t len) {
    int n_pieces = count_pieces(data, len);
    if (n_pieces <= 0)
        return -1;
    tok->piece_count = n_pieces;
    tok->id_to_piece = (char **)calloc((size_t)n_pieces, sizeof(*tok->id_to_piece));
    tok->piece_lens = (uint16_t *)calloc((size_t)n_pieces, sizeof(*tok->piece_lens));
    tok->scores = (float *)calloc((size_t)n_pieces, sizeof(*tok->scores));
    tok->types = (uint8_t *)calloc((size_t)n_pieces, sizeof(*tok->types));
    if (!tok->id_to_piece || !tok->piece_lens || !tok->scores || !tok->types)
        return -1;

    const unsigned char *p = data, *end = data + len;
    int piece_idx = 0;
    int saw_trainer = 0;
    while (p < end) {
        int field = 0, wire = 0;
        if (pb_read_key(&p, end, &field, &wire) != 0)
            return -1;
        if (wire == 2 && field == 1) {
            const unsigned char *msg = NULL;
            size_t msg_len = 0;
            if (pb_read_len(&p, end, &msg, &msg_len) != 0 || piece_idx >= n_pieces ||
                parse_piece(tok, piece_idx++, msg, msg_len) != 0)
                return -1;
        } else if (wire == 2 && field == 2) {
            const unsigned char *msg = NULL;
            size_t msg_len = 0;
            if (pb_read_len(&p, end, &msg, &msg_len) != 0 || parse_trainer(tok, msg, msg_len) != 0)
                return -1;
            saw_trainer = 1;
        } else if (wire == 2 && field == 3) {
            const unsigned char *msg = NULL;
            size_t msg_len = 0;
            if (pb_read_len(&p, end, &msg, &msg_len) != 0 || parse_normalizer(tok, msg, msg_len) != 0)
                return -1;
        } else if (pb_skip(&p, end, wire) != 0) {
            return -1;
        }
    }
    if (piece_idx != n_pieces || !saw_trainer)
        return -1;
    return 0;
}

static int raw_piece_id(const tok_spm_t *tok, const char *piece) {
    if (!tok || !piece)
        return -1;
    return map_get((const sp_entry_t *)tok->piece_map, tok->piece_map_cap, piece, (int)strlen(piece));
}

static int raw_to_model_id(const tok_spm_t *tok, int raw_id) {
    if (!tok || raw_id < 0 || raw_id >= tok->piece_count)
        return -1;
    if (tok->id_map == TOK_SPM_XLM_ROBERTA) {
        if (raw_id == 1)
            return 0; /* <s> */
        if (raw_id == 2)
            return 2; /* </s> */
        if (raw_id == 0)
            return 3; /* <unk> */
        return raw_id + 1;
    }
    return raw_id;
}

static int model_to_raw_id(const tok_spm_t *tok, int id) {
    if (!tok || id < 0)
        return -1;
    if (tok->id_map == TOK_SPM_XLM_ROBERTA) {
        if (id == 0)
            return 1;
        if (id == 2)
            return 2;
        if (id == 3)
            return 0;
        if (id >= 4 && id <= tok->piece_count)
            return id - 1;
        return -1;
    }
    return id < tok->piece_count ? id : -1;
}

static int file_contains_xlm_roberta(const char *model_dir, const char *name) {
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", model_dir, name);
    if (n < 0 || (size_t)n >= sizeof(path))
        return 0;
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    long sz = ftell(f);
    if (sz < 0 || (size_t)sz == SIZE_MAX || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return 0;
    }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = '\0';
    int yes = strstr(buf, "XLMRobertaTokenizer") != NULL ||
              strstr(buf, "XLMRobertaTokenizerFast") != NULL ||
              strstr(buf, "XLMRobertaModel") != NULL || strstr(buf, "xlm-roberta") != NULL;
    free(buf);
    return yes;
}

static int detect_xlm_roberta(const char *model_dir) {
    return file_contains_xlm_roberta(model_dir, "tokenizer_config.json") ||
           file_contains_xlm_roberta(model_dir, "config.json");
}

static int find_model_file(const char *model_dir, char path[1024]) {
    static const char *names[] = {
        "sentencepiece.bpe.model",
        "tokenizer.model",
        "spiece.model",
    };
    for (int i = 0; i < (int)(sizeof(names) / sizeof(names[0])); i++) {
        int n = snprintf(path, 1024, "%s/%s", model_dir, names[i]);
        if (n < 0 || n >= 1024)
            return -1;
        FILE *f = fopen(path, "rb");
        if (f) {
            fclose(f);
            return 0;
        }
        if (errno != ENOENT)
            return -1;
    }
    return -1;
}

static int load_file(const char *path, unsigned char **out, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long sz = ftell(f);
    if (sz < 0 || (size_t)sz == SIZE_MAX || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        free(buf);
        return -1;
    }
    fclose(f);
    *out = buf;
    *out_len = (size_t)sz;
    return 0;
}

static uint32_t xcda_base(const tok_spm_t *tok, size_t idx) {
    uint32_t v = tok->xcda[idx];
    return (v >> 10) << ((v & (1u << 9)) >> 6);
}

static uint32_t xcda_lcheck(const tok_spm_t *tok, size_t idx) {
    return tok->xcda[idx] & ((1u << 31) | 0xffu);
}

static int xcda_leaf(const tok_spm_t *tok, size_t idx) { return (tok->xcda[idx] >> 8) & 1u; }

static uint32_t xcda_value(const tok_spm_t *tok, size_t idx) {
    return tok->xcda[idx] & ((1u << 31) - 1u);
}

typedef struct {
    const char *data;
    size_t len;
    size_t consumed;
} norm_prefix_t;

static norm_prefix_t
normalize_prefix(const tok_spm_t *tok, const char *input, size_t input_len, size_t off) {
    static const char repl[] = "\xef\xbf\xbd";
    if (off >= input_len)
        return (norm_prefix_t){input + off, 0, 0};

    size_t best_len = 0;
    size_t best_repl = 0;
    if (tok->xcda && tok->xcda_count > 0) {
        uint32_t node = xcda_base(tok, 0);
        for (size_t p = off; p < input_len; p++) {
            unsigned char c = (unsigned char)input[p];
            if (c == 0)
                break;
            node ^= c;
            if (node >= tok->xcda_count || xcda_lcheck(tok, node) != c)
                break;
            int leaf = xcda_leaf(tok, node);
            node ^= xcda_base(tok, node);
            if (leaf) {
                best_len = p - off + 1;
                best_repl = xcda_value(tok, node);
            }
        }
    }
    if (best_len > 0 && best_repl < tok->prefix_replacements_len) {
        const char *r = tok->prefix_replacements + best_repl;
        size_t max = tok->prefix_replacements_len - best_repl;
        size_t n = strnlen(r, max);
        return (norm_prefix_t){r, n, best_len};
    }

    size_t clen = 1;
    if (valid_utf8_one(input, input_len, off, &clen))
        return (norm_prefix_t){input + off, clen, clen};
    return (norm_prefix_t){repl, 3, 1};
}

static int normalize_into(const tok_spm_t *tok, tok_spm_workspace_t *ws, const char *text) {
    ws->norm_len = 0;
    if (!text)
        return -1;
    size_t input_len = strlen(text);
    const char space[] = "\xe2\x96\x81"; /* U+2581 */
    const char *space_s = tok->escape_whitespaces ? space : " ";
    size_t space_n = tok->escape_whitespaces ? 3 : 1;
    int processing_non_ws = 0;
    int space_prepended = 0;

    for (size_t off = 0; off < input_len;) {
        norm_prefix_t r = normalize_prefix(tok, text, input_len, off);
        for (size_t i = 0; i < r.len; i++) {
            char c = r.data[i];
            if (c != ' ') {
                if (!processing_non_ws) {
                    processing_non_ws = 1;
                    if ((tok->add_dummy_prefix && !space_prepended) ||
                        tok->remove_extra_whitespaces) {
                        if (append_bytes(&ws->norm, &ws->norm_len, &ws->norm_cap, space_s, space_n) !=
                            0)
                            return -1;
                        space_prepended = 1;
                    }
                }
                if (append_bytes(&ws->norm, &ws->norm_len, &ws->norm_cap, &c, 1) != 0)
                    return -1;
            } else {
                processing_non_ws = 0;
                if (!tok->remove_extra_whitespaces) {
                    if (append_bytes(&ws->norm, &ws->norm_len, &ws->norm_cap, space_s, space_n) != 0)
                        return -1;
                }
            }
        }
        off += r.consumed ? r.consumed : 1;
    }
    if (ws->norm)
        ws->norm[ws->norm_len] = '\0';
    return 0;
}

static double compute_unknown_score(const tok_spm_t *tok) {
    float min_score = FLT_MAX;
    for (int i = 0; i < tok->piece_count; i++) {
        if (tok->types[i] == SP_PIECE_NORMAL && tok->scores[i] < min_score)
            min_score = tok->scores[i];
    }
    if (min_score == FLT_MAX)
        min_score = -10.0f;
    return (double)min_score - 10.0;
}

static int segment(tok_spm_t const *tok, tok_spm_workspace_t *ws) {
    size_t n = ws->norm_len;
    if (n == 0)
        return 0;
    if (n > (size_t)INT_MAX - 1 || reserve_best(ws, (int)n + 1) != 0 ||
        reserve_ids(ws, (int)n + 1) != 0)
        return -1;
    for (size_t i = 0; i <= n; i++)
        ws->best[i] = (sp_best_t){tok->unk_id, -1, -DBL_MAX};
    ws->best[0] = (sp_best_t){tok->unk_id, 0, 0.0};
    double unk_score = tok->unknown_score;

    for (size_t off = 0; off < n;) {
        size_t cp_len = 1;
        valid_utf8_one(ws->norm, n, off, &cp_len);
        sp_best_t cur = ws->best[off];
        if (cur.score <= -DBL_MAX / 2.0) {
            off += cp_len;
            continue;
        }

        int single_cp_found = 0;
        for (size_t end = off + cp_len; end <= n && end - off <= (size_t)tok->max_piece_len;) {
            int raw = map_get((const sp_entry_t *)tok->piece_map, tok->piece_map_cap, ws->norm + off,
                              (int)(end - off));
            if (raw >= 0 && is_tokenizable_type(tok->types[raw])) {
                if (end - off == cp_len)
                    single_cp_found = 1;
                double sc =
                    cur.score + (tok->types[raw] == SP_PIECE_USER_DEFINED ? 0.0 : tok->scores[raw]);
                if (sc > ws->best[end].score)
                    ws->best[end] = (sp_best_t){raw_to_model_id(tok, raw), (int)off, sc};
            }
            if (end == n)
                break;
            size_t next_len = 1;
            valid_utf8_one(ws->norm, n, end, &next_len);
            end += next_len;
        }
        if (!single_cp_found) {
            size_t end = off + cp_len;
            double sc = cur.score + unk_score;
            if (sc > ws->best[end].score)
                ws->best[end] = (sp_best_t){tok->unk_id, (int)off, sc};
        }
        off += cp_len;
    }

    int out_n = 0;
    int prev_unk = 0;
    for (size_t off = n; off > 0;) {
        sp_best_t b = ws->best[off];
        if (b.prev < 0 || b.prev >= (int)off)
            return -1;
        int is_unk = b.id == tok->unk_id;
        if (!(prev_unk && is_unk)) {
            if (out_n >= ws->ids_cap)
                return -1;
            ws->ids[out_n++] = b.id;
        }
        prev_unk = is_unk;
        off = (size_t)b.prev;
    }
    for (int i = 0; i < out_n / 2; i++) {
        int t = ws->ids[i];
        ws->ids[i] = ws->ids[out_n - 1 - i];
        ws->ids[out_n - 1 - i] = t;
    }
    return out_n;
}

tok_spm_t *tok_spm_load(const char *model_dir) {
    if (!model_dir)
        return NULL;
    char path[1024];
    if (find_model_file(model_dir, path) != 0) {
        fprintf(stderr, "sentencepiece: model file not found under %s\n", model_dir);
        return NULL;
    }
    unsigned char *data = NULL;
    size_t len = 0;
    if (load_file(path, &data, &len) != 0) {
        fprintf(stderr, "sentencepiece: cannot read %s\n", path);
        return NULL;
    }

    tok_spm_t *tok = (tok_spm_t *)calloc(1, sizeof(tok_spm_t));
    if (!tok) {
        free(data);
        return NULL;
    }
    tok->add_dummy_prefix = 1;
    tok->remove_extra_whitespaces = 1;
    tok->escape_whitespaces = 1;
    tok->id_map = detect_xlm_roberta(model_dir) ? TOK_SPM_XLM_ROBERTA : TOK_SPM_IDENTITY;
    if (parse_model(tok, data, len) != 0) {
        fprintf(stderr, "sentencepiece: malformed or unsupported model: %s\n", path);
        free(data);
        tok_spm_free(tok);
        return NULL;
    }
    free(data);

    int cap = next_pow2(tok->piece_count * 2 + 1);
    tok->piece_map = calloc((size_t)cap, sizeof(sp_entry_t));
    if (!tok->piece_map) {
        tok_spm_free(tok);
        return NULL;
    }
    tok->piece_map_cap = cap;
    for (int i = 0; i < tok->piece_count; i++) {
        if (!tok->id_to_piece[i] || map_put((sp_entry_t *)tok->piece_map, cap, tok->id_to_piece[i],
                                            tok->piece_lens[i], i) != 0) {
            tok_spm_free(tok);
            return NULL;
        }
    }

    tok->unk_id = raw_to_model_id(tok, raw_piece_id(tok, "<unk>"));
    tok->bos_id = raw_to_model_id(tok, raw_piece_id(tok, "<s>"));
    tok->eos_id = raw_to_model_id(tok, raw_piece_id(tok, "</s>"));
    tok->pad_id =
        tok->id_map == TOK_SPM_XLM_ROBERTA ? 1 : raw_to_model_id(tok, raw_piece_id(tok, "<pad>"));
    tok->mask_id = tok->id_map == TOK_SPM_XLM_ROBERTA
                       ? tok->piece_count + 1
                       : raw_to_model_id(tok, raw_piece_id(tok, "<mask>"));
    tok->vocab_size = tok->id_map == TOK_SPM_XLM_ROBERTA ? tok->piece_count + 2 : tok->piece_count;
    tok->unknown_score = compute_unknown_score(tok);
    if (tok->unk_id < 0 || tok->max_piece_len <= 0) {
        fprintf(stderr, "sentencepiece: missing <unk> or normal pieces in %s\n", path);
        tok_spm_free(tok);
        return NULL;
    }
    return tok;
}

const char *tok_spm_decode(const tok_spm_t *tok, int id) {
    if (!tok || id < 0)
        return NULL;
    if (tok->id_map == TOK_SPM_XLM_ROBERTA) {
        if (id == 1)
            return "<pad>";
        if (id == tok->piece_count + 1)
            return "<mask>";
    }
    int raw = model_to_raw_id(tok, id);
    return raw >= 0 ? tok->id_to_piece[raw] : NULL;
}

int tok_spm_token_id(const tok_spm_t *tok, const char *token) {
    if (!tok || !token)
        return -1;
    if (tok->id_map == TOK_SPM_XLM_ROBERTA) {
        if (!strcmp(token, "<s>"))
            return 0;
        if (!strcmp(token, "<pad>"))
            return 1;
        if (!strcmp(token, "</s>"))
            return 2;
        if (!strcmp(token, "<unk>"))
            return 3;
        if (!strcmp(token, "<mask>"))
            return tok->piece_count + 1;
    }
    return raw_to_model_id(tok, raw_piece_id(tok, token));
}

int tok_spm_encode_into(const tok_spm_t *tok,
                        tok_spm_workspace_t *ws,
                        const char *text,
                        int *out_ids,
                        int out_cap,
                        int *out_n_tokens) {
    if (out_n_tokens)
        *out_n_tokens = 0;
    if (!tok || !ws || !text)
        return -1;
    if (normalize_into(tok, ws, text) != 0)
        return -1;
    int n = segment(tok, ws);
    if (n < 0)
        return -1;
    if (out_n_tokens)
        *out_n_tokens = n;
    if (n == 0)
        return 0;
    if (!out_ids || out_cap < n)
        return -2;
    memcpy(out_ids, ws->ids, (size_t)n * sizeof(*out_ids));
    return 0;
}

int *tok_spm_encode_with_workspace(const tok_spm_t *tok,
                                   tok_spm_workspace_t *ws,
                                   const char *text,
                                   int *out_n_tokens) {
    if (out_n_tokens)
        *out_n_tokens = 0;
    if (!tok || !ws || !text)
        return NULL;
    int n = 0;
    int rc = tok_spm_encode_into(tok, ws, text, NULL, 0, &n);
    if (rc < 0 && rc != -2)
        return NULL;
    if (out_n_tokens)
        *out_n_tokens = n;
    if (n == 0)
        return NULL;
    int *ids = (int *)malloc((size_t)n * sizeof(*ids));
    if (!ids)
        return NULL;
    memcpy(ids, ws->ids, (size_t)n * sizeof(*ids));
    return ids;
}

int *tok_spm_encode(const tok_spm_t *tok, const char *text, int *out_n_tokens) {
    tok_spm_workspace_t *ws = tok_spm_workspace_new();
    if (!ws) {
        if (out_n_tokens)
            *out_n_tokens = 0;
        return NULL;
    }
    int *ids = tok_spm_encode_with_workspace(tok, ws, text, out_n_tokens);
    tok_spm_workspace_free(ws);
    return ids;
}

tok_spm_workspace_t *tok_spm_workspace_new(void) {
    return (tok_spm_workspace_t *)calloc(1, sizeof(tok_spm_workspace_t));
}

void tok_spm_workspace_free(tok_spm_workspace_t *ws) {
    if (!ws)
        return;
    free(ws->norm);
    free(ws->best);
    free(ws->ids);
    free(ws);
}

void tok_spm_free(tok_spm_t *tok) {
    if (!tok)
        return;
    if (tok->id_to_piece) {
        for (int i = 0; i < tok->piece_count; i++)
            free(tok->id_to_piece[i]);
    }
    free(tok->id_to_piece);
    free(tok->piece_lens);
    free(tok->scores);
    free(tok->types);
    free(tok->piece_map);
    free(tok->xcda);
    free(tok->prefix_replacements);
    free(tok);
}
