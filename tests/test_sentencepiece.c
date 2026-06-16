/* tests/test_sentencepiece.c - hermetic SentencePiece Unigram checks. */

#include "tokenizer_sentencepiece.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    unsigned char *p;
    size_t n;
    size_t cap;
} buf_t;

static int reserve(buf_t *b, size_t need) {
    if (need <= b->cap)
        return 0;
    size_t nc = b->cap ? b->cap : 128;
    while (nc < need)
        nc *= 2;
    unsigned char *p = (unsigned char *)realloc(b->p, nc);
    if (!p)
        return -1;
    b->p = p;
    b->cap = nc;
    return 0;
}

static int put_byte(buf_t *b, int c) {
    if (reserve(b, b->n + 1) != 0)
        return -1;
    b->p[b->n++] = (unsigned char)c;
    return 0;
}

static int put_varint(buf_t *b, uint64_t v) {
    while (v >= 0x80) {
        if (put_byte(b, (int)(v | 0x80)) != 0)
            return -1;
        v >>= 7;
    }
    return put_byte(b, (int)v);
}

static int put_key(buf_t *b, int field, int wire) {
    return put_varint(b, ((uint64_t)field << 3) | (uint64_t)wire);
}

static int put_bytes(buf_t *b, const void *data, size_t n) {
    if (reserve(b, b->n + n) != 0)
        return -1;
    memcpy(b->p + b->n, data, n);
    b->n += n;
    return 0;
}

static int put_fixed32(buf_t *b, float f) {
    uint32_t v;
    memcpy(&v, &f, sizeof(v));
    unsigned char p[4] = {
        (unsigned char)(v & 0xff),
        (unsigned char)((v >> 8) & 0xff),
        (unsigned char)((v >> 16) & 0xff),
        (unsigned char)((v >> 24) & 0xff),
    };
    return put_bytes(b, p, sizeof(p));
}

static int put_len_msg(buf_t *dst, int field, const buf_t *msg) {
    return put_key(dst, field, 2) || put_varint(dst, msg->n) || put_bytes(dst, msg->p, msg->n);
}

static int add_piece(buf_t *model, const char *piece, float score, int type) {
    buf_t msg = {0};
    int rc = put_key(&msg, 1, 2) || put_varint(&msg, strlen(piece)) ||
             put_bytes(&msg, piece, strlen(piece)) || put_key(&msg, 2, 5) ||
             put_fixed32(&msg, score) || put_key(&msg, 3, 0) || put_varint(&msg, (uint64_t)type) ||
             put_len_msg(model, 1, &msg);
    free(msg.p);
    return rc;
}

static int write_model(const char *dir, int xlm_roberta) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/sentencepiece.bpe.model", dir);
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;

    buf_t model = {0};
    int rc = 0;
    rc |= add_piece(&model, "<unk>", 0.0f, 2);
    rc |= add_piece(&model, "<s>", 0.0f, 3);
    rc |= add_piece(&model, "</s>", 0.0f, 3);
    rc |= add_piece(&model, ",", -3.0f, 1);
    rc |= add_piece(&model, ".", -3.0f, 1);
    rc |= add_piece(&model, "\xe2\x96\x81", -4.0f, 1);
    rc |= add_piece(&model, "!", -3.0f, 1);
    rc |= add_piece(&model, "\xe2\x96\x81Hello", -1.0f, 1);
    rc |= add_piece(&model, "\xe2\x96\x81world", -1.0f, 1);
    rc |= add_piece(&model, "\xe5\x8d\x97", -1.0f, 1);
    rc |= add_piece(&model, "\xe7\x93\x9c", -1.0f, 1);
    rc |= add_piece(&model, "\xe2\x96\x81Hi", -1.0f, 1);

    buf_t trainer = {0};
    rc |= put_key(&trainer, 3, 0) || put_varint(&trainer, 1) || put_len_msg(&model, 2, &trainer);
    free(trainer.p);

    buf_t normalizer = {0};
    rc |= put_key(&normalizer, 3, 0) || put_varint(&normalizer, 1);
    rc |= put_key(&normalizer, 4, 0) || put_varint(&normalizer, 1);
    rc |= put_key(&normalizer, 5, 0) || put_varint(&normalizer, 1);
    rc |= put_len_msg(&model, 3, &normalizer);
    free(normalizer.p);

    if (rc == 0 && fwrite(model.p, 1, model.n, f) != model.n)
        rc = -1;
    fclose(f);
    free(model.p);
    if (rc != 0)
        return -1;

    if (xlm_roberta) {
        snprintf(path, sizeof(path), "%s/tokenizer_config.json", dir);
        f = fopen(path, "w");
        if (!f)
            return -1;
        fputs("{\"tokenizer_class\":\"XLMRobertaTokenizer\"}", f);
        fclose(f);
    }
    return 0;
}

static int check_ids(const sentencepiece_tokenizer_t *tok,
                     sentencepiece_workspace_t *ws,
                     const char *text,
                     const int *want,
                     int want_n,
                     const char *name) {
    int n = 0;
    int *ids = sentencepiece_tokenizer_encode_with_workspace(tok, ws, text, &n);
    int ok = n == want_n;
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

static int run_hermetic(void) {
    const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/embed-sp-test-XXXXXX", tmp);
    if (!mkdtemp(dir) || write_model(dir, 1) != 0) {
        fprintf(stderr, "fixture creation failed\n");
        return 2;
    }

    sentencepiece_tokenizer_t *tok = sentencepiece_tokenizer_load(dir);
    sentencepiece_workspace_t *ws = sentencepiece_workspace_new();
    if (!tok || !ws) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    if (tok->piece_count != 12 || tok->vocab_size != 14 || tok->unk_id != 3 ||
        tok->bos_id != 0 || tok->eos_id != 2 || tok->pad_id != 1 || tok->mask_id != 13) {
        fprintf(stderr, "metadata wrong\n");
        return 1;
    }
    if (sentencepiece_tokenizer_token_id(tok, "<s>") != 0 ||
        sentencepiece_tokenizer_token_id(tok, "</s>") != 2 ||
        sentencepiece_tokenizer_token_id(tok, "<unk>") != 3 ||
        sentencepiece_tokenizer_token_id(tok, ",") != 4 ||
        sentencepiece_tokenizer_token_id(tok, "<mask>") != 13 ||
        strcmp(sentencepiece_tokenizer_decode(tok, 8), "\xe2\x96\x81Hello") != 0 ||
        strcmp(sentencepiece_tokenizer_decode(tok, 1), "<pad>") != 0) {
        fprintf(stderr, "special id/decode mapping wrong\n");
        return 1;
    }

    int rc = 0;
    rc |= check_ids(tok, ws, "Hello world!", (int[]){8, 9, 7}, 3, "hello");
    rc |= check_ids(tok, ws, "   ", NULL, 0, "spaces");
    rc |= check_ids(tok, ws, "\xe5\x8d\x97\xe7\x93\x9c", (int[]){6, 10, 11}, 3, "cjk");
    rc |= check_ids(tok, ws, "missing", (int[]){6, 3}, 2, "unknown-collapse");

    int tiny[1], need = 0;
    if (sentencepiece_tokenizer_encode_into(tok, ws, "Hello world!", tiny, 1, &need) != -2 ||
        need != 3) {
        fprintf(stderr, "small-buffer handling wrong\n");
        return 1;
    }
    sentencepiece_workspace_free(ws);
    sentencepiece_tokenizer_free(tok);

    char direct[1024];
    snprintf(direct, sizeof(direct), "%s/embed-sp-direct-XXXXXX", tmp);
    if (!mkdtemp(direct) || write_model(direct, 0) != 0)
        return 2;
    tok = sentencepiece_tokenizer_load(direct);
    ws = sentencepiece_workspace_new();
    if (!tok || !ws)
        return 1;
    rc |= check_ids(tok, ws, "Hello world!", (int[]){7, 8, 6}, 3, "direct-map");
    sentencepiece_workspace_free(ws);
    sentencepiece_tokenizer_free(tok);
    if (rc)
        return 1;

    puts("ok: SentencePiece Unigram normalization, Viterbi, XLM-R id mapping");
    return 0;
}

static int run_live(const char *model_dir) {
    sentencepiece_tokenizer_t *tok = sentencepiece_tokenizer_load(model_dir);
    sentencepiece_workspace_t *ws = sentencepiece_workspace_new();
    if (!tok || !ws) {
        fprintf(stderr, "live load failed\n");
        return 1;
    }
    int rc = 0;
    rc |= check_ids(tok, ws, "Hello world! \xe5\x8d\x97\xe7\x93\x9c",
                    (int[]){35378, 8999, 38, 6, 4617, 39613}, 6, "live-mixed");
    rc |= check_ids(tok, ws, "caf\xc3\xa9 d\xc3\xa9j\xc3\xa0 vu",
                    (int[]){26216, 15154, 13946}, 3, "live-accents");
    rc |= check_ids(tok, ws, "\xef\xbc\xa1\xef\xbc\xa2\xef\xbc\xa3 \xef\xbc\x91\xef\xbc\x92\xef\xbc\x93",
                    (int[]){47457, 37638}, 2, "live-nfkc");
    rc |= check_ids(tok, ws, "query: what is snowflake?",
                    (int[]){41, 1294, 12, 2367, 83, 108203, 13034, 350, 32}, 9,
                    "live-query");
    rc |= check_ids(tok, ws, "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80",
                    (int[]){1813, 18454, 11373}, 3, "live-cyrillic");
    rc |= check_ids(tok, ws,
                    "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85",
                    (int[]){665, 193478, 258, 1705, 77796}, 5, "live-arabic");
    rc |= check_ids(tok, ws, "", NULL, 0, "live-empty");
    rc |= check_ids(tok, ws, "   ", NULL, 0, "live-spaces");
    if (sentencepiece_tokenizer_token_id(tok, "<s>") != 0 ||
        sentencepiece_tokenizer_token_id(tok, "<pad>") != 1 ||
        sentencepiece_tokenizer_token_id(tok, "</s>") != 2 ||
        sentencepiece_tokenizer_token_id(tok, "<unk>") != 3 ||
        sentencepiece_tokenizer_token_id(tok, "<mask>") != 250001) {
        fprintf(stderr, "live special ids wrong\n");
        rc = 1;
    }
    sentencepiece_workspace_free(ws);
    sentencepiece_tokenizer_free(tok);
    return rc ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc > 1)
        return run_live(argv[1]);
    return run_hermetic();
}
