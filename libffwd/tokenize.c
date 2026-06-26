/*
 * Model-aware tokenization: text -> token ids the model expects.
 *
 * Backend-independent, so this uses its own handle, ffwd_tok_t, instead of the
 * per-backend ffwd_t.
 *
 * ffwd_tok_open selects the tokenizer from the files present and resolves the
 * model's special-token ids.
 *
 * ffwd_tokenize applies the model's token layout: BERT/XLM-R [CLS]..[SEP], or
 * Qwen appended terminal token.
 */

#include "ffwd.h"

#include "bpe.h"
#include "sentencepiece.h"
#include "wordpiece.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef enum { TOK_KIND_BPE, TOK_KIND_WP, TOK_KIND_SPM } tok_kind_t;

struct ffwd_tok {
    tok_kind_t kind;
    /* exactly one trio is populated, by kind */
    tok_bpe_t *bpe;
    tok_bpe_workspace_t *bpe_ws;
    tok_wp_t *wp;
    tok_wp_workspace_t *wp_ws;
    tok_spm_t *spm;
    tok_spm_workspace_t *spm_ws;

    int cls_id; /* [CLS] / <s>  — wp/spm wrap with these */
    int sep_id; /* [SEP] / </s> */
    int context_separator_id;
    int terminal_id;     /* appended on the bpe path when append_terminal */
    int append_terminal; /* Qwen3-Embedding pools the last token; see note */

    int is_late;
    int late_mask_id;
    int late_query_prefix_id;
    int late_document_prefix_id;
    int late_skip_ids[64];
    int n_late_skip_ids;
};

static void set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen)
        snprintf(err, errlen, "%s", msg);
}

ffwd_tok_t *ffwd_tok_open(const char *model_dir, int is_late, char *err, size_t errlen) {
    if (!model_dir) {
        set_err(err, errlen, "model_dir is NULL");
        return NULL;
    }
    ffwd_tok_t *t = calloc(1, sizeof(*t));
    if (!t) {
        set_err(err, errlen, "out of memory");
        return NULL;
    }
    t->cls_id = -1;
    t->sep_id = -1;
    t->terminal_id = -1;
    t->is_late = is_late;

    char path[1024];

    /* WordPiece (BERT family): vocab.txt. */
    snprintf(path, sizeof(path), "%s/vocab.txt", model_dir);
    if (access(path, R_OK) == 0) {
        t->kind = TOK_KIND_WP;
        t->wp = tok_wp_load(model_dir);
        if (!t->wp) {
            set_err(err, errlen, "failed to load WordPiece tokenizer");
            goto fail;
        }
        t->wp_ws = tok_wp_workspace_new();
        if (!t->wp_ws) {
            set_err(err, errlen, "out of memory");
            goto fail;
        }
        t->cls_id = tok_wp_token_id(t->wp, "[CLS]");
        t->sep_id = tok_wp_token_id(t->wp, "[SEP]");
        if (t->cls_id < 0 || t->sep_id < 0) {
            set_err(err, errlen, "WordPiece vocab missing [CLS]/[SEP]");
            goto fail;
        }
        return t;
    }

    /* SentencePiece (XLM-R family): sentencepiece.bpe.model / tokenizer.model /
     * spiece.model. */
    snprintf(path, sizeof(path), "%s/sentencepiece.bpe.model", model_dir);
    if (access(path, R_OK) != 0) {
        snprintf(path, sizeof(path), "%s/tokenizer.model", model_dir);
        if (access(path, R_OK) != 0)
            snprintf(path, sizeof(path), "%s/spiece.model", model_dir);
    }
    if (access(path, R_OK) == 0) {
        t->kind = TOK_KIND_SPM;
        t->spm = tok_spm_load(model_dir);
        if (!t->spm) {
            set_err(err, errlen, "failed to load SentencePiece tokenizer");
            goto fail;
        }
        t->spm_ws = tok_spm_workspace_new();
        if (!t->spm_ws) {
            set_err(err, errlen, "out of memory");
            goto fail;
        }
        t->cls_id = tok_spm_token_id(t->spm, "<s>");
        t->sep_id = tok_spm_token_id(t->spm, "</s>");
        if (t->cls_id < 0 || t->sep_id < 0) {
            set_err(err, errlen, "SentencePiece vocab missing <s>/</s>");
            goto fail;
        }
        return t;
    }

    /* Byte-level BPE (Qwen family): vocab.json. */
    snprintf(path, sizeof(path), "%s/vocab.json", model_dir);
    t->kind = TOK_KIND_BPE;
    t->bpe = tok_bpe_load(path);
    if (!t->bpe) {
        set_err(err, errlen, "failed to load tokenizer");
        goto fail;
    }
    t->bpe_ws = tok_bpe_workspace_new();
    if (!t->bpe_ws) {
        set_err(err, errlen, "out of memory");
        goto fail;
    }
    /* The contextual chunk separator is the tokenizer's <|endoftext|>; released
     * models keep it out of vocab.json, so the family constant is the fallback.
     * Qwen3-Embedding pools the last token (the same <|endoftext|>), so the bpe
     * path appends it. A config that disables append_terminal_token would need
     * to override this; the released models all enable it. */
    int sep = tok_bpe_token_id(t->bpe, "<|endoftext|>");
    t->context_separator_id = sep >= 0 ? sep : FFWD_CONTEXT_SEPARATOR_TOKEN_ID;
    t->terminal_id = t->context_separator_id;
    /* Off until the caller sets it from config->append_terminal_token; it is
     * model-specific (pplx-embed does not append, for example). */
    t->append_terminal = 0;

    if (is_late) {
        int id = tok_bpe_token_id(t->bpe, "[MASK]");
        t->late_mask_id = id >= 0 ? id : FFWD_LATE_MASK_TOKEN_ID;
        id = tok_bpe_token_id(t->bpe, "[Q]");
        t->late_query_prefix_id = id >= 0 ? id : FFWD_LATE_QUERY_PREFIX_ID;
        id = tok_bpe_token_id(t->bpe, "[D]");
        t->late_document_prefix_id = id >= 0 ? id : FFWD_LATE_DOCUMENT_PREFIX_ID;

        const char *punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
        for (const char *p = punct;
             *p && t->n_late_skip_ids < (int)(sizeof(t->late_skip_ids) / sizeof(t->late_skip_ids[0]));
             p++) {
            char text[2] = {*p, '\0'};
            int n_ids = 0;
            int *ids = tok_bpe_encode(t->bpe, text, &n_ids);
            if (ids && n_ids == 1)
                t->late_skip_ids[t->n_late_skip_ids++] = ids[0];
            free(ids);
        }
    }
    return t;

fail:
    ffwd_tok_free(t);
    return NULL;
}

void ffwd_tok_free(ffwd_tok_t *t) {
    if (!t)
        return;
    if (t->bpe_ws)
        tok_bpe_workspace_free(t->bpe_ws);
    if (t->bpe)
        tok_bpe_free(t->bpe);
    if (t->wp_ws)
        tok_wp_workspace_free(t->wp_ws);
    if (t->wp)
        tok_wp_free(t->wp);
    if (t->spm_ws)
        tok_spm_workspace_free(t->spm_ws);
    if (t->spm)
        tok_spm_free(t->spm);
    free(t);
}

int ffwd_tok_context_separator_id(const ffwd_tok_t *t) { return t ? t->context_separator_id : -1; }

void ffwd_tok_set_append_terminal(ffwd_tok_t *t, int on) {
    if (t)
        t->append_terminal = on ? 1 : 0;
}

/* Encode text and apply the model's special-token layout. Returns malloc'd ids
 * and sets *n_out; NULL on error. */
static int *tokenize_core(ffwd_tok_t *t, const char *text, int *n_out) {
    *n_out = 0;

    if (t->kind == TOK_KIND_WP || t->kind == TOK_KIND_SPM) {
        int n = 0;
        int *core = (t->kind == TOK_KIND_WP)
                        ? tok_wp_encode_with_workspace(t->wp, t->wp_ws, text, &n)
                        : tok_spm_encode_with_workspace(t->spm, t->spm_ws, text, &n);
        /* Wrap [CLS] core [SEP]. A whitespace-only input cleans to zero core
         * tokens (core == NULL, n == 0); the bare [CLS][SEP] pair is what
         * Hugging Face produces for it too. Grow in place so only the one
         * allocation is touched. */
        if (n < 0 || n > INT_MAX - 2) {
            free(core);
            return NULL;
        }
        int *ids = realloc(core, (size_t)(n + 2) * sizeof(*ids));
        if (!ids) {
            free(core);
            return NULL;
        }
        memmove(ids + 1, ids, (size_t)n * sizeof(*ids));
        ids[0] = t->cls_id;
        ids[n + 1] = t->sep_id;
        *n_out = n + 2;
        return ids;
    }

    /* BPE. */
    int n = 0;
    int *ids = tok_bpe_encode_with_workspace(t->bpe, t->bpe_ws, text, &n);
    if (!ids || n <= 0) {
        free(ids);
        return NULL;
    }
    if (t->append_terminal) {
        if (n == INT_MAX) {
            free(ids);
            return NULL;
        }
        int *grown = realloc(ids, (size_t)(n + 1) * sizeof(*ids));
        if (!grown) {
            free(ids);
            return NULL;
        }
        ids = grown;
        ids[n++] = t->terminal_id;
    }
    *n_out = n;
    return ids;
}

int *ffwd_tokenize(ffwd_tok_t *t, const char *text, const char *query_instruct, int *n_out) {
    if (n_out)
        *n_out = 0;
    if (!t || !text)
        return NULL;

    int n = 0;
    int *ids;
    if (!query_instruct) {
        ids = tokenize_core(t, text, &n);
    } else {
        /* Tokenize the instruction and the input in one pass. */
        size_t plen = strlen(query_instruct);
        size_t tlen = strlen(text);
        char *buf = malloc(plen + tlen + 1);
        if (!buf)
            return NULL;
        memcpy(buf, query_instruct, plen);
        memcpy(buf + plen, text, tlen + 1);
        ids = tokenize_core(t, buf, &n);
        free(buf);
    }
    if (!ids)
        return NULL;
    if (n_out)
        *n_out = n;
    return ids;
}

static int late_id_is_skipped(const ffwd_tok_t *t, int id) {
    for (int i = 0; i < t->n_late_skip_ids; i++) {
        if (t->late_skip_ids[i] == id)
            return 1;
    }
    return 0;
}

int ffwd_tokenize_late(ffwd_tok_t *t, const char *text, int is_query, ffwd_late_tokens_t *out) {
    memset(out, 0, sizeof(*out));
    if (!t || !text)
        return -1;

    int raw_n = 0;
    int *raw = tokenize_core(t, text, &raw_n);
    if (!raw)
        return -1;

    int target = raw_n + 1;
    if (is_query && target < FFWD_LATE_QUERY_TOKENS)
        target = FFWD_LATE_QUERY_TOKENS;

    out->ids = malloc((size_t)target * sizeof(*out->ids));
    if (!out->ids) {
        free(raw);
        return -1;
    }
    out->ids[0] = raw[0];
    out->ids[1] = is_query ? t->late_query_prefix_id : t->late_document_prefix_id;
    if (raw_n > 1)
        memcpy(out->ids + 2, raw + 1, (size_t)(raw_n - 1) * sizeof(*out->ids));
    out->n_tokens = raw_n + 1;
    free(raw);

    if (is_query) {
        while (out->n_tokens < FFWD_LATE_QUERY_TOKENS)
            out->ids[out->n_tokens++] = t->late_mask_id;
        out->n_keep = out->n_tokens;
    } else {
        out->keep = malloc((size_t)out->n_tokens * sizeof(*out->keep));
        if (!out->keep) {
            ffwd_late_tokens_free(out);
            return -1;
        }
        for (int i = 0; i < out->n_tokens; i++) {
            if (!late_id_is_skipped(t, out->ids[i]))
                out->keep[out->n_keep++] = i;
        }
    }

    if (out->n_keep <= 0) {
        ffwd_late_tokens_free(out);
        return -1;
    }
    return 0;
}

void ffwd_late_tokens_free(ffwd_late_tokens_t *out) {
    if (!out)
        return;
    free(out->ids);
    free(out->keep);
    memset(out, 0, sizeof(*out));
}
