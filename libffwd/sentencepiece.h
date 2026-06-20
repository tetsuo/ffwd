/*
 * sentencepiece.h - SentencePiece tokenizer for XLM-R style models.
 *
 * Loads a SentencePiece ModelProto (`sentencepiece.bpe.model`, `tokenizer.model`,
 * or `spiece.model`) and implements the Unigram Viterbi encoder used by XLM-R
 * and multilingual E5. It encodes text only; frontends add the model's special
 * token layout, e.g. <s> ... </s> for XLM-R.
 */

#ifndef TOK_SPM_H
#define TOK_SPM_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TOK_SPM_IDENTITY = 0,
    TOK_SPM_XLM_ROBERTA = 1,
} tok_spm_id_map_t;

typedef struct {
    char **id_to_piece;   /* [piece_count] raw SentencePiece ids */
    uint16_t *piece_lens; /* byte length for id_to_piece entries */
    float *scores;        /* SentencePiece score per raw id */
    uint8_t *types;       /* SentencePiece piece type per raw id */
    int piece_count;      /* raw SentencePiece vocabulary size */
    int vocab_size;       /* model embedding vocab size after id mapping */
    int unk_id;           /* model-space unknown-token id */
    int bos_id;           /* model-space beginning-token id, or -1 */
    int eos_id;           /* model-space end-token id, or -1 */
    int pad_id;           /* model-space pad-token id, or -1 */
    int mask_id;          /* model-space mask-token id, or -1 */
    int max_piece_len;    /* max normal/user/unused piece byte length */
    double unknown_score; /* score assigned to an unmatched UTF-8 codepoint */
    int add_dummy_prefix;
    int remove_extra_whitespaces;
    int escape_whitespaces;
    tok_spm_id_map_t id_map;

    void *piece_map; /* piece bytes -> raw SentencePiece id */
    int piece_map_cap;

    uint32_t *xcda;
    size_t xcda_count;
    char *prefix_replacements;
    size_t prefix_replacements_len;
} tok_spm_t;

typedef struct tok_spm_workspace tok_spm_workspace_t;

/* Load the first supported SentencePiece model found in model_dir. Returns NULL
 * on missing model file, unsupported model type, malformed protobuf, or OOM. */
tok_spm_t *tok_spm_load(const char *model_dir);

/* Decode one model-space id to an internal token string, or NULL when invalid. */
const char *tok_spm_decode(const tok_spm_t *tok, int id);

/* Resolve a named token (e.g. "<s>", "</s>", "<pad>", "<mask>") to a
 * model-space id, or -1 when absent. */
int tok_spm_token_id(const tok_spm_t *tok, const char *token);

/* Encode UTF-8 text into malloc'd model-space token ids. Empty text returns
 * NULL with *out_n_tokens = 0. NULL with a negative/unspecified count is an
 * error. */
int *tok_spm_encode(const tok_spm_t *tok, const char *text, int *out_n_tokens);

tok_spm_workspace_t *tok_spm_workspace_new(void);
void tok_spm_workspace_free(tok_spm_workspace_t *ws);

/* Encode into caller-provided out_ids. Returns 0 on success, -1 on error, -2 if
 * out_cap is too small. *out_n_tokens receives the emitted/needed token count. */
int tok_spm_encode_into(const tok_spm_t *tok,
                        tok_spm_workspace_t *ws,
                        const char *text,
                        int *out_ids,
                        int out_cap,
                        int *out_n_tokens);

/* Same tokenization as tok_spm_encode(), reusing ws internally
 * and performing only the final returned int[] allocation. */
int *tok_spm_encode_with_workspace(const tok_spm_t *tok,
                                   tok_spm_workspace_t *ws,
                                   const char *text,
                                   int *out_n_tokens);

void tok_spm_free(tok_spm_t *tok);

#endif /* TOK_SPM_H */
