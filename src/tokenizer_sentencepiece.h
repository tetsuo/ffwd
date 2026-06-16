/*
 * tokenizer_sentencepiece.h - SentencePiece tokenizer for XLM-R style models.
 *
 * Loads a SentencePiece ModelProto (`sentencepiece.bpe.model`, `tokenizer.model`,
 * or `spiece.model`) and implements the Unigram Viterbi encoder used by XLM-R
 * and multilingual E5. It encodes text only; frontends add the model's special
 * token layout, e.g. <s> ... </s> for XLM-R.
 */

#ifndef SENTENCEPIECE_TOKENIZER_H
#define SENTENCEPIECE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifndef EMBED_API
#if defined(__GNUC__)
#define EMBED_API __attribute__((visibility("default")))
#else
#define EMBED_API
#endif
#endif

typedef enum {
    SENTENCEPIECE_IDENTITY = 0,
    SENTENCEPIECE_XLM_ROBERTA = 1,
} sentencepiece_id_map_t;

typedef struct {
    char **id_to_piece;      /* [piece_count] raw SentencePiece ids */
    uint16_t *piece_lens;    /* byte length for id_to_piece entries */
    float *scores;           /* SentencePiece score per raw id */
    uint8_t *types;          /* SentencePiece piece type per raw id */
    int piece_count;         /* raw SentencePiece vocabulary size */
    int vocab_size;          /* model embedding vocab size after id mapping */
    int unk_id;              /* model-space unknown-token id */
    int bos_id;              /* model-space beginning-token id, or -1 */
    int eos_id;              /* model-space end-token id, or -1 */
    int pad_id;              /* model-space pad-token id, or -1 */
    int mask_id;             /* model-space mask-token id, or -1 */
    int max_piece_len;       /* max normal/user/unused piece byte length */
    double unknown_score;    /* score assigned to an unmatched UTF-8 codepoint */
    int add_dummy_prefix;
    int remove_extra_whitespaces;
    int escape_whitespaces;
    sentencepiece_id_map_t id_map;

    void *piece_map; /* piece bytes -> raw SentencePiece id */
    int piece_map_cap;

    uint32_t *xcda;
    size_t xcda_count;
    char *prefix_replacements;
    size_t prefix_replacements_len;
} sentencepiece_tokenizer_t;

typedef struct sentencepiece_workspace sentencepiece_workspace_t;

/* Load the first supported SentencePiece model found in model_dir. Returns NULL
 * on missing model file, unsupported model type, malformed protobuf, or OOM. */
EMBED_API sentencepiece_tokenizer_t *sentencepiece_tokenizer_load(const char *model_dir);

/* Decode one model-space id to an internal token string, or NULL when invalid. */
EMBED_API const char *sentencepiece_tokenizer_decode(const sentencepiece_tokenizer_t *tok, int id);

/* Resolve a named token (e.g. "<s>", "</s>", "<pad>", "<mask>") to a
 * model-space id, or -1 when absent. */
EMBED_API int sentencepiece_tokenizer_token_id(const sentencepiece_tokenizer_t *tok,
                                               const char *token);

/* Encode UTF-8 text into malloc'd model-space token ids. Empty text returns
 * NULL with *out_n_tokens = 0. NULL with a negative/unspecified count is an
 * error. */
EMBED_API int *sentencepiece_tokenizer_encode(const sentencepiece_tokenizer_t *tok,
                                              const char *text,
                                              int *out_n_tokens);

EMBED_API sentencepiece_workspace_t *sentencepiece_workspace_new(void);
EMBED_API void sentencepiece_workspace_free(sentencepiece_workspace_t *ws);

/* Encode into caller-provided out_ids. Returns 0 on success, -1 on error, -2 if
 * out_cap is too small. *out_n_tokens receives the emitted/needed token count. */
EMBED_API int sentencepiece_tokenizer_encode_into(const sentencepiece_tokenizer_t *tok,
                                                  sentencepiece_workspace_t *ws,
                                                  const char *text,
                                                  int *out_ids,
                                                  int out_cap,
                                                  int *out_n_tokens);

/* Same tokenization as sentencepiece_tokenizer_encode(), reusing ws internally
 * and performing only the final returned int[] allocation. */
EMBED_API int *sentencepiece_tokenizer_encode_with_workspace(const sentencepiece_tokenizer_t *tok,
                                                             sentencepiece_workspace_t *ws,
                                                             const char *text,
                                                             int *out_n_tokens);

EMBED_API void sentencepiece_tokenizer_free(sentencepiece_tokenizer_t *tok);

#endif /* SENTENCEPIECE_TOKENIZER_H */
