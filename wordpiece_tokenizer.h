/*
 * wordpiece_tokenizer.h - BERT-family WordPiece tokenizer.
 *
 * Implements the BERT tokenization pipeline: a BasicTokenizer (clean, optional
 * lowercase + accent strip, CJK spacing, whitespace and punctuation splitting)
 * followed by greedy longest-match WordPiece against a `vocab.txt`. It encodes
 * text only; the frontend wraps the result with [CLS]/[SEP], exactly as the
 * byte-level BPE backend leaves terminal/separator tokens to the frontend.
 *
 * The public surface mirrors qwen_tokenizer.h (load / encode with a reusable
 * workspace / decode / resolve a named special token / free) so both can sit
 * behind one tokenizer interface later.
 */

#ifndef WORDPIECE_TOKENIZER_H
#define WORDPIECE_TOKENIZER_H

#ifndef EMBED_API
#if defined(__GNUC__)
#define EMBED_API __attribute__((visibility("default")))
#else
#define EMBED_API
#endif
#endif

typedef struct {
    char **id_to_token; /* [vocab_size] vocab.txt entries, line order = id */
    int vocab_size;
    int unk_id;             /* [UNK] id, resolved at load */
    int do_lower_case;      /* lowercase + strip accents before WordPiece */
    int max_chars_per_word; /* words longer than this map to [UNK] (BERT: 100) */

    /* Opaque token-string -> id hash map. */
    void *vocab_map;
    int vocab_map_cap;
} wordpiece_tokenizer_t;

typedef struct wordpiece_workspace wordpiece_workspace_t;

/* Load from <model_dir>/vocab.txt, reading do_lower_case from
 * <model_dir>/tokenizer_config.json (default: lowercase). Returns NULL on
 * error (missing vocab.txt, missing [UNK], allocation failure). */
EMBED_API wordpiece_tokenizer_t *wordpiece_tokenizer_load(const char *model_dir);

/* Decode a single id to its vocab token string (internal pointer), or NULL. */
EMBED_API const char *wordpiece_tokenizer_decode(const wordpiece_tokenizer_t *tok, int id);

/* Resolve a named token (e.g. "[CLS]") to its id, or -1 when absent. */
EMBED_API int wordpiece_tokenizer_token_id(const wordpiece_tokenizer_t *tok, const char *token);

/* Encode UTF-8 text into a malloc'd id array; sets *out_n_tokens. NULL on
 * error (and *out_n_tokens = 0). */
EMBED_API int *
wordpiece_tokenizer_encode(const wordpiece_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* Reusable scratch for allocation-light repeated encoding. */
EMBED_API wordpiece_workspace_t *wordpiece_workspace_new(void);
EMBED_API void wordpiece_workspace_free(wordpiece_workspace_t *ws);

/* Encode into caller-provided out_ids[0..out_cap-1]. Returns 0 on success, -1
 * on error, -2 if out_cap is too small. *out_n_tokens (when non-NULL) is set to
 * the number of tokens needed/emitted in all cases. */
EMBED_API int wordpiece_tokenizer_encode_into(const wordpiece_tokenizer_t *tok,
                                              wordpiece_workspace_t *ws,
                                              const char *text,
                                              int *out_ids,
                                              int out_cap,
                                              int *out_n_tokens);

/* Same tokenization as wordpiece_tokenizer_encode(), reusing ws internally and
 * performing only the final returned int[] allocation. */
EMBED_API int *wordpiece_tokenizer_encode_with_workspace(const wordpiece_tokenizer_t *tok,
                                                         wordpiece_workspace_t *ws,
                                                         const char *text,
                                                         int *out_n_tokens);

EMBED_API void wordpiece_tokenizer_free(wordpiece_tokenizer_t *tok);

#endif /* WORDPIECE_TOKENIZER_H */
