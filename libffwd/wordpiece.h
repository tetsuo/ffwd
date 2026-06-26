/*
 * BERT-family WordPiece tokenizer.
 *
 * Implements BERT tokenization: BasicTokenizer (cleanup, optional lowercase
 * and accent stripping, CJK spacing, whitespace splitting, and punctuation
 * splitting), then greedy longest-match WordPiece using vocab.txt.
 *
 * Encodes text only. The frontend adds [CLS]/[SEP], just as the byte-level BPE
 * backend leaves terminal and separator tokens to the frontend.
 *
 * Public API mirrors bpe.h: load, encode with reusable workspace, decode,
 * resolve named special tokens, and free. This lets both tokenizers share one
 * tokenizer interface later.
 */

#ifndef TOK_WP_H
#define TOK_WP_H

typedef struct {
    char **id_to_token; /* [vocab_size] vocab.txt entries, line order = id */
    int vocab_size;
    int unk_id;             /* [UNK] id, resolved at load */
    int do_lower_case;      /* lowercase + strip accents before WordPiece */
    int max_chars_per_word; /* words longer than this map to [UNK] (BERT: 100) */

    /* Opaque token-string -> id hash map. */
    void *vocab_map;
    int vocab_map_cap;
} tok_wp_t;

typedef struct tok_wp_workspace tok_wp_workspace_t;

/* Load <model_dir>/vocab.txt and do_lower_case from
 * <model_dir>/tokenizer_config.json; defaults to lowercase.
 * Returns NULL on error: missing vocab.txt, missing [UNK], or allocation failure. */
tok_wp_t *tok_wp_load(const char *model_dir);

/* Decode one id to its vocab token string, or NULL.
 * Returns an internal pointer. */
const char *tok_wp_decode(const tok_wp_t *tok, int id);

/* Resolve a named token (e.g. "[CLS]") to its id, or -1 when absent. */
int tok_wp_token_id(const tok_wp_t *tok, const char *token);

/* Encode UTF-8 text into a malloc'd id array and set *out_n_tokens.
 * Returns NULL on error and sets *out_n_tokens = 0. */
int *tok_wp_encode(const tok_wp_t *tok, const char *text, int *out_n_tokens);

/* Reusable scratch space for repeated encoding with fewer allocations. */
tok_wp_workspace_t *tok_wp_workspace_new(void);
void tok_wp_workspace_free(tok_wp_workspace_t *ws);

/* Encode into caller-provided out_ids[0..out_cap-1].
 * Returns 0 on success, -1 on error, or -2 if out_cap is too small.
 * If non-NULL, *out_n_tokens is always set to tokens needed/emitted. */
int tok_wp_encode_into(const tok_wp_t *tok,
                       tok_wp_workspace_t *ws,
                       const char *text,
                       int *out_ids,
                       int out_cap,
                       int *out_n_tokens);

/* Same tokenization as tok_wp_encode().
 * Reuses ws internally and only allocates the returned int[]. */
int *tok_wp_encode_with_workspace(const tok_wp_t *tok,
                                  tok_wp_workspace_t *ws,
                                  const char *text,
                                  int *out_n_tokens);

void tok_wp_free(tok_wp_t *tok);

#endif /* TOK_WP_H */
