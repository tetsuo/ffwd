/*
 * wordpiece.h - BERT-family WordPiece tokenizer.
 *
 * Implements the BERT tokenization pipeline: a BasicTokenizer (clean, optional
 * lowercase + accent strip, CJK spacing, whitespace and punctuation splitting)
 * followed by greedy longest-match WordPiece against a `vocab.txt`. It encodes
 * text only; the frontend wraps the result with [CLS]/[SEP], exactly as the
 * byte-level BPE backend leaves terminal/separator tokens to the frontend.
 *
 * The public surface mirrors bpe.h (load / encode with a reusable
 * workspace / decode / resolve a named special token / free) so both can sit
 * behind one tokenizer interface later.
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

/* Load from <model_dir>/vocab.txt, reading do_lower_case from
 * <model_dir>/tokenizer_config.json (default: lowercase). Returns NULL on
 * error (missing vocab.txt, missing [UNK], allocation failure). */
tok_wp_t *tok_wp_load(const char *model_dir);

/* Decode a single id to its vocab token string (internal pointer), or NULL. */
const char *tok_wp_decode(const tok_wp_t *tok, int id);

/* Resolve a named token (e.g. "[CLS]") to its id, or -1 when absent. */
int tok_wp_token_id(const tok_wp_t *tok, const char *token);

/* Encode UTF-8 text into a malloc'd id array; sets *out_n_tokens. NULL on
 * error (and *out_n_tokens = 0). */
int *tok_wp_encode(const tok_wp_t *tok, const char *text, int *out_n_tokens);

/* Reusable scratch for allocation-light repeated encoding. */
tok_wp_workspace_t *tok_wp_workspace_new(void);
void tok_wp_workspace_free(tok_wp_workspace_t *ws);

/* Encode into caller-provided out_ids[0..out_cap-1]. Returns 0 on success, -1
 * on error, -2 if out_cap is too small. *out_n_tokens (when non-NULL) is set to
 * the number of tokens needed/emitted in all cases. */
int tok_wp_encode_into(const tok_wp_t *tok,
                       tok_wp_workspace_t *ws,
                       const char *text,
                       int *out_ids,
                       int out_cap,
                       int *out_n_tokens);

/* Same tokenization as tok_wp_encode(), reusing ws internally and
 * performing only the final returned int[] allocation. */
int *tok_wp_encode_with_workspace(const tok_wp_t *tok,
                                  tok_wp_workspace_t *ws,
                                  const char *text,
                                  int *out_n_tokens);

void tok_wp_free(tok_wp_t *tok);

#endif /* TOK_WP_H */
