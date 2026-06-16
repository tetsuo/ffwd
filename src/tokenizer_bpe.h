/*
 * embed_tokenizer.h - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 *
 * Supports:
 *  - decode token ID   -> UTF-8 text
 *  - encode UTF-8 text -> token IDs (vocab.json + merges.txt)
 */

#ifndef EMBED_TOKENIZER_H
#define EMBED_TOKENIZER_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying EMBED_API are exported
 * from the shared library; everything else stays internal. */
#ifndef EMBED_API
#if defined(__GNUC__)
#define EMBED_API __attribute__((visibility("default")))
#else
#define EMBED_API
#endif
#endif

typedef struct {
    char **id_to_text; /* [vocab_size] decoded text strings */
    char **id_to_bpe;  /* [vocab_size] raw BPE token strings from vocab.json */
    int vocab_size;

    /* Internal hash maps (opaque to callers) */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
    /* Integer-pair BPE merge table: (left_id, right_id) -> (rank, merged_id).
     * NULL when the vocab could not resolve every merge; encoding then falls
     * back to the string-based merge loop. */
    void *int_merges;
    int int_merges_cap;
    /* Byte-level unicode codepoint -> single-character token id (512 slots),
     * or NULL when any byte token is missing from the vocab. */
    int *cp_to_id;
} embed_tokenizer_t;

typedef struct embed_tokenizer_workspace embed_tokenizer_workspace_t;

/* Load tokenizer from vocab.json in model directory */
EMBED_API embed_tokenizer_t *embed_tokenizer_load(const char *vocab_json_path);

/* Decode a single token ID to text. Returns pointer to internal string. */
EMBED_API const char *embed_tokenizer_decode(const embed_tokenizer_t *tok, int token_id);

/* Look up the id of a raw vocab.json token string (e.g. "<|endoftext|>").
 * Returns -1 when the token is not in the vocab. */
EMBED_API int embed_tokenizer_token_id(const embed_tokenizer_t *tok, const char *token);

/* Encode UTF-8 text into token IDs using BPE.
 * Returns malloc'd array of token IDs and sets *out_n_tokens.
 * Returns NULL on error (and sets *out_n_tokens to 0). */
EMBED_API int *
embed_tokenizer_encode(const embed_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* Reusable scratch for allocation-light repeated encoding. */
EMBED_API embed_tokenizer_workspace_t *embed_tokenizer_workspace_new(void);
EMBED_API void embed_tokenizer_workspace_free(embed_tokenizer_workspace_t *ws);

/* Encode into caller-provided out_ids[0..out_cap-1].
 * Returns 0 on success, -1 on error, -2 if out_cap is too small.
 * In all cases *out_n_tokens is set to the number of tokens needed/emitted
 * when out_n_tokens is non-NULL. */
EMBED_API int embed_tokenizer_encode_into(const embed_tokenizer_t *tok,
                                         embed_tokenizer_workspace_t *ws,
                                         const char *text,
                                         int *out_ids,
                                         int out_cap,
                                         int *out_n_tokens);

/* Same tokenization as embed_tokenizer_encode(), but reuses ws internally and
 * performs only the final returned int[] allocation. */
EMBED_API int *embed_tokenizer_encode_with_workspace(const embed_tokenizer_t *tok,
                                                    embed_tokenizer_workspace_t *ws,
                                                    const char *text,
                                                    int *out_n_tokens);

/* Free tokenizer */
EMBED_API void embed_tokenizer_free(embed_tokenizer_t *tok);

#endif /* EMBED_TOKENIZER_H */
