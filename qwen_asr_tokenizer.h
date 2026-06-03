/*
 * qwen_asr_tokenizer.h - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 *
 * Supports:
 *  - decode token ID   -> UTF-8 text
 *  - encode UTF-8 text -> token IDs (vocab.json + merges.txt)
 */

#ifndef QWEN_ASR_TOKENIZER_H
#define QWEN_ASR_TOKENIZER_H

typedef struct {
    char **id_to_text;   /* [vocab_size] decoded text strings */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings from vocab.json */
    int vocab_size;

    /* Internal hash maps (opaque to callers) */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
} qwen_tokenizer_t;

typedef struct qwen_tokenizer_workspace qwen_tokenizer_workspace_t;

/* Load tokenizer from vocab.json in model directory */
qwen_tokenizer_t *qwen_tokenizer_load(const char *vocab_json_path);

/* Decode a single token ID to text. Returns pointer to internal string. */
const char *qwen_tokenizer_decode(const qwen_tokenizer_t *tok, int token_id);

/* Encode UTF-8 text into token IDs using BPE.
 * Returns malloc'd array of token IDs and sets *out_n_tokens.
 * Returns NULL on error (and sets *out_n_tokens to 0). */
int *qwen_tokenizer_encode(const qwen_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* Reusable scratch for allocation-light repeated encoding. */
qwen_tokenizer_workspace_t *qwen_tokenizer_workspace_new(void);
void qwen_tokenizer_workspace_free(qwen_tokenizer_workspace_t *ws);

/* Encode into caller-provided out_ids[0..out_cap-1].
 * Returns 0 on success, -1 on error, -2 if out_cap is too small.
 * In all cases *out_n_tokens is set to the number of tokens needed/emitted
 * when out_n_tokens is non-NULL. */
int qwen_tokenizer_encode_into(const qwen_tokenizer_t *tok,
                               qwen_tokenizer_workspace_t *ws,
                               const char *text,
                               int *out_ids, int out_cap,
                               int *out_n_tokens);

/* Same tokenization as qwen_tokenizer_encode(), but reuses ws internally and
 * performs only the final returned int[] allocation. */
int *qwen_tokenizer_encode_with_workspace(const qwen_tokenizer_t *tok,
                                          qwen_tokenizer_workspace_t *ws,
                                          const char *text,
                                          int *out_n_tokens);

/* Free tokenizer */
void qwen_tokenizer_free(qwen_tokenizer_t *tok);

#endif /* QWEN_ASR_TOKENIZER_H */
