/*
 * ffwd.h - ffwd API
 */

#ifndef FFWD_H
#define FFWD_H

/* Public-API export annotation. The libraries are built with
 * -fvisibility=hidden, so only declarations carrying FFWD_API are exported
 * from the shared library; everything else stays internal. */
#ifndef FFWD_API
#    if defined(__GNUC__)
#        define FFWD_API __attribute__((visibility("default")))
#    else
#        define FFWD_API
#    endif
#endif

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Shared constants and family defaults
 * ======================================================================== */

#define FFWD_VOCAB_SIZE                 151936
#define FFWD_HEAD_DIM                   128
#define FFWD_MAX_LAYERS                 64     /* upper bound for stack arrays */
#define FFWD_CONTEXT_SEPARATOR_TOKEN_ID 151643 /* <|endoftext|> */

/* Late-interaction (ColBERT) special-token id fallbacks and the query pad
 * length, used when a model's tokenizer does not define the tokens itself. */
#define FFWD_LATE_QUERY_TOKENS       32
#define FFWD_LATE_MASK_TOKEN_ID      151642
#define FFWD_LATE_QUERY_PREFIX_ID    151669
#define FFWD_LATE_DOCUMENT_PREFIX_ID 151670

/* ========================================================================
 * Model Configuration (populated from config.json at load time)
 * ======================================================================== */

typedef enum {
    FFWD_ATTENTION_BIDIRECTIONAL = 0,
    FFWD_ATTENTION_CAUSAL = 1,
} ffwd_attention_mode_t;

typedef enum {
    FFWD_POOL_MEAN = 0,
    FFWD_POOL_LAST_TOKEN = 1,
    FFWD_POOL_CLS = 2, /* first token; the encoder-family [CLS] sentence vector */
} ffwd_pooling_mode_t;

typedef enum {
    FFWD_FAMILY_QWEN3 = 0, /* RMSNorm + RoPE + SwiGLU decoder block (Qwen2/3, pplx-embed) */
    FFWD_FAMILY_BERT = 1,  /* LayerNorm + learned positions + GeLU encoder block */
} ffwd_family_t;

/* BERT feed-forward activation. The released MiniLM/BGE encoders use exact erf
 * GeLU; some encoders publish the tanh approximation (hidden_act gelu_new or
 * gelu_pytorch_tanh), which is a different curve and must be matched exactly. */
typedef enum {
    FFWD_ACT_GELU_ERF = 0,  /* 0.5 x (1 + erf(x / sqrt(2))) */
    FFWD_ACT_GELU_TANH = 1, /* 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) */
} ffwd_act_t;

typedef struct {
    ffwd_family_t family;
    ffwd_act_t ffn_act; /* BERT feed-forward GeLU variant; ERF for Qwen/SwiGLU */
    int hidden_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int q_dim;  /* n_heads    * head_dim */
    int kv_dim; /* n_kv_heads * head_dim */
    int intermediate_size;
    int vocab_size;
    int max_position_embeddings; /* BERT learned position table size */
    int position_id_offset;      /* RoBERTa/XLM-R positions start at pad_id + 1 */
    int type_vocab_size;         /* BERT token-type embedding rows; XLM-R uses 1 */
    float rms_norm_eps;
    float layer_norm_eps; /* BERT LayerNorm epsilon */
    float rope_theta;
    int qk_norm;
    int qkv_bias;
    ffwd_attention_mode_t attention_mode;
    ffwd_pooling_mode_t pooling_mode;
    int normalize_embeddings;
    /* Qwen3-Embedding pools the last token, which is the tokenizer's
     * <|endoftext|> suffix added by its post-processor - not the model's chat
     * eos_token_id (<|im_end|>). Text frontends append it and resolve the id
     * from the tokenizer exactly like the contextual separator
     * (FFWD_CONTEXT_SEPARATOR_TOKEN_ID is the released-model fallback). The
     * model forward consumes token ids exactly and never appends. */
    int append_terminal_token;
} ffwd_config_t;

typedef struct {
    const int *ids;
    int n_tokens;
} ffwd_input_t;

typedef struct {
    int start;
    int n_tokens;
} ffwd_span_t;

typedef struct {
    ffwd_input_t input;
    const ffwd_span_t *spans;
    int n_spans;
} ffwd_context_input_t;

/* ========================================================================
 * Engine API — the backend-selected inference handle
 *
 * This is the library's primary public interface and the entire supported
 * surface of the shared library. Exactly one backend (CPU, Apple Silicon GPU,
 * or NVIDIA GPU) is linked at build time and implements these functions;
 * callers never branch on the backend. The CLI, the server, and any language
 * binding use only these functions plus the math helpers below.
 *
 * A model loads in two phases so a GPU context can be created on the thread
 * that will own it. ffwd_open runs on the caller's thread and loads a CPU
 * model immediately; a GPU build only records what to load. ffwd_activate then
 * runs on the inference thread and creates the GPU context (a no-op for CPU).
 * Teardown mirrors this: ffwd_worker_free releases the GPU context from the
 * inference thread, ffwd_free releases the rest.
 * ======================================================================== */

/* Opaque inference handle; its layout is private to the linked backend. */
typedef struct embed ffwd_t;

/* Backend tuning, parsed from the command line. A backend reads the fields it
 * supports and ignores the rest, so the front-ends parse them with no backend
 * branch. */
typedef struct {
    int n_threads;                /* CPU worker threads; <= 0 auto-detects */
    int gpu_quant_bits;           /* GPU load-time weight quantization (0 or 8) */
    int gpu_quant_group_size;     /* GPU quantization group size (> 0) */
    const char *gpu_gemm_mode;    /* GPU GEMM compute mode, or NULL */
    const char *gpu_weight_dtype; /* GPU weight storage dtype, or NULL */
    double memory_utilization;    /* GPU memory budget fraction; 0 = default */
} ffwd_options_t;

/* One late-interaction rerank: a query and N candidate documents, each given as
 * token ids plus the kept-token indices. The vector dimension is the loaded
 * late model's token_dim. */
typedef struct {
    const int *query_ids;
    int query_n_tokens;
    int query_n_keep;
    const int *const *doc_ids;
    const int *doc_n_tokens;
    const int *const *doc_keep;
    const int *doc_n_keep;
    int n_docs;
} ffwd_rerank_input_t;

/* Apply process-wide settings before any load (CPU thread count, GPU GEMM and
 * weight settings, GPU option checks) and print the startup banner. Returns -1 and
 * writes a reason into err on a rejected option value. */
FFWD_API int ffwd_init(const ffwd_options_t *opts, char *err, size_t errlen);

/* First load phase, on the caller's thread: load a CPU model now, or record what
 * a GPU build will load in ffwd_activate. is_late selects a late-interaction
 * model. NULL on failure with a reason in err. */
FFWD_API ffwd_t *ffwd_open(
    const char *model_dir, int is_late, const ffwd_options_t *opts, char *err, size_t errlen);
/* Second load phase, on the inference thread: create the GPU context. CPU
 * no-op. Returns -1 with a reason in err on failure. */
FFWD_API int ffwd_activate(ffwd_t *e, char *err, size_t errlen);
/* Release the inference-thread-owned GPU context; CPU no-op. Idempotent. */
FFWD_API void ffwd_worker_free(ffwd_t *e);
/* Release CPU state and the handle; call after ffwd_worker_free. */
FFWD_API void ffwd_free(ffwd_t *e);

/* Available once the model is loaded (after open for CPU, after activate for
 * GPU); config is NULL until then. */
FFWD_API const ffwd_config_t *ffwd_config(const ffwd_t *e);
/* Late projection dimension, or 0 for a standard model. */
FFWD_API int ffwd_token_dim(const ffwd_t *e);
/* Whether the backend pads batches to a common length rather than packing them. */
FFWD_API int ffwd_uses_dense_batches(const ffwd_t *e);

/* Encode a packed batch into out[batch, hidden_size]. Returns 0 on success. */
FFWD_API int
ffwd_encode_batch(ffwd_t *e, const ffwd_input_t *inputs, int batch, float *out);
/* Encode a contextual document batch, pooling each selected span in order. */
FFWD_API int ffwd_encode_spans_batch(ffwd_t *e,
                                           const ffwd_context_input_t *inputs,
                                           int batch,
                                           float *out);
/* Score one query against its candidate documents into scores[n_docs]. */
FFWD_API int ffwd_rerank(ffwd_t *e, const ffwd_rerank_input_t *in, float *scores);
/* Scratch memory the model currently holds, for diagnostics; 0 on GPU. */
FFWD_API size_t ffwd_scratch_nbytes(const ffwd_t *e);

/* Capability label for --build-info and the startup banner: "CPU", "Apple
 * Silicon GPU", or "NVIDIA GPU". A function, not a macro, so a shared object
 * does not bake in one backend's label and stays correct across a rebuild. */
FFWD_API const char *ffwd_capability(void);

/* First-arrival micro-batch wait the backend prefers, in microseconds. */
FFWD_API int ffwd_default_batch_wait_us(void);
/* Reject a model set that would not fit host memory (GPU backends); 0 otherwise. On
 * rejection returns -1 with a reason in err. paths are the model directories. */
FFWD_API int ffwd_preflight(
    const char *const *paths, int n_paths, const ffwd_options_t *opts, char *err, size_t errlen);

/*
 * L2-normalize vec[dim] in place.
 *
 * Returns 0 on success. Returns -1 for invalid arguments or a zero-length
 * vector, which cannot be normalized.
 */
FFWD_API int ffwd_l2_normalize(float *vec, int dim);

/*
 * Cosine similarity between two vectors.
 * Returns 0 for invalid arguments or if either vector has zero length.
 */
FFWD_API float ffwd_cosine_similarity(const float *a, const float *b, int dim);

/* ========================================================================
 * Tokenization — text into the token ids a model expects
 *
 * The tokenizer is backend-independent (text -> ids is identical on CPU or
 * GPU), so it is its own handle, separate from ffwd_t. ffwd_tok_open picks the
 * tokenizer by the files in model_dir (WordPiece vocab.txt, SentencePiece
 * .model, or byte-level vocab.json) and resolves the model's special-token ids.
 * ffwd_tokenize then applies that model's layout: BERT and XLM-R wrap the ids
 * with [CLS]..[SEP] (<s>..</s>), Qwen appends its terminal token.
 * ======================================================================== */

/* Opaque tokenizer handle; owns the tokenizer and its resolved special tokens. */
typedef struct ffwd_tok ffwd_tok_t;

/* Late-interaction tokens: ids, plus the document positions to score. For a
 * query, keep is NULL (every position is kept) and n_keep == n_tokens. */
typedef struct {
    int *ids;
    int n_tokens;
    int *keep;
    int n_keep;
} ffwd_late_tokens_t;

/* Load the tokenizer for the model in model_dir and resolve its special-token
 * ids. is_late additionally resolves the late-interaction [Q]/[D]/[MASK] ids.
 * NULL on failure with a reason in err. */
FFWD_API ffwd_tok_t *
ffwd_tok_open(const char *model_dir, int is_late, char *err, size_t errlen);
FFWD_API void ffwd_tok_free(ffwd_tok_t *t);

/* Tokenize text for this model: applies the model's special-token layout and
 * optionally prepends a query instruction. Returns a malloc'd id array (caller
 * frees) and sets *n_out. NULL on error. */
FFWD_API int *
ffwd_tokenize(ffwd_tok_t *t, const char *text, const char *query_instruct, int *n_out);

/* Late-interaction tokenization: [Q]/[D] prefix, [MASK]-pad queries to a fixed
 * length, and mark the document positions to score. Returns 0 on success. */
FFWD_API int
ffwd_tokenize_late(ffwd_tok_t *t, const char *text, int is_query, ffwd_late_tokens_t *out);
FFWD_API void ffwd_late_tokens_free(ffwd_late_tokens_t *out);

/* The contextual chunk-separator id (the tokenizer's <|endoftext|>), for
 * frontends that split a document into chunks. */
FFWD_API int ffwd_tok_context_separator_id(const ffwd_tok_t *t);

/* Whether ffwd_tokenize appends the model's terminal token (<|endoftext|>) for
 * last-token pooling. Off until set; the caller sets it from the loaded model's
 * config (ffwd_config(...)->append_terminal_token), since it is model-specific
 * and only known after the backend is open. */
FFWD_API void ffwd_tok_set_append_terminal(ffwd_tok_t *t, int on);

/* Verbose level: 0=quiet, 1=info, 2=debug */
FFWD_API extern int ffwd_verbose;

#endif /* FFWD_H */
