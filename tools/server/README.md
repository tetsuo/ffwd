# ffwd-server

ffwd HTTP server.

## Quickstart

```bash
make # use make gpu for GPU

./ffwd-server \
  --model pplx=./pplx-model:api=perplexity:min_dim=128 \
  --contextual-model pplx-context=./pplx-context-model:api=perplexity \
  --late-model pplx-late=./pplx-late-model \
  --model qwen=./qwen3-model \
  --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","input":["What is the capital of France?",
                               "Paris is the capital of France."]}'
```

## Loading models

Each load flag binds a `LABEL` to a model directory and enables one endpoint.
The flags are repeatable, so several models can share a server.

| Flag                           | Enables                             |
| ------------------------------ | ----------------------------------- |
| `--model LABEL=DIR`            | `POST /v1/embeddings`               |
| `--contextual-model LABEL=DIR` | `POST /v1/contextualizedembeddings` |
| `--late-model LABEL=DIR`       | `POST /v1/rerank`                   |

Each flag also accepts inline, colon-separated per-model options appended to the
directory:

- `:api=openai|perplexity` — the output-compatibility mode (see below).
- `:min_dim=N` — the Matryoshka truncation floor; requests may ask for any
  `dimensions` from `N` up to the model's full size.
- `:query=STR` — a query-prompt override for models that prepend one.

For example: `--model qwen=./qwen3-model:api=openai:min_dim=32`.

Loading several labels keeps several complete model runtimes resident at once.
Large models need substantial memory; on a memory-limited host, host one large
model per process and route requests across processes rather than loading
several into one server.

## Options

Run `./ffwd-server --help` for the exact, current list. Some important flags:

| Flag                       | Default     | Meaning                                                                           |
| -------------------------- | ----------- | --------------------------------------------------------------------------------- |
| `--host HOST`              | `127.0.0.1` | Bind address.                                                                     |
| `--port PORT`              | —           | Listen port.                                                                      |
| `-t, --threads N`          | backend     | Worker thread count for CPU inference.                                            |
| `-b, --batch-size N`       | `32`        | Max texts or contextual documents per inference micro-batch.                      |
| `--max-batch-tokens N`     | `16384`     | Token budget per batch (packed for CPU, padded dense rows for MLX).               |
| `--batch-wait-us N`        | backend     | Micro-batch deadline from the first request's arrival (CPU/MLX `0`, CUDA `1000`). |
| `--memory-utilization F`   | `0.90`      | Fraction of physical memory the MLX load preflight may plan for.                  |
| `--gpu-quant-bits N`       | off         | MLX: quantize transformer linear weights at load. Opt-in, approximate.            |
| `--gpu-quant-group-size N` | `64`        | MLX quantization group size.                                                      |
| `--gpu-gemm-mode MODE`     | `f32`       | CUDA GEMM compute: `f32` (exact), or `tf32`/`bf16`/`16f` (reduced).               |
| `--gpu-weight-dtype DTYPE` | `f32`       | CUDA weight storage: `f32`, or `bf16` (half the weight memory).                   |
| `--api-key TOKEN`          | off         | Require `Authorization: Bearer TOKEN`. `FFWD_API_KEY` is also honored.            |
| `-v, --verbose`            | off         | Verbose logging.                                                                  |

`-b 32` is a balanced default for concurrent short requests. Long-document
throughput is compute-bound and largely insensitive to `-b`; for large models on
memory-limited GPUs, prefer a smaller `-b` (for example `8`) to bound activation
memory and queue latency.

The reduced-precision GPU modes (`--gpu-gemm-mode`, `--gpu-weight-dtype bf16`,
`--gpu-quant-bits`) are precision trades judged by output quality (cosine /
ranking), never by latency against the exact path.

## API compatibility modes

Each model serves in one of two output modes, chosen per model with
`:api=openai` (the default) or `:api=perplexity`. The mode fixes the whole
output contract for that model's endpoint, not just a default encoding:

| Mode         | Default       | Accepted encodings                      | `float` returns              |
| ------------ | ------------- | --------------------------------------- | ---------------------------- |
| `openai`     | `float`       | `float`, `base64`                       | the true float32 vector      |
| `perplexity` | `base64_int8` | `base64_int8`, `base64_binary`, `float` | the int8 vector, dequantized |

`perplexity` exists because some models (such as pplx-embed) are int8-native:
the embedding _is_ the tanh-quantized int8 vector, so even a `float` response is
that vector dequantized, not a separate full-precision one.

- `base64_int8` — `round(tanh(x) * 127)` after Matryoshka truncation, base64 of
  the signed int8 bytes.
- `base64_binary` — sign bits packed LSB-first; requires `dimensions` divisible
  by 8.
- `float` (perplexity) — the int8 values as floats (`int8 / 128`), lossy.
- `base64` (openai) — the raw little-endian float32 bytes, base64-encoded,
  lossless.

A request whose `encoding_format` is not in the model's accepted set returns
`422`. pplx-embed vectors are unnormalized — always compare with cosine
similarity; the core API exposes `ffwd_l2_normalize()` for clients that need an
inner-product index.

## API

Every request body takes a `model` label and returns results in input order.

| Endpoint                            | Returns                          |
| ----------------------------------- | -------------------------------- |
| `POST /v1/embeddings`               | one pooled embedding per input   |
| `POST /v1/contextualizedembeddings` | one embedding per document chunk |
| `POST /v1/rerank`                   | documents ranked by MaxSim score |

### POST /v1/embeddings

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","input":["What is the capital of France?",
                               "Paris is the capital of France."]}'
```

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.013, -0.021] },
    { "object": "embedding", "index": 1, "embedding": [0.046, 0.009] }
  ],
  "model": "qwen",
  "usage": { "prompt_tokens": 16, "total_tokens": 16 }
}
```

| Field             | Type            | Description                                                           |
| ----------------- | --------------- | --------------------------------------------------------------------- |
| `input`           | string or array | One text, or up to 512 texts. Required.                               |
| `dimensions`      | integer         | Truncate the embedding, from the configured `min_dim` up to its size. |
| `encoding_format` | string          | Output encoding; follows the model's API compatibility mode.          |

### POST /v1/contextualizedembeddings

`input` is a list of documents, each a list of chunk strings. Every chunk is
embedded with its document as context, and results are grouped the same way: one
list per document, one embedding per chunk, in order.

```bash
curl http://127.0.0.1:8000/v1/contextualizedembeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"pplx-context",
       "input":[["Intro paragraph.","Methods paragraph."],
                ["A single-chunk note."]]}'
```

```json
{
  "object": "list",
  "data": [
    {
      "object": "list",
      "index": 0,
      "data": [
        { "object": "embedding", "index": 0, "embedding": "<base64>" },
        { "object": "embedding", "index": 1, "embedding": "<base64>" }
      ]
    },
    {
      "object": "list",
      "index": 1,
      "data": [{ "object": "embedding", "index": 0, "embedding": "<base64>" }]
    }
  ],
  "model": "pplx-context",
  "usage": { "prompt_tokens": 12, "total_tokens": 12 }
}
```

Documents are concatenated with the model tokenizer's separator for context;
separator tokens are excluded from chunk pooling.

### POST /v1/rerank

Scores each document against the query with token-level MaxSim and returns
documents ranked by descending raw score.

```bash
curl http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"pplx-late","query":"scientific curiosity",
       "documents":["Scientists explore from curiosity.","SQLite stores data."]}'
```

```json
{
  "object": "list",
  "model": "pplx-late",
  "results": [
    { "index": 0, "relevance_score": 2.14 },
    { "index": 1, "relevance_score": 0.87 }
  ],
  "usage": { "query_tokens": 3, "document_tokens": 8, "total_tokens": 11 }
}
```

- `documents` takes up to 1000 strings.
- `top_n` (alias `top_k`) caps the number of results (default: all).
- `return_documents: true` echoes each document text.
- Scores are raw MaxSim values, not probabilities.

### Limits and errors

- A request allows up to 512 inputs (1000 documents for `/v1/rerank`), 32768
  tokens per item, 120000 tokens total, and a 64 MiB body.
- Invalid input returns `422` with a `detail` list naming the offending field.
  Unknown labels, and labels loaded for another endpoint, also return `422`.
