# ffwd

Super fast inference server for transformer-based text embedding models.

It runs on NVIDIA and Apple silicon GPUs, as well as CPU-only hosts with BLAS
acceleration. It's mostly tested on Blackwell, Ada, and Apple M4 GPUs, but
should work with any CUDA or Metal capable device.

Supported models include Qwen3 embedding models such as `Qwen3-Embedding`, the
`pplx-embed-v1` family, Qwen2 models such as `GTE-Qwen2`, and BERT/RoBERTa-style
encoders including `BERT`, `BGE`, `MiniLM`, `XLM-R`, `multilingual-E5`, and
`Snowflake Arctic`.

## Installation

Download the latest build for your platform from the
[releases](https://github.com/tetsuo/ffwd/releases) page.

ffwd supports models in
[safetensors](https://huggingface.co/docs/safetensors/index) format. You can
download compatible models from Hugging Face using the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

## Building

To build ffwd from source, first install the dependencies for your platform:

- **Linux:** Install `libopenblas-dev` (required for both CPU and GPU builds).
- **macOS:** Install `mlx` and `mlx-c` via Homebrew if you're targeting
  Metal/GPU. Otherwise, no extra dependencies needed.

Once the dependencies are in place, build the project by running:

```sh
make  # or make gpu for GPU build
```

## Getting started

You can start the server with one or more models loaded, each with a label:

```bash
ffwd-server \
  --model pplx=/path/to/pplx-model:api=perplexity:min_dim=128 \
  --contextual-model pplx-context=/path/to/pplx-context-model:api=perplexity \
  --late-model pplx-late=/path/to/pplx-late-model \
  --model qwen=/path/to/qwen3-model \
  --port 8000
```

Retrieve embeddings with `curl`:

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

- `:api=openai|perplexity` - the output-compatibility mode (see below).
- `:min_dim=N` - the Matryoshka truncation floor; requests may ask for any
  `dimensions` from `N` up to the model's full size.
- `:query=STR` - a query-prompt override for models that prepend one.

For example: `--model qwen=/path/to/qwen3-model:api=openai:min_dim=32`.

Loading several labels keeps several complete model runtimes resident at once.
Large models need substantial memory; on a memory-limited host, host one large
model per process and route requests across processes rather than loading
several into one server.

## Usage

Run `ffwd-server --help` for the exact, current list of options. Some important
ones are:

| Flag                          | Default     | Meaning                                                                                       |
| ----------------------------- | ----------- | --------------------------------------------------------------------------------------------- |
| `--host HOST`                 | `127.0.0.1` | Bind address.                                                                                 |
| `--port PORT`                 | -           | Listen port.                                                                                  |
| `-t, --threads N`             | backend     | Worker thread count for CPU inference.                                                        |
| `-b, --batch-size N`          | `32`        | Max texts or contextual documents per inference micro-batch.                                  |
| `--max-batch-tokens N`        | `16384`     | Token budget per batch (packed for CPU, padded dense rows for MLX). Also caps a single input. |
| `--max-concurrent-requests N` | `512`       | In-flight request cap; excess requests are answered `429` immediately.                        |
| `--auto-truncate`             | off         | Truncate inputs that exceed the per-input token cap instead of rejecting them.                |
| `--batch-wait-us N`           | backend     | Micro-batch deadline from the first request's arrival (CPU/MLX `0`, CUDA `1000`).             |
| `--memory-utilization F`      | `0.90`      | Fraction of physical memory the MLX load preflight may plan for.                              |
| `--gpu-quant-bits N`          | off         | MLX: quantize transformer linear weights at load. Opt-in, approximate.                        |
| `--gpu-quant-group-size N`    | `64`        | MLX quantization group size.                                                                  |
| `--gpu-gemm-mode MODE`        | `f32`       | CUDA GEMM compute: `f32` (exact), or `tf32`/`bf16`/`16f` (reduced).                           |
| `--gpu-weight-dtype DTYPE`    | `f32`       | CUDA weight storage: `f32`, or `bf16` (half the weight memory).                               |
| `--api-key TOKEN`             | off         | Require `Authorization: Bearer TOKEN`. `FFWD_API_KEY` is also honored.                        |
| `-v, --verbose`               | off         | Verbose logging.                                                                              |

**Batching:**

- `-b 32` is a balanced default for concurrent short requests.
- Long-document throughput is compute-bound and largely insensitive to `-b`; for
  large models on memory-limited GPUs, prefer a smaller `-b` (for example `8`)
  to bound activation memory and queue latency.

**GPU precision flags:**

- `--gpu-gemm-mode` sets the GEMM compute precision - how the matmuls are
  computed. `f32` (the default) is exact. `bf16`/`16f` run the tensor-core
  reduced-precision matmul (16-bit operands, F32 accumulation). This flag also
  gates the reduced-precision attention/FFN fast paths; with the default `f32`
  they stay off.
- `--gpu-weight-dtype` sets the weight storage precision - how weights are held
  in GPU memory. `bf16`/`f16` roughly halve the weight footprint.

Because these two settings are independent, changing only `--gpu-weight-dtype`
bf16 means your calculations (GEMMs) will still run in full, precise `F32`.
While this saves memory by compressing the weights, it won't actually speed up
your processing.

To get the full speed benefits of reduced precision during serving, you need to
set the compute mode and ensure your storage dtype matches it:

```
# bf16 inference (recommended production GPU default):
ffwd-server --cuda --model my=/path/to/model --gpu-gemm-mode bf16 --gpu-weight-dtype bf16
# fp16 inference:
ffwd-server --cuda --model my=/path/to/model --gpu-gemm-mode 16f  --gpu-weight-dtype f16
```

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

- `base64_int8` - `round(tanh(x) * 127)` after Matryoshka truncation, base64 of
  the signed int8 bytes.
- `base64_binary` - sign bits packed LSB-first; requires `dimensions` divisible
  by 8.
- `float` (perplexity) - the int8 values as floats (`int8 / 128`), lossy.
- `base64` (openai) - the raw little-endian float32 bytes, base64-encoded,
  lossless.

A request whose `encoding_format` is not in the model's accepted set returns
`422`. pplx-embed vectors are unnormalized - always compare with cosine
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

- A request allows up to 512 inputs (1000 documents for `/v1/rerank`), 120000
  tokens total, and a 64 MiB body.
- Each input (or packed contextual document, or rerank text) is capped at the
  smallest of: the model's positional capacity (for example 512 for BERT
  encoders), `--max-batch-tokens`, and 32768. Inputs past the cap return `422`
  (`value_error.context_length`, the message states the cap), or are truncated
  to it when the server runs with `--auto-truncate`. Truncation preserves the
  model's trailing special token. The server logs each model's effective cap at
  startup.
- Invalid input returns `422` with a `detail` list naming the offending field.
  Unknown labels, and labels loaded for another endpoint, also return `422`.
- Beyond `--max-concurrent-requests` in-flight requests, new requests return
  `429` immediately instead of queueing; retry with backoff.

## Acknowledgements

- This project started as a fork of
  [antirez/qwen-asr](https://github.com/antirez/qwen-asr). Most of it has been
  rewritten since then, but the early kernels and some of the structure come
  from that codebase.
- The event loop library in `deps/ae` comes from
  [Redis](https://github.com/redis/redis). Thanks to Salvatore Sanfilippo
  (antirez) and the Redis contributors.
- Many thanks to the authors of all the open-weight models that ffwd supports.
