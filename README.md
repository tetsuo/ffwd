# embed.c

`embed.c` implements high-performance embedding inference for Hugging Face
safetensors checkpoints, with CPU, Apple MLX, and NVIDIA CUDA backends.

Tested model families include pplx-embed, Qwen3-Embedding, GTE-Qwen2, BERT/BGE
encoders, and XLM-R encoders such as multilingual-E5 and Snowflake Arctic.
Weights load directly from F32, BF16, or F16 safetensors.

The engine handles three transformer blocks — the Qwen3 block (pplx-embed and
Qwen3-Embedding), the Qwen2 block (GTE-Qwen2), and the BERT/RoBERTa block
(MiniLM, BGE, and the XLM-R multilingual encoders E5 and Arctic) — and selects
the tokenizer from the model files: byte-level BPE for the Qwen families,
WordPiece for BERT, and SentencePiece (Unigram) for XLM-R. Pooling and
L2-normalization are read from the checkpoint's Sentence Transformers config
(`1_Pooling/config.json` and `modules.json`), so a model's pooling mode does not
have to be hardcoded. pplx-embed vectors are unnormalized int8 (rank them by
cosine similarity); the other families emit L2-normalized float32.

## Building

**Prerequisites**:

- On Linux, install `libopenblas-dev`.
- On macOS, install `mlx` and `mlx-c` via Homebrew.

You also need `libcjson-dev`/`cjson` to build the server; the CLI and the shared
library can be built without it.

**Makefile targets**:

- Run `make gpu` to auto-select the GPU backend (MLX on macOS, CUDA on Linux).
- Run `make cpu` to build the CPU backend only (BLAS: Accelerate on macOS,
  OpenBLAS on Linux).

## embed

Pass one text to get its embedding, two or more to get a cosine similarity
matrix:

```bash
# one text: the embedding as a line of space-separated floats
./embed -d ./pplx-embed-v1-0.6b "What is the capital of France?"

# two or more: a cosine similarity matrix, one row per line, no labels
./embed -d ./pplx-embed-v1-0.6b "What is the capital of France?" \
  "Paris is the capital of France." \
  "Berlin is the capital of Germany."
# 1.000000 0.719822 0.422054
# 0.719822 1.000000 0.601722
# 0.422054 0.601722 1.000000
```

With Qwen3, prefix the retrieval query with a task instruction and embed
documents as-is. Pass them together to rank the documents against the query (the
query is the first row of the similarity matrix):

```bash
# arg 1: the query, with Qwen3's instruction prefix
# args 2+: documents
./embed -d ./Qwen3-Embedding-0.6B \
  $'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?' \
  "Beijing is the capital of China." \
  "The Great Wall is a famous landmark."
```

> pplx-embed vectors are unnormalized, so rank them with cosine similarity.
> Qwen3-Embedding vectors are L2-normalized, so cosine similarity and dot
> product give the same ranking.

Pipe in lines and use `--stream` to get one JSON embedding per line:

```bash
cat texts.txt | ./embed -d ./model --stream -b 8
# {"embedding":[...],"dim":2560,"tokens":7,"ms":12.3,"workspace_bytes":1048576}
```

Without `--stream`, reading from stdin accumulates all lines then prints the
similarity matrix.

Add `--json` to emit JSON. One text prints an array of floats; two or more print
an array of rows (the matrix):

```bash
./embed -d ./model --json "cat" "dog"
# [[1.000000,0.380861],[0.380861,1.000000]]
```

Use `--mlx` or `--cuda` to select a GPU backend (build-dependent). Default is
CPU. `-t` sets thread count, `-b` sets batch size, `-v` enables verbose output.
`-V`/`--version` prints the version; `--build-info` also reports the commit,
build date, target, and compiler.

## embed-server

Serves the Perplexity/OpenAI-compatible HTTP API. Each loaded model gets an
operator-chosen label, and requests use that label in the `model` field. The
label is only a routing name; architecture, pooling, normalization, tokenizer,
and dimensionality are read from the model directory.

Use a separate load flag for each endpoint contract:

- `--model LABEL=DIR` serves `/v1/embeddings`.
- `--contextual-model LABEL=DIR` serves `/v1/contextualizedembeddings`.
- `--late-model LABEL=DIR` serves `/v1/rerank`.

Scalar serving options that are not in standard model files can be appended to
the load argument: `:api=openai|perplexity`, `:min_dim=N`, and a final
`:query=TEXT` prompt override. `api` defaults to `openai`; use
`api=perplexity` for pplx-compatible base64-int8 output. `min_dim` defaults to
the model's full embedding size; set it for Matryoshka truncation floors.

```bash
./embed-server \
  --model pplx=./pplx-model:api=perplexity:min_dim=128 \
  --contextual-model pplx-context=./pplx-context-model:api=perplexity:min_dim=128 \
  --late-model pplx-late=./pplx-late-model \
  --model qwen=./qwen3-model:min_dim=32 \
  --port 8000
```

### API

Every request body takes a `model` label and returns results in input order.

| Endpoint                            | Returns                          |
| ----------------------------------- | -------------------------------- |
| `POST /v1/embeddings`               | one pooled embedding per input   |
| `POST /v1/contextualizedembeddings` | one embedding per document chunk |
| `POST /v1/rerank`                   | documents ranked by MaxSim score |

#### POST /v1/embeddings

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "input": [
      "What is the capital of France?",
      "Paris is the capital of France."
    ]
  }'
```

Returns a list of embeddings, one per input text:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.013, -0.021, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.046, 0.009, ...]}
  ],
  "model": "qwen",
  "usage": {"prompt_tokens": 16, "total_tokens": 16}
}
```

| Field             | Type            | Description                                             |
| ----------------- | --------------- | ------------------------------------------------------- |
| `input`           | string or array | One text, or up to 512 texts. Required.                 |
| `dimensions`      | integer         | Truncate the embedding, from the configured `min_dim` up to its size. |
| `encoding_format` | string          | Output encoding (see below).                            |

**Encoding format:**

`encoding_format` follows the serving API selected at load time:

- `api=openai` defaults to `float` and also accepts `base64` (base64 of
  little-endian float32).
- `api=perplexity` defaults to `base64_int8` and also accepts `base64_binary`
  and `float` (the decoded int8 view).

#### POST /v1/contextualizedembeddings

`input` is a list of documents, each a list of chunk strings. Every chunk is
embedded with its document as context, and the results are grouped the same way:
one list per document, one embedding per chunk.

```bash
curl http://127.0.0.1:8000/v1/contextualizedembeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pplx-context",
    "input": [
      ["Intro paragraph.", "Methods paragraph."],
      ["A single-chunk note."]
    ]
  }'
```

Returns a list of chunk embeddings per document, in order:

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

#### POST /v1/rerank

Scores each document against the query with token-level MaxSim and returns
documents ranked by raw score.

```bash
curl http://127.0.0.1:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pplx-late",
    "query": "scientific curiosity",
    "documents": [
      "Scientists explore from curiosity.",
      "SQLite stores data."
    ]
  }'
```

Documents ranked by descending relevance score:

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

- `documents` takes up to 1000 strings. `top_n` caps the number of results
  (default: all); `return_documents: true` echoes each document text.
- Scores are raw MaxSim values, not probabilities.

#### Limits and errors

- A request allows up to 512 inputs (1000 documents for `/v1/rerank`), 32768
  tokens per item, 120000 tokens total, and a 64 MiB body.
- Invalid input returns `422` with a `detail` list naming the offending field.
  Unknown labels and labels loaded for another endpoint also return `422`.

#### Deployment

The server runs one inference worker and serves one in-flight request per
connection (HTTP/1.1 keep-alive).

For concurrency and availability, run several processes behind a load balancer
and route by model label; keep one large model per process. Terminate TLS and
authenticate requests at the proxy.

## Acknowledgements

- `embed.c` was inspired by and initially forked from
  [antirez/qwen-asr](https://github.com/antirez/qwen-asr). It has since been
  mostly rewritten, including a major CUDA upgrade, but the kernels and some
  structural ideas are derived from that codebase.
- The server event loop library in `deps/ae` is extracted from
  [Redis](https://github.com/redis/redis), credited to Salvatore Sanfilippo
  (antirez) and Redis contributors.
- The
  [pplx-embed](https://research.perplexity.ai/articles/pplx-embed-state-of-the-art-embedding-models-for-web-scale-retrieval)
  models were developed by the Perplexity AI research team.
- The [Qwen3-Embedding](https://qwen.ai/blog?id=qwen3-embedding) models are part
  of the Qwen model family from Alibaba.

## License

MIT
