# embed.c

`embed.c` implements inference for several text embedding model families —
[pplx-embed](https://huggingface.co/collections/perplexity-ai/pplx-embed),
[Qwen3-Embedding](https://huggingface.co/collections/Qwen/qwen3-embedding),
GTE-Qwen2, and BERT/BGE encoders. It serves them through an
OpenAI/Perplexity-compatible HTTP API and runs on CPU, Apple MLX, and NVIDIA
CUDA. See [Supported models](#supported-models).

## Supported models

The server registers these model IDs out of the box; pass any of them as
`--model ID=DIR`, loading a Hugging Face safetensors checkpoint from `DIR`.
Every model runs on all three backends, and `dimensions` truncates
Matryoshka-trained vectors (pplx-embed down to 128, Qwen3-Embedding down to 32).

**Pooled embeddings** — one vector per text, via `POST /v1/embeddings`:

| Model ID                | Family          | Dim  | Pooling    | Output                 |
| ----------------------- | --------------- | ---- | ---------- | ---------------------- |
| pplx-embed-v1-0.6b      | pplx-embed      | 1024 | mean       | int8, unnormalized     |
| pplx-embed-v1-4b        | pplx-embed      | 2560 | mean       | int8, unnormalized     |
| Qwen3-Embedding-0.6B    | Qwen3-Embedding | 1024 | last token | float32, L2-normalized |
| Qwen3-Embedding-4B      | Qwen3-Embedding | 2560 | last token | float32, L2-normalized |
| Qwen3-Embedding-8B      | Qwen3-Embedding | 4096 | last token | float32, L2-normalized |
| gte-Qwen2-1.5B-instruct | GTE-Qwen2       | 1536 | last token | float32, L2-normalized |
| all-MiniLM-L6-v2        | BERT (MiniLM)   | 384  | mean       | float32, L2-normalized |
| bge-small-en-v1.5       | BERT (BGE)      | 384  | CLS        | float32, L2-normalized |
| bge-base-en-v1.5        | BERT (BGE)      | 768  | CLS        | float32, L2-normalized |
| bge-large-en-v1.5       | BERT (BGE)      | 1024 | CLS        | float32, L2-normalized |

**Contextual embeddings** — one vector per document chunk, each embedded with
its document as context, via `POST /v1/contextualizedembeddings`:

| Model ID                   | Dim  | Output             |
| -------------------------- | ---- | ------------------ |
| pplx-embed-context-v1-0.6b | 1024 | int8, unnormalized |
| pplx-embed-context-v1-4b   | 2560 | int8, unnormalized |

**Late interaction** — token-level vectors scored with MaxSim for reranking, via
`POST /v1/rerank`:

| Model ID                | Token dim | Output              |
| ----------------------- | --------- | ------------------- |
| pplx-embed-v1-late-0.6b | 128       | MaxSim rerank score |

The engine handles three transformer blocks — the Qwen3 block (pplx-embed and
Qwen3-Embedding), the Qwen2 block (GTE-Qwen2), and the BERT block (MiniLM, BGE)
— and selects the tokenizer from the model files (byte-level BPE for the Qwen
families, WordPiece for BERT). pplx-embed vectors are unnormalized int8 (rank
them by cosine similarity); the other families emit L2-normalized float32.

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

Serves the Perplexity/OpenAI-compatible HTTP API. Takes one or more
`--model ID=DIR` pairs:

```bash
./embed-server \
  --model pplx-embed-v1-0.6b=./model \
  --model pplx-embed-context-v1-0.6b=./context-model \
  --model Qwen3-Embedding-0.6B=./qwen3-model \
  --port 8000
```

### API

Every request body takes a `model` parameter and returns results in input order.

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
    "model": "Qwen3-Embedding-0.6B",
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
  "model": "Qwen3-Embedding-0.6B",
  "usage": {"prompt_tokens": 16, "total_tokens": 16}
}
```

| Field             | Type            | Description                                             |
| ----------------- | --------------- | ------------------------------------------------------- |
| `input`           | string or array | One text, or up to 512 texts. Required.                 |
| `dimensions`      | integer         | Truncate the embedding (Matryoshka), 32 up to its size. |
| `encoding_format` | string          | Output encoding (see below).                            |

**Encoding format:**

`encoding_format` follows the model family:

- The OpenAI-API models (Qwen3-Embedding, GTE-Qwen2, MiniLM, BGE) default to
  `float` (the true float32 vector) and also accept `base64` (base64 of
  float32).
- pplx-embed (Perplexity-compatible) defaults to `base64_int8` and also accepts
  `base64_binary` and `float` (the decoded int8 view).

#### POST /v1/contextualizedembeddings

`input` is a list of documents, each a list of chunk strings. Every chunk is
embedded with its document as context, and the results are grouped the same way:
one list per document, one embedding per chunk.

```bash
curl http://127.0.0.1:8000/v1/contextualizedembeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pplx-embed-context-v1-0.6b",
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
  "model": "pplx-embed-context-v1-0.6b",
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
    "model": "pplx-embed-v1-late-0.6b",
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
  "model": "pplx-embed-v1-late-0.6b",
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
- Invalid input returns `422` with a `detail` list naming the offending field; a
  registered model that is not loaded returns `503`.

#### Deployment

The server runs one inference worker and serves one in-flight request per
connection (HTTP/1.1 keep-alive).

For concurrency and availability, run several processes behind a load balancer
and route by model ID; keep one large model per process. Terminate TLS and
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
