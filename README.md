# embed.c

`embed.c` implements inference for:

- [pplx-embed](https://huggingface.co/collections/perplexity-ai/pplx-embed)
  dense, contextual, and late-interaction models
- [Qwen3-Embedding](https://huggingface.co/collections/Qwen/qwen3-embedding)
  causal, last-token pooled models

Supported platforms are CUDA/cuBLAS on Linux/NVIDIA, Apple Silicon MLX, and
CPU/BLAS.

## Building

**Prerequisites**:

- On Linux, install `libcjson-dev`, `libcjson1`, and `libopenblas-dev`.
- On macOS, install `cjson`, `mlx` and `mlx-c` via Homebrew.

**Makefile targets**:

- Run `make gpu` to auto-select the GPU backend (MLX on macOS, CUDA on Linux).
- Run `make cpu` to build the CPU backend only (BLAS: Accelerate on macOS,
  OpenBLAS on Linux).

## Usage

### embed

Pass one text to get its embedding, two or more to get a cosine similarity
matrix:

```bash
./embed -d ./model "What is the capital of France?"
./embed -d ./model "What is the capital of France?" \
  "Paris is the capital of France." \
  "Berlin is the capital of Germany."
```

Qwen3 retrieval queries benefit from an explicit task instruction. Documents are
embedded directly:

```bash
./embed -d ./Qwen3-Embedding-0.6B \
  $'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is the capital of China?'
```

The Qwen3 tokenizer terminal token is appended automatically by the CLI and
server. Qwen3 embeddings are L2-normalized; pplx embeddings remain unnormalized.

Pipe in lines and use `--stream` to get one JSON embedding per line:

```bash
cat texts.txt | ./embed -d ./model --stream -b 8
# {"embedding":[...],"dim":2048,"tokens":7,"ms":12.3,"workspace_bytes":1048576}
```

Without `--stream`, reading from stdin accumulates all lines then prints the
similarity matrix.

Use `--mlx` or `--cuda` to select a GPU backend (build-dependent). Default is
CPU. `-t` sets thread count, `-b` sets batch size, `-v` enables verbose output.

### embed-server

Serves the HTTP API. Takes one or more `--model ID=DIR` pairs:

```bash
./embed-server \
  --model pplx-embed-v1-0.6b=./model \
  --model Qwen3-Embedding-0.6B=./qwen3-model \
  --port 8000
```

Perplexity API compatible endpoints:

- `POST /v1/embeddings` — pooled embeddings
- `POST /v1/contextualizedembeddings` — contextual document embeddings

Late-interaction reranking endpoint:

- `POST /v1/rerank` — late-interaction MaxSim reranking

For `Qwen3-Embedding-0.6B`, `/v1/embeddings` accepts `dimensions` from 32
to 1024. Truncated Matryoshka embeddings are re-normalized before encoding.

Example:

```bash
./embed-server --model pplx-embed-v1-late-0.6b=./model
# can be tested with:
curl -s http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"pplx-embed-v1-late-0.6b",
    "query":"scientific curiosity",
    "documents":["Scientists explore from curiosity.","SQLite stores data."]
  }'
```
