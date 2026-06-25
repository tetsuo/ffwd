# ffwd

ffwd is a high-performance text embedding inference library and toolkit for
transformer-based models.

It runs natively on Apple Silicon and NVIDIA GPUs, with CPU support for both x86
and ARM architectures.

## Supported models

Tested architectures include:

- Qwen3: Qwen3-Embedding and pplx-embed families
- Qwen2: GTE-Qwen2
- BERT/RoBERTa: BERT, BGE, MiniLM, and XLM-R multilingual encoders such as
  multilingual-E5 and Snowflake Arctic

## Components

### libffwd

[libffwd](./libffwd) is the core embedding inference library, providing
optimized kernels, tokenization, and model loading.

### ffwd-server

[ffwd-server](./tools/server) is an HTTP server that provides OpenAI- and
Perplexity-compatible embedding APIs, including support for late-interaction and
contextualized embeddings.

### ffwd-cli

[ffwd-cli](./tools/cli) is a command-line tool for generating embeddings and
computing cosine similarity.

## Building

Before building, install the dependencies for your platform:

- On Linux, install `libopenblas-dev` and `libyyjson-dev`.
- On macOS, install `mlx`, `mlx-c` and `yyjson` through Homebrew.

Once the dependencies are in place, run `make` to build the libraries and tools.

## Acknowledgements

- This project started as a fork of
  [antirez/qwen-asr](https://github.com/antirez/qwen-asr). Most of it has been
  rewritten since then, but the early kernels and some of the structure come
  from that codebase.
- The event loop library in `deps/ae` comes from
  [Redis](https://github.com/redis/redis). Thanks to Salvatore Sanfilippo
  (antirez) and the Redis contributors.
- Many thanks to the authors of all the open-weight models that ffwd supports.
