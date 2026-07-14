# ffwd

ffwd is a lightweight inference engine for transformer-based text embedding
models.

It runs on NVIDIA and Apple silicon GPUs, as well as x86 and arm64 CPUs with
BLAS acceleration.

ffwd supports Qwen3 embedding models such as `Qwen3-Embedding`, the
`pplx-embed-v1` family, Qwen2 models such as `GTE-Qwen2`, and BERT/RoBERTa-style
encoders including `BERT`, `BGE`, `MiniLM`, `XLM-R`, `multilingual-E5`, and
`Snowflake Arctic`.

## Components

### libffwd

[libffwd](./libffwd) is the core embedding inference library. It provides
CPU/GPU kernels, tokenizers, and model loading.

### ffwd-server

[ffwd-server](./tools/server) is an HTTP server that provides OpenAI and
Perplexity compatible embedding APIs, including support for late-interaction and
contextualized embeddings.

### ffwd-cli

[ffwd-cli](./tools/cli) is a command-line tool for generating embeddings and
computing cosine similarity.

## Installation

You can download the latest ffwd build for your platform from the
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

## Acknowledgements

- This project started as a fork of
  [antirez/qwen-asr](https://github.com/antirez/qwen-asr). Most of it has been
  rewritten since then, but the early kernels and some of the structure come
  from that codebase.
- The event loop library in `deps/ae` comes from
  [Redis](https://github.com/redis/redis). Thanks to Salvatore Sanfilippo
  (antirez) and the Redis contributors.
- Many thanks to the authors of all the open-weight models that ffwd supports.
