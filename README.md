# ffwd

ffwd is a library and toolkit for text embedding inference.

It works with a variety of transformer-based models and runs natively on Apple
Silicon and NVIDIA GPUs, with CPU fallback on x86/ARM.

## Supported models

ffwd supports models in the Hugging Face
[safetensors](https://huggingface.co/docs/safetensors/index) format. Tested
architectures include:

- Qwen3: Qwen3-Embedding and pplx-embed families
- Qwen2: GTE-Qwen2
- BERT/RoBERTa: BERT, BGE, MiniLM, and XLM-R multilingual encoders such as
  multilingual-E5 and Snowflake Arctic

## Components

- [libffwd](./libffwd) is the core embedding inference library with kernels,
  tokenizer, and model loading.
- [ffwd-server](./tools/server) is a lightweight HTTP server with
  OpenAI/Perplexity-compatible embedding APIs, including late-interaction and
  contextualized embeddings.
- [ffwd-cli](./tools/cli) is a basic command-line tool that generates embeddings
  and computes cosine similarity.

## Building

Before building, install the dependencies for your platform:

- On Linux, install `libopenblas-dev`.
- On macOS, install `mlx` and `mlx-c` through Homebrew.

The server also needs `libcjson-dev`/`cjson`. You can skip it if you only want
`libffwd` and the command-line tool.

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
