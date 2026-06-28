# ffwd

ffwd is an experimental, lightweight inference engine for text embedding models,
written from scratch in C.

It runs on NVIDIA GPUs (CUDA), Apple silicon (via MLX), and x86/arm64 CPUs
(OpenBLAS/Accelerate).

Supported models include:

- **Qwen3:** the `Qwen3-Embedding` family, with full support for `pplx-embed-v1`
- **Qwen2:** `GTE-Qwen2`
- **BERT/RoBERTa:** `BERT`, `BGE`, `MiniLM`, and `XLM-R` multilingual encoders,
  including `multilingual-E5` and `Snowflake Arctic`

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

## Building

Before building, install the dependencies for your platform:

- **Linux:** Install `libopenblas-dev` (required for both CPU and GPU builds).
- **macOS:** Install `mlx` and `mlx-c` via Homebrew if you're targeting
  Metal/GPU. Otherwise, no extra dependencies needed.

Once the dependencies are in place, build the project by running:

```sh
make  # or make gpu for GPU build
```

### Performance

ffwd implements custom FlashAttention-style kernels from scratch. On NVIDIA
Blackwell and Ada GPUs, it approaches the theoretical peak performance of `bf16`
math without relying on libraries such as CUTLASS or cuDNN. However, at longer
context lengths—for example, above 16k tokens for `Qwen3-Embedding`—vendor
kernels begin to take the lead.

When comparing ffwd with
[TEI](https://github.com/huggingface/text-embeddings-inference), the main factor
is attention speed. ffwd is very fast at short and medium context lengths
because the work around the attention kernel is kept extremely low-overhead. But
once both engines are pushed toward saturation—especially with a single long
context or very large batches—the bottleneck becomes attention-kernel
throughput. Unsurprisingly, this is where TEI, which leverages CuTe-based
[Flash Attention](https://github.com/HazyResearch/flash-attention), began to
outperform ffwd in my benchmarks.

## Acknowledgements

- This project started as a fork of
  [antirez/qwen-asr](https://github.com/antirez/qwen-asr). Most of it has been
  rewritten since then, but the early kernels and some of the structure come
  from that codebase.
- The event loop library in `deps/ae` comes from
  [Redis](https://github.com/redis/redis). Thanks to Salvatore Sanfilippo
  (antirez) and the Redis contributors.
- Many thanks to the authors of all the open-weight models that ffwd supports.
