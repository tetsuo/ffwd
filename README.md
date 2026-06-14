# embed.c

`embed.c` implements inference for Perplexity
[pplx-embed](https://huggingface.co/collections/perplexity-ai/pplx-embed)
dense/contextual text embedding models plus the recently added MaxSim
late-interaction model.

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
