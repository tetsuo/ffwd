#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.venv/uv-cache}"

RED=""
GREEN=""
YELLOW=""
NORMAL=""
MAXWIDTH=0

if [[ -t 2 ]] && command -v tput >/dev/null 2>&1; then
  RED="$(tput setaf 1 2>/dev/null || true)"
  GREEN="$(tput setaf 2 2>/dev/null || true)"
  YELLOW="$(tput setaf 3 2>/dev/null || true)"
  NORMAL="$(tput sgr0 2>/dev/null || true)"
  MAXWIDTH="$(tput cols 2>/dev/null || printf '0')"
  if (( MAXWIDTH > 2 )); then
    MAXWIDTH=$((MAXWIDTH - 2))
  else
    MAXWIDTH=0
  fi
fi

info() { printf '%sINFO: %s%s\n' "$GREEN" "$*" "$NORMAL" >&2; }
warn() { printf '%sWARN: %s%s\n' "$YELLOW" "$*" "$NORMAL" >&2; }
err() { printf '%sERROR: %s%s\n' "$RED" "$*" "$NORMAL" >&2; }

run() {
  local rendered
  printf -v rendered '%q ' "$@"
  rendered="${rendered% }"
  if (( MAXWIDTH > 0 && ${#rendered} > MAXWIDTH )); then
    rendered="${rendered:0:$((MAXWIDTH - 3))}..."
  fi
  info "\$ $rendered"
  (cd "$ROOT" && "$@")
}

run_uv() {
  run uv run "$@"
}

# Run a Python check that needs the pinned SentenceTransformers reference stack
# (torch, sentence-transformers); the pins live in requirements-reference.txt.
run_reference() {
  run uv run --with-requirements requirements-reference.txt "$@"
}

check_code_quality() {
  run ./devtools/check_code_quality.sh "$@"
}

run_checks() {
  run ./devtools/verify_builds.sh "$@"
}

check_kernel_golden() {
  # Build and run the math-kernel golden tests straight from the test Makefile.
  # test_kernels exercises the BLAS-built kernels; test_kernels_generic the
  # scalar/SIMD path with USE_BLAS off. Extra args (e.g. --blas, kept for
  # backward compatibility) are accepted but ignored; both variants always run.
  run make -C tests test_kernels test_kernels_generic
  run ./tests/test_kernels
  run ./tests/test_kernels_generic
}

check_batch_parity() {
  run_uv ./tests/check_batch_parity.py "$@"
}

check_contextual_batch_parity() {
  run_uv ./tests/check_contextual_batch_parity.py "$@"
}

check_late_interaction() {
  run_uv ./tests/check_late_interaction.py "$@"
}

check_cuda_fast_gemm_parity() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    warn "CUDA fast GEMM parity requires Linux; skipping on $(uname -s)"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "CUDA fast GEMM parity requires an NVIDIA GPU; nvidia-smi not found"
    return 0
  fi
  run_uv ./tests/check_cuda_fast_gemm_parity.py "$@"
}

check_embedding_parity() {
  run_uv ./devtools/compare_model_embeddings.py "$@"
}

check_mlx_model_limits() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    warn "MLX metadata limits require macOS; skipping on $(uname -s)"
    return 0
  fi
  run_uv ./tests/check_mlx_model_limits.py "$@"
}

check_mlx_quantized_parity() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    warn "MLX quantized parity requires macOS; skipping on $(uname -s)"
    return 0
  fi
  run_uv ./tests/check_mlx_quantized_parity.py "$@"
}

check_quant_precision_drift() {
  run_reference ./devtools/quant_precision_drift.py "$@"
}

check_reference_parity() {
  run_reference ./tests/check_reference_parity.py "$@"
}

check_bert_parity() {
  run_reference ./tests/check_bert_parity.py "$@"
}

check_e5_parity() {
  run_reference ./tests/check_e5_parity.py "$@"
}

check_qwen3_parity() {
  run uv run --with sentence-transformers --with torch --with numpy \
    ./tests/check_qwen3_parity.py "$@"
}

check_tokenizer_parity() {
  run_uv ./tests/check_tokenizer_parity.py "$@"
}

check_wordpiece_parity() {
  run_uv ./tests/check_wordpiece_parity.py "$@"
}

check_model_matrix() {
  run_uv ./tests/check_model_matrix.py "$@"
}

check_perplexity_sdk_compat() {
  run uv run --with perplexityai --with numpy \
    ./tests/check_perplexity_sdk_compat.py "$@"
}

check_openai_sdk_compat() {
  run uv run --with openai ./tests/check_openai_sdk_compat.py "$@"
}

check_server_api() {
  run_uv ./tests/check_server_api.py "$@"
}

check_cli() {
  run_uv ./tests/check_cli.py "$@"
}

check_workspace_api() {
  run_uv ./tests/check_workspace_api.py "$@"
}

bench_contextual_api() {
  run_uv ./bench/bench_contextual_api.py "$@"
}

bench_late_rerank() {
  run_uv ./bench/bench_late_rerank.py "$@"
}

bench_late_api() {
  run_uv ./bench/bench_late_api.py "$@"
}

bench_matrix() {
  run_uv ./bench/bench_matrix.py "$@"
}

bench_qwen_tokenizer() {
  run_uv ./bench/bench_qwen_tokenizer.py "$@"
}

bench_server_api() {
  run_uv ./bench/bench_server_api.py "$@"
}

usage() {
  cat <<'EOF'
Usage: ./all.bash COMMAND [ARGS...]

Aggregate:
  checks                        Build and run the comprehensive local check suite

Checks:
  check-code-quality            Run clang builds, analysis, and linting
  check-kernel-golden           Run math-kernel golden tests
  check-batch-parity            Compare singleton and true batched stdin execution
  check-context-batch-parity    Compare contextual batch and singleton execution
  check-cuda-fast-gemm-parity   Gate CUDA reduced-precision modes against exact F32
  check-embedding-parity        Compare embeddings from two model directories
  check-late-interaction        Check late-interaction vectors and MaxSim ranking
  check-mlx-model-limits        Check MLX model metadata rejection
  check-mlx-quantized-parity    Compare normal and quantized MLX embeddings
  check-precision-drift         Compare low-precision drift across engines
  check-reference-parity        Compare a running server with the Python reference
  check-bert-parity             Compare BERT embeddings with SentenceTransformers
  check-e5-parity               Compare XLM-R/E5 embeddings with SentenceTransformers
  check-qwen3-parity            Compare Qwen3 embeddings with SentenceTransformers
  check-tokenizer-parity        Compare the C BPE tokenizer with stored vectors
  check-wordpiece-parity        Compare the C WordPiece tokenizer with stored vectors
  check-model-matrix            Smoke-check supported model variants
  check-server-api              Verify a running local HTTP server
  check-cli                     Drive the CLI binary through its modes and flags
  check-workspace-api           Check the model/workspace API with real weights
  check-perplexity-sdk-compat   Verify a running server through the Perplexity SDK
  check-openai-sdk-compat       Verify a running server through the OpenAI SDK

Benchmarks:
  bench-contextual-api          Benchmark contextual inference through HTTP
  bench-late-rerank             Benchmark late-interaction MaxSim reranking
  bench-late-api                Benchmark HTTP reranking; optionally compare PyLate
  bench-matrix                  Run the batch-aware stdin benchmark matrix
  bench-qwen-tokenizer          Benchmark tokenizer encode paths
  bench-server-api              Benchmark concurrent HTTP embedding requests
EOF
}

main() {
  local command="${1:-help}"
  if (($#)); then
    shift
  fi

  case "$command" in
    help|-h|--help) usage ;;
    checks) run_checks "$@" ;;
    check-code-quality) check_code_quality "$@" ;;
    check-kernel-golden) check_kernel_golden "$@" ;;
    check-batch-parity) check_batch_parity "$@" ;;
    check-context-batch-parity) check_contextual_batch_parity "$@" ;;
    check-late-interaction) check_late_interaction "$@" ;;
    check-cuda-fast-gemm-parity) check_cuda_fast_gemm_parity "$@" ;;
    check-embedding-parity) check_embedding_parity "$@" ;;
    check-mlx-model-limits) check_mlx_model_limits "$@" ;;
    check-mlx-quantized-parity) check_mlx_quantized_parity "$@" ;;
    check-precision-drift) check_quant_precision_drift "$@" ;;
    check-reference-parity) check_reference_parity "$@" ;;
    check-bert-parity) check_bert_parity "$@" ;;
    check-e5-parity) check_e5_parity "$@" ;;
    check-qwen3-parity) check_qwen3_parity "$@" ;;
    check-tokenizer-parity) check_tokenizer_parity "$@" ;;
    check-wordpiece-parity) check_wordpiece_parity "$@" ;;
    check-model-matrix) check_model_matrix "$@" ;;
    check-perplexity-sdk-compat) check_perplexity_sdk_compat "$@" ;;
    check-openai-sdk-compat) check_openai_sdk_compat "$@" ;;
    check-server-api) check_server_api "$@" ;;
    check-cli) check_cli "$@" ;;
    check-workspace-api) check_workspace_api "$@" ;;
    bench-contextual-api) bench_contextual_api "$@" ;;
    bench-late-rerank) bench_late_rerank "$@" ;;
    bench-late-api) bench_late_api "$@" ;;
    bench-matrix) bench_matrix "$@" ;;
    bench-qwen-tokenizer) bench_qwen_tokenizer "$@" ;;
    bench-server-api) bench_server_api "$@" ;;
    *)
      err "unknown command: $command"
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
