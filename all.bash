#!/usr/bin/env bash

set -uo pipefail

if [[ -n "${ZSH_VERSION:-}" ]]; then
  ROOT="$(cd -- "$(dirname -- "${(%):-%x}")" && pwd -P)"
else
  ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
fi
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

_ALL_CMDS="help checks check-code-quality check-kernel-golden check-batch-parity \
check-context-batch-parity check-late-interaction check-cuda-fast-gemm-parity \
check-embedding-parity check-mlx-model-limits check-mlx-quantized-parity \
check-precision-drift check-reference-parity check-bert-parity check-e5-parity \
check-qwen3-parity check-tokenizer-parity check-wordpiece-parity check-model-matrix \
check-server-api check-cli check-workspace-api check-perplexity-sdk-compat \
check-openai-sdk-compat bench-contextual-api bench-late-rerank bench-late-api \
bench-matrix bench-qwen-tokenizer bench-server-api"

if [[ -n "${ZSH_VERSION:-}" ]]; then
  _all_complete_zsh() {
    # shellcheck disable=SC2296
    local -a cmds=("${(ps: :)_ALL_CMDS}")
    _arguments "1: :(${cmds[*]})"
  }
  compdef _all_complete_zsh ./all.bash all.bash all 2>/dev/null || true
else
  _all_complete() {
    local cur
    cur="${COMP_WORDS[COMP_CWORD]}"
    # shellcheck disable=SC2207
    COMPREPLY=( $(compgen -W "$_ALL_CMDS" -- "$cur") )
  }
  complete -F _all_complete ./all.bash
  complete -F _all_complete all.bash
  complete -F _all_complete all
fi

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

  ███████╗███████╗██╗    ██╗██████╗
  ██╔════╝██╔════╝██║    ██║██╔══██╗
  █████╗  █████╗  ██║ █╗ ██║██║  ██║
  ██╔══╝  ██╔══╝  ██║███╗██║██║  ██║
  ██║     ██║     ╚███╔███╔╝██████╔╝
  ╚═╝     ╚═╝      ╚══╝╚══╝ ╚═════╝

  Usage: ./all.bash COMMAND [ARGS...]

  ┏━━[ checks ]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃                                                                ┃
  ┃  check-code-quality            Run clang builds, analysis      ┃
  ┃  check-kernel-golden           Run math-kernel golden tests    ┃
  ┃  check-batch-parity            Compare singleton vs batched    ┃
  ┃  check-context-batch-parity    Compare contextual execution    ┃
  ┃  check-cuda-fast-gemm-parity   Gate CUDA reduced-precision     ┃
  ┃  check-embedding-parity        Compare embeddings              ┃
  ┃  check-late-interaction        Check late-interaction vectors  ┃
  ┃  check-mlx-model-limits        Check MLX metadata rejection    ┃
  ┃  check-mlx-quantized-parity    Compare quantized embeddings    ┃
  ┃  check-precision-drift         Compare low-precision drift     ┃
  ┃  check-reference-parity        Compare server vs Python ref    ┃
  ┃  check-bert-parity             Compare BERT embeddings         ┃
  ┃  check-e5-parity               Compare XLM-R/E5 embeddings     ┃
  ┃  check-qwen3-parity            Compare Qwen3 embeddings        ┃
  ┃  check-tokenizer-parity        Compare BPE tokenizer           ┃
  ┃  check-wordpiece-parity        Compare WordPiece tokenizer     ┃
  ┃  check-model-matrix            Smoke-check model variants      ┃
  ┃  check-server-api              Verify running HTTP server      ┃
  ┃  check-cli                     Drive CLI through modes/flags   ┃
  ┃  check-workspace-api           Check model/workspace API       ┃
  ┃  check-perplexity-sdk-compat   Verify via Perplexity SDK       ┃
  ┃  check-openai-sdk-compat       Verify via OpenAI SDK           ┃
  ┃                                                                ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  ┏━━[ benchmarks ]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃                                                                ┃
  ┃  bench-contextual-api          Benchmark contextual inference  ┃
  ┃  bench-late-rerank             Benchmark late-interaction      ┃
  ┃  bench-late-api                Benchmark HTTP reranking        ┃
  ┃  bench-matrix                  Run batch-aware stdin matrix    ┃
  ┃  bench-qwen-tokenizer          Benchmark tokenizer encode      ┃
  ┃  bench-server-api              Benchmark concurrent HTTP       ┃
  ┃                                                                ┃
  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
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

if [[ -n "${ZSH_VERSION:-}" ]]; then
  [[ "${ZSH_EVAL_CONTEXT:-}" == "toplevel" ]] && main "$@"
else
  [[ "${BASH_SOURCE[0]}" == "$0" ]] && main "$@"
fi
