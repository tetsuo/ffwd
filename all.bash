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

_ALL_CMDS="help checks check-code-quality check-sanitize check-kernel-golden check-batch-parity \
check-context-batch-parity check-late-interaction check-cuda-fast-gemm-parity \
check-embedding-parity check-mlx-model-limits check-mlx-quantized-parity \
check-reference-parity check-bert-parity check-e5-parity \
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
  run ./tests/check_code_quality.sh "$@"
}

# The dynamic-analysis counterpart to check-code-quality (which is static only:
# the clang analyzer + clang-tidy). It rebuilds the hermetic C suites - the
# library test_* and the server test_* - with the address and undefined-behavior
# sanitizers (MODE=debug; see build.mk) and runs them. This catches the
# memory-safety and undefined-behavior class of bug the static analyzer misses:
# the 2026-06-21 strdup pointer corruption was exactly such a bug (ASan and
# valgrind caught it, the analyzer did not). It is model-free, so it runs on both
# macOS and Linux. The debug build lands in build/debug/blas, a tree separate
# from the release artifacts and the root ffwd-cli/ffwd-server symlinks, so it
# never disturbs them. Extra args pass through to make (e.g. SANITIZE=thread to
# swap the sanitizer set). Note on leaks: the leak detector inside ASan
# (LeakSanitizer) is Linux-only - on macOS `-fsanitize=address` does memory-error
# and UB checking but not leak detection (detect_leaks is unsupported there), so
# a Linux run of this check is stronger. macOS leak detection is a separate tool
# (`leaks`, or Instruments), not part of this command.
check_sanitize() {
  run make MODE=debug test "$@"
  # The test rules compile-and-link each .c in a single step, so -g makes macOS
  # auto-run dsymutil and drop *.dSYM symbol bundles next to the in-tree binaries
  # (Linux produces none). They are regenerable debug-symbol output, not build
  # inputs, and tests/*.dSYM is not gitignored - remove the bundles so the run
  # leaves the tree as clean as a release `make test` does. The binaries stay.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    rm -rf "$ROOT"/tests/*.dSYM "$ROOT"/tools/server/*.dSYM
  fi
}

# The comprehensive local suite, folded in from the old verify_builds.sh so
# all.bash is the single entry point. It builds, runs the hermetic C tests, then
# reruns them under the address and undefined-behavior sanitizers (the dynamic
# gate), then the math-kernel goldens and the hermetic tokenizer checks (all
# model-free), then the CPU model checks and - on macOS - the MLX model checks,
# and finishes with the static compiler/analyzer/tidy gate. Model-dependent steps
# run only when --model-dir is given. CUDA validation stays a separate RunPod
# step.
run_checks() {
  local model_dir="" context_model_dir="" late_model_dir=""
  local batch_size=4 context_runs=0
  local long=0 skip_build=0 skip_hermetic=0 skip_mlx=0
  local skip_code_quality=0 skip_sanitize=0 skip_tokenizer=0 skip_mlx_quantized=0
  local platform_skip_mlx=0

  while (($#)); do
    case "$1" in
      --model-dir) model_dir="$2"; shift 2 ;;
      --model-dir=*) model_dir="${1#*=}"; shift ;;
      --context-model-dir) context_model_dir="$2"; shift 2 ;;
      --context-model-dir=*) context_model_dir="${1#*=}"; shift ;;
      --late-model-dir) late_model_dir="$2"; shift 2 ;;
      --late-model-dir=*) late_model_dir="${1#*=}"; shift ;;
      --batch-size) batch_size="$2"; shift 2 ;;
      --batch-size=*) batch_size="${1#*=}"; shift ;;
      --context-runs) context_runs="$2"; shift 2 ;;
      --context-runs=*) context_runs="${1#*=}"; shift ;;
      --long) long=1; shift ;;
      --skip-build) skip_build=1; shift ;;
      --skip-hermetic) skip_hermetic=1; shift ;;
      --skip-mlx) skip_mlx=1; shift ;;
      --skip-code-quality) skip_code_quality=1; shift ;;
      --skip-sanitize) skip_sanitize=1; shift ;;
      --skip-tokenizer) skip_tokenizer=1; shift ;;
      --skip-mlx-quantized) skip_mlx_quantized=1; shift ;;
      -h|--help)
        cat <<'USAGE'
Usage: ./all.bash checks [options]
  --model-dir DIR            model for the workspace/batch parity checks
  --context-model-dir DIR    contextual model (enables contextual parity)
  --late-model-dir DIR       late-interaction model (enables late checks)
  --batch-size N             batch size for the parity checks (default 4)
  --context-runs N           timed contextual batches per backend
  --long                     also run the long CPU attention parity check
  --skip-build               skip make cpu / make gpu
  --skip-hermetic            skip the model-free make test suite
  --skip-sanitize            skip the ASan/UBSan hermetic run (MODE=debug)
  --skip-mlx                 stop after the CPU build and CPU checks
  --skip-tokenizer           skip the hermetic tokenizer parity checks
  --skip-mlx-quantized       skip the MLX Q8 parity check
  --skip-code-quality        skip the compiler/analyzer/tidy gate
USAGE
        return 0 ;;
      *) err "checks: unknown option: $1"; return 2 ;;
    esac
  done

  if [[ "$(uname -s)" != "Darwin" && $skip_mlx -eq 0 ]]; then
    warn "MLX checks require macOS; skipping the MLX phase on $(uname -s)"
    skip_mlx=1
    platform_skip_mlx=1
  fi

  # Any failing step aborts the suite with a non-zero exit, as the old script did.
  _step() { "$@" || { err "checks: step failed -> $*"; exit 1; }; }

  # Model-free: build, hermetic C tests, the ASan/UBSan rerun of those tests,
  # kernel goldens, hermetic tokenizers. The sanitize step is itself a hermetic
  # run, so --skip-hermetic suppresses it too.
  (( skip_hermetic )) || _step run make test
  (( skip_sanitize || skip_hermetic )) || _step check_sanitize
  (( skip_build )) || _step run make cpu
  _step check_kernel_golden
  if (( ! skip_tokenizer )); then
    _step check_tokenizer_parity
    _step check_wordpiece_parity
  fi

  # CPU model checks (need a model directory).
  if [[ -n "$model_dir" ]]; then
    _step check_workspace_api --model-dir "$model_dir"
    # An empty array under `set -u` errors on bash 3.2 (macOS default), so pass
    # --long with an explicit branch rather than expanding a maybe-empty array.
    if (( long )); then
      _step check_batch_parity --model-dir "$model_dir" --backend cpu \
        --batch-size "$batch_size" --long
    else
      _step check_batch_parity --model-dir "$model_dir" --backend cpu \
        --batch-size "$batch_size"
    fi
    if [[ -n "$context_model_dir" ]]; then
      _step check_contextual_batch_parity --model-dir "$context_model_dir" \
        --backend cpu --runs "$context_runs"
    fi
    if [[ -n "$late_model_dir" ]]; then
      _step check_late_interaction --model-dir "$late_model_dir" --backend cpu
    fi
  else
    warn "checks: no --model-dir; skipping CPU model checks (workspace, batch/contextual/late parity)"
  fi

  # MLX phase (macOS only): rebuild as the GPU backend, rerun the model checks.
  if (( ! skip_mlx )); then
    (( skip_build )) || _step run make gpu
    if [[ -n "$model_dir" ]]; then
      _step check_batch_parity --model-dir "$model_dir" --backend mlx \
        --batch-size "$batch_size"
      _step check_mlx_model_limits --model-dir "$model_dir"
      (( skip_mlx_quantized )) || _step check_mlx_quantized_parity \
        --model-dir "$model_dir" --batch-size "$batch_size"
      if [[ -n "$context_model_dir" ]]; then
        _step check_contextual_batch_parity --model-dir "$context_model_dir" \
          --backend mlx --runs "$context_runs"
      fi
      if [[ -n "$late_model_dir" ]]; then
        _step check_late_interaction --model-dir "$late_model_dir" --backend mlx
      fi
    fi
  fi

  (( skip_code_quality )) || _step check_code_quality
  info "checks: all green"
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
  run_uv ./tests/check_embedding_parity.py "$@"
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

  Aggregate:  checks   full local suite: build + hermetic tests + CPU/MLX
                       model checks + code quality. Pass --model-dir DIR to
                       add the model checks; `checks --help` for options.

  ┏━━[ checks ]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃                                                                ┃
  ┃  check-code-quality            Run clang builds, analysis      ┃
  ┃  check-sanitize                Run hermetic tests under ASan   ┃
  ┃  check-kernel-golden           Run math-kernel golden tests    ┃
  ┃  check-batch-parity            Compare singleton vs batched    ┃
  ┃  check-context-batch-parity    Compare contextual execution    ┃
  ┃  check-cuda-fast-gemm-parity   Gate CUDA reduced-precision     ┃
  ┃  check-embedding-parity        Compare embeddings              ┃
  ┃  check-late-interaction        Check late-interaction vectors  ┃
  ┃  check-mlx-model-limits        Check MLX metadata rejection    ┃
  ┃  check-mlx-quantized-parity    Compare quantized embeddings    ┃
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
    check-sanitize) check_sanitize "$@" ;;
    check-kernel-golden) check_kernel_golden "$@" ;;
    check-batch-parity) check_batch_parity "$@" ;;
    check-context-batch-parity) check_contextual_batch_parity "$@" ;;
    check-late-interaction) check_late_interaction "$@" ;;
    check-cuda-fast-gemm-parity) check_cuda_fast_gemm_parity "$@" ;;
    check-embedding-parity) check_embedding_parity "$@" ;;
    check-mlx-model-limits) check_mlx_model_limits "$@" ;;
    check-mlx-quantized-parity) check_mlx_quantized_parity "$@" ;;
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
