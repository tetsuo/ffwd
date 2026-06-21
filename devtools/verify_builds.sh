#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

DEFAULT_MODEL_GLOB='models--perplexity-ai--pplx-embed-v1-0.6b/snapshots/*'
DEFAULT_CONTEXT_MODEL_GLOB='models--perplexity-ai--pplx-embed-context-v1-0.6B/snapshots/*'

model_dir=""
context_model_dir=""
binary="./ffwd-cli"
batch_size=4
skip_build=0
skip_mlx=0
platform_skip_mlx=0
skip_contextual=0
context_runs=0
late_model_dir=""
long=0
skip_code_quality=0
skip_hermetic=0
skip_tokenizer=0
skip_mlx_quantized=0

die() {
  echo "error: $*" >&2
  exit 2
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --model-dir DIR             0.6B model snapshot; defaults to the HF cache
  --context-model-dir DIR     context 0.6B snapshot; defaults to the HF cache
  --binary PATH               binary to test, default: ./ffwd-cli
  --batch-size N              batch size, default: 4
  --skip-build                skip make cpu / make gpu
  --skip-hermetic             skip the model-free make test suite
  --skip-mlx                  stop after the CPU build and CPU checks
  --skip-contextual           skip contextual parity checks
  --skip-tokenizer            skip Hugging Face tokenizer parity
  --skip-mlx-quantized        skip the MLX Q8 parity check
  --context-runs N            timed contextual batches per backend after parity
  --late-model-dir DIR        optional late-interaction 0.6B snapshot to verify
  --long                      also run the long CPU attention parity check
  --skip-code-quality         skip the final compiler/analyzer/tidy CI gate
  -h, --help                  show this help
EOF
}

find_cached_snapshot() {
  local glob="$1"
  local label="$2"
  local hub="$HOME/.cache/huggingface/hub"
  local last=""
  local p

  while IFS= read -r p; do
    if [[ -d "$p" && -f "$p/config.json" ]]; then
      last="$p"
    fi
  done < <({ compgen -G "$hub/$glob" || true; } | sort)

  if [[ -z "$last" ]]; then
    echo "no cached $label snapshot found; pass the model dir" >&2
    exit 1
  fi

  printf '%s\n' "$last"
}

find_default_model() {
  find_cached_snapshot "$DEFAULT_MODEL_GLOB" "pplx-embed-v1-0.6b"
}

find_default_context_model() {
  find_cached_snapshot "$DEFAULT_CONTEXT_MODEL_GLOB" "pplx-embed-context-v1-0.6b"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  (cd "$ROOT" && "$@")
}

run_parity() {
  local binary_arg="$1"
  local model_dir_arg="$2"
  local backend="$3"
  local batch_size_arg="$4"
  local use_long="$5"

  local cmd=(
    "$ROOT/all.bash" check-batch-parity
    --binary "$binary_arg"
    --model-dir "$model_dir_arg"
    --backend "$backend"
    --batch-size "$batch_size_arg"
  )

  if (( use_long )); then
    cmd+=(--long)
  fi

  run "${cmd[@]}"
}

run_contextual() {
  local model_dir_arg="$1"
  local backend="$2"
  local runs="$3"

  run \
    "$ROOT/all.bash" check-context-batch-parity \
    --model-dir "$model_dir_arg" \
    --backend "$backend" \
    --runs "$runs"
}

run_late() {
  local model_dir_arg="$1"
  local backend="$2"

  run \
    "$ROOT/all.bash" check-late-interaction \
    --model-dir "$model_dir_arg" \
    --backend "$backend"
}

while (($#)); do
  case "$1" in
    --model-dir)
      (($# >= 2)) || die "--model-dir requires a value"
      model_dir="$2"
      shift 2
      ;;
    --model-dir=*)
      model_dir="${1#*=}"
      shift
      ;;

    --context-model-dir)
      (($# >= 2)) || die "--context-model-dir requires a value"
      context_model_dir="$2"
      shift 2
      ;;
    --context-model-dir=*)
      context_model_dir="${1#*=}"
      shift
      ;;

    --binary)
      (($# >= 2)) || die "--binary requires a value"
      binary="$2"
      shift 2
      ;;
    --binary=*)
      binary="${1#*=}"
      shift
      ;;

    --batch-size)
      (($# >= 2)) || die "--batch-size requires a value"
      batch_size="$2"
      shift 2
      ;;
    --batch-size=*)
      batch_size="${1#*=}"
      shift
      ;;

    --skip-build)
      skip_build=1
      shift
      ;;
    --skip-hermetic)
      skip_hermetic=1
      shift
      ;;
    --skip-mlx)
      skip_mlx=1
      shift
      ;;
    --skip-contextual)
      skip_contextual=1
      shift
      ;;
    --skip-tokenizer)
      skip_tokenizer=1
      shift
      ;;
    --skip-mlx-quantized)
      skip_mlx_quantized=1
      shift
      ;;

    --context-runs)
      (($# >= 2)) || die "--context-runs requires a value"
      context_runs="$2"
      shift 2
      ;;
    --context-runs=*)
      context_runs="${1#*=}"
      shift
      ;;

    --late-model-dir)
      (($# >= 2)) || die "--late-model-dir requires a value"
      late_model_dir="$2"
      shift 2
      ;;
    --late-model-dir=*)
      late_model_dir="${1#*=}"
      shift
      ;;

    --long)
      long=1
      shift
      ;;
    --skip-code-quality)
      skip_code_quality=1
      shift
      ;;

    -h|--help)
      usage
      exit 0
      ;;

    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" && $skip_mlx -eq 0 ]]; then
  echo "info: MLX checks require macOS; skipping on $(uname -s)"
  skip_mlx=1
  platform_skip_mlx=1
fi

if [[ -z "$model_dir" ]]; then
  model_dir="$(find_default_model)"
fi

if (( skip_contextual )); then
  context_model_dir=""
elif [[ -z "$context_model_dir" ]]; then
  context_model_dir="$(find_default_context_model)"
fi

if (( ! skip_hermetic )); then
  run make test
fi

if (( ! skip_build )); then
  run make cpu
fi

run "$ROOT/all.bash" check-kernel-golden --blas

if (( ! skip_tokenizer )); then
  run "$ROOT/all.bash" check-tokenizer-parity --model-dir "$model_dir"
fi

run "$ROOT/all.bash" check-workspace-api --model-dir "$model_dir"

run_parity "$binary" "$model_dir" cpu "$batch_size" 0

if (( long )); then
  run_parity "$binary" "$model_dir" cpu "$batch_size" 1
fi

if [[ -n "$context_model_dir" ]]; then
  run_contextual "$context_model_dir" cpu "$context_runs"
fi

if [[ -n "$late_model_dir" ]]; then
  run_late "$late_model_dir" cpu
fi

if (( skip_mlx )); then
  if (( platform_skip_mlx && ! skip_code_quality )); then
    run "$ROOT/all.bash" check-code-quality
  fi
  if (( skip_build )); then
    echo "ok: CPU checks passed; builds and MLX checks skipped"
  else
    echo "ok: CPU build verified; MLX build skipped"
  fi
  exit 0
fi

if (( ! skip_build )); then
  run make gpu
fi

run_parity "$binary" "$model_dir" mlx "$batch_size" 0

run \
  "$ROOT/all.bash" check-mlx-model-limits \
  --binary "$binary" \
  --model-dir "$model_dir"

if (( ! skip_mlx_quantized )); then
  run \
    "$ROOT/all.bash" check-mlx-quantized-parity \
    --binary "$binary" \
    --model-dir "$model_dir" \
    --batch-size "$batch_size"
fi

if [[ -n "$context_model_dir" ]]; then
  run_contextual "$context_model_dir" mlx "$context_runs"
fi

if [[ -n "$late_model_dir" ]]; then
  run_late "$late_model_dir" mlx
fi

if (( ! skip_code_quality )); then
  run "$ROOT/all.bash" check-code-quality
fi

echo "verify ok"
