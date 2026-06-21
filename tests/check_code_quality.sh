#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

# Lint and analyze every C source under libffwd/, tests/, and bench/. Each
# platform keeps its own GPU backend and skips the other's, which it does not
# build (macOS builds MLX, Linux builds CUDA).
SYS="$(uname -s)"
SRC_SOURCES=()
for f in libffwd/*.c; do
  [[ "$SYS" == Darwin && "$f" == libffwd/backend_cuda.c ]] && continue
  [[ "$SYS" != Darwin && ( "$f" == libffwd/mlx.c || "$f" == libffwd/backend_mlx.c ) ]] && continue
  SRC_SOURCES+=("$f")
done

TEST_C_SOURCES=(tests/*.c)
BENCH_C_SOURCES=(bench/*.c)

# CUDA backend sources are C++17 and only built on Linux. Parse them with the
# clang CUDA front end in host-only mode so no GPU or device toolchain is
# required; macOS has no CUDA toolkit, so skip them there.
CU_SOURCES=()
if [[ "$SYS" != Darwin ]]; then
  for f in libffwd/*.cu; do
    [[ -e "$f" ]] && CU_SOURCES+=("$f")
  done
fi

ANALYZE_SOURCES=("${SRC_SOURCES[@]}")
TIDY_SOURCES=("${SRC_SOURCES[@]}" "${TEST_C_SOURCES[@]}" "${BENCH_C_SOURCES[@]}")

ANALYZE_COMMON=(
  clang --analyze -Xanalyzer -analyzer-output=text
  -Wall -Wextra
  -I. -Ilibffwd -Ideps/ae
)

PLATFORM_CFLAGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  PLATFORM_CFLAGS=(
    -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DUSE_MLX
    -I/opt/homebrew/opt/mlx-c/include
    -I/opt/homebrew/include
    -I/usr/local/include
  )
else
  PLATFORM_CFLAGS=(
    -DUSE_BLAS -DUSE_OPENBLAS
    -I/usr/include/openblas
  )
fi

# Flags for the CUDA backend front end. CUDA_HOME matches the Makefile default.
# --cuda-host-only parses host code only so device codegen and libdevice are not
# needed; the macros mirror the nvcc command in the Makefile.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_CFLAGS=(
  -x cuda --cuda-host-only "--cuda-path=$CUDA_HOME"
  -std=c++17
  -DUSE_BLAS -DUSE_OPENBLAS -DUSE_GPU
  -I"$CUDA_HOME/include"
  -I/usr/include/openblas
)

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_cmd "$@"
  "$@"
}

run_analyze() {
  local output
  local status

  print_cmd "$@"

  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e

  if [[ -n "$output" ]]; then
    printf '%s\n' "$output"
  fi

  if (( status != 0 )); then
    return "$status"
  fi

  if grep -q 'warning:' <<<"$output"; then
    echo "clang analyzer warnings found" >&2
    return 1
  fi
}

have() {
  command -v "$1" >/dev/null 2>&1
}

find_clang_tidy() {
  local brew_path="/opt/homebrew/opt/llvm@21/bin/clang-tidy"

  if command -v clang-tidy >/dev/null 2>&1; then
    command -v clang-tidy
    return 0
  fi

  if [[ -x "$brew_path" ]]; then
    printf '%s\n' "$brew_path"
    return 0
  fi

  return 1
}

main() {
  local clang_tidy

  if ! have clang; then
    echo "clang not found" >&2
    return 1
  fi

  run git --no-pager diff --check
  run make cpu CC=clang

  run_analyze "${ANALYZE_COMMON[@]}" "${PLATFORM_CFLAGS[@]}" \
    "${ANALYZE_SOURCES[@]}"
  run_analyze "${ANALYZE_COMMON[@]}" "${PLATFORM_CFLAGS[@]}" \
    -I. -Ilibffwd -Itests "${TEST_C_SOURCES[@]}"
  run_analyze "${ANALYZE_COMMON[@]}" "${PLATFORM_CFLAGS[@]}" \
    -I. -Ilibffwd -Ibench -Itools/server "${BENCH_C_SOURCES[@]}"

  if (( ${#CU_SOURCES[@]} > 0 )); then
    run_analyze "${ANALYZE_COMMON[@]}" "${CUDA_CFLAGS[@]}" \
      "${CU_SOURCES[@]}"
  fi

  if clang_tidy="$(find_clang_tidy)"; then
    run \
      "$clang_tidy" \
      --config-file=.clang-tidy \
      "${TIDY_SOURCES[@]}" \
      -- \
      -std=c17 \
      "${PLATFORM_CFLAGS[@]}" \
      -I. \
      -Ilibffwd \
      -Itests \
      -Ibench \
      -Itools/server \
      -Ideps/ae

    if (( ${#CU_SOURCES[@]} > 0 )); then
      run \
        "$clang_tidy" \
        --config-file=.clang-tidy \
        "${CU_SOURCES[@]}" \
        -- \
        "${CUDA_CFLAGS[@]}" \
        -I. \
        -Ilibffwd \
        -Ideps/ae
    fi
  else
    echo "clang-tidy not found; skipping"
  fi

  if [[ "$(uname -s)" == "Darwin" ]]; then
    run make gpu CC=clang
  fi
}

main "$@"
