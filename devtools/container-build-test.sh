#!/usr/bin/env bash
# Build, test and bench the OpenBLAS backend in a throwaway Ubuntu container to
# catch Linux portability breakage a macOS/Accelerate build hides. Needs Apple's
# `container` CLI running. Run from anywhere in the repo: it finds the repo root
# via git, bind-mounts it at /work, installs the Runpod toolchain, re-execs
# itself inside with --in-container, and runs make cpu/test/bench/bench-server-
# utils/debug, then `make clean` (skip with KEEP=1). Override the image with
# IMAGE=ubuntu:22.04. Exits non-zero at the first failing step.
set -euo pipefail

IMAGE="${IMAGE:-ubuntu:24.04}"

# --- inside the container: bootstrap toolchain, then build/test ---------------
run_in_container() {
    export DEBIAN_FRONTEND=noninteractive

    echo "==================== apt bootstrap ===================="
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates libcjson-dev libopenblas-dev \
        pkg-config git >/dev/null

    echo "==================== system / toolchain ===================="
    uname -a
    cc --version | sed -n '1p'
    echo "libcjson  $(pkg-config --modversion libcjson 2>/dev/null || echo MISSING) :: $(pkg-config --cflags --libs libcjson 2>/dev/null)"
    echo "openblas  $(pkg-config --modversion openblas 2>/dev/null || echo MISSING) :: $(pkg-config --cflags --libs openblas 2>/dev/null)"

    cd /work
    # /work is a bind mount; git refuses to operate on it without this opt-in.
    git config --global --add safe.directory /work 2>/dev/null || true
    make clean >/dev/null 2>&1 || true

    echo "==================== make cpu ===================="
    make cpu
    ls -la ffwd-cli ffwd-server
    ./ffwd-cli --version || true

    echo "==================== make test ===================="
    make test

    echo "==================== make bench ===================="
    make bench

    echo "==================== make bench-server-utils ===================="
    make bench-server-utils

    echo "==================== make debug (sanitizers) ===================="
    make debug

    if [ -n "${KEEP:-}" ]; then
        echo "==================== KEEP set: leaving artifacts in tree ===================="
    else
        echo "==================== make clean ===================="
        make clean
        residual=$(ls -d build ffwd-cli ffwd-server 2>/dev/null | wc -l | tr -d ' ')
        echo "residual build artifacts: $residual (want 0)"
    fi

    echo "==================== ALL STEPS COMPLETED ===================="
}

# --- host side: launch the container and re-exec this script inside it --------
launch_on_host() {
    command -v container >/dev/null 2>&1 || {
        echo "error: Apple 'container' CLI not found on PATH" >&2; exit 1; }

    local repo_root script_rel
    repo_root="$(git rev-parse --show-toplevel)"
    # Path of this script relative to the repo root, so it resolves under /work.
    script_rel="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    script_rel="${script_rel#"$repo_root"/}"

    echo "repo:  $repo_root"
    echo "image: $IMAGE"
    exec container run --rm \
        --env "KEEP=${KEEP:-}" \
        --mount "source=$repo_root,target=/work" \
        "$IMAGE" \
        bash "/work/$script_rel" --in-container
}

if [ "${1:-}" = "--in-container" ]; then
    run_in_container
else
    launch_on_host
fi
