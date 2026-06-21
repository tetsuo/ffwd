#!/usr/bin/env python3
"""Compare embeddings from two model directories through the CLI."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Each backend is its own binary; the backend label alone does not change the
# CLI's behaviour, so cpu-vs-cuda comparisons must run two different binaries.
BUILD_DIR = {"cpu": "blas", "mlx": "mlx", "cuda": "cuda"}


def default_binary(backend: str) -> Path:
    return ROOT / "build" / "release" / BUILD_DIR[backend] / "ffwd-cli"


def resolve_binary(backend: str, explicit: str, shared: str) -> Path:
    """Pick the CLI for one side: an explicit --binary-{a,b}, then the shared
    --binary override, then the per-backend build-tree default."""
    if explicit:
        return Path(explicit)
    if shared:
        return Path(shared)
    return default_binary(backend)


def ensure_binary(backend: str, binary: Path) -> None:
    """Build the backend's CLI on demand when using the build-tree default."""
    if binary == default_binary(backend) and not binary.exists():
        subprocess.run(["make", backend], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    if not binary.exists():
        raise SystemExit(f"binary not found for backend {backend}: {binary}")


TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital of France.",
    "document: Berlin is the capital of Germany.",
    "The quick brown fox jumps over the lazy dog near the riverbank",
    (
        "Retrieval augmented generation combines embedding models with external "
        "knowledge sources and ranks documents by semantic similarity."
    ),
]


def run_model(binary: str, model_dir: str, texts: list[str], batch_size: int,
              backend: str, threads: int) -> list[list[float]]:
    cmd = [binary, "-d", model_dir, "--stream", "-b", str(batch_size)]
    if backend == "cpu" and threads > 0:
        cmd.extend(["-t", str(threads)])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write("\n".join(texts) + "\n")
    proc.stdin.close()

    embeddings = []
    for _ in texts:
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"stdin exited early for {model_dir}\n{stderr}")
        row = json.loads(line)
        if "error" in row:
            raise RuntimeError(f"stdin error for {model_dir}: {row}")
        embeddings.append([float(x) for x in row["embedding"]])

    rc = proc.wait(timeout=10)
    if rc != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"stdin failed for {model_dir}: rc={rc}\n{stderr}")
    return embeddings


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default="",
                    help="shared CLI override for both sides; default resolves per backend")
    ap.add_argument("--binary-a", default="", help="CLI for --a (overrides --binary)")
    ap.add_argument("--binary-b", default="", help="CLI for --b (overrides --binary)")
    ap.add_argument("--a", required=True, help="reference model directory")
    ap.add_argument("--b", required=True, help="comparison model directory")
    ap.add_argument("--backend", choices=["cpu", "mlx", "cuda"], default="cpu")
    ap.add_argument("--backend-a", choices=["cpu", "mlx", "cuda"],
                    help="backend for --a; defaults to --backend")
    ap.add_argument("--backend-b", choices=["cpu", "mlx", "cuda"],
                    help="backend for --b; defaults to --backend")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-diff", type=float, default=0.01)
    ap.add_argument("--min-cos", type=float, default=0.999)
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    backend_a = args.backend_a or args.backend
    backend_b = args.backend_b or args.backend
    binary_a = resolve_binary(backend_a, args.binary_a, args.binary)
    binary_b = resolve_binary(backend_b, args.binary_b, args.binary)
    ensure_binary(backend_a, binary_a)
    ensure_binary(backend_b, binary_b)
    emb_a = run_model(str(binary_a), args.a, TEXTS, args.batch_size,
                      backend_a, args.threads)
    emb_b = run_model(str(binary_b), args.b, TEXTS, args.batch_size,
                      backend_b, args.threads)

    worst_diff = 0.0
    worst_cos = 1.0
    for i, (a, b) in enumerate(zip(emb_a, emb_b)):
        if len(a) != len(b):
            raise SystemExit(f"dimension mismatch at text {i}: {len(a)} vs {len(b)}")
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        cos = cosine(a, b)
        worst_diff = max(worst_diff, max_diff)
        worst_cos = min(worst_cos, cos)
        print(f"[{i}] cosine={cos:.8f} max_abs_diff={max_diff:.8g}")

    print(
        f"ok: backend_a={backend_a} backend_b={backend_b} "
        f"worst_diff={worst_diff:.8g} worst_cos={worst_cos:.8f}"
    )
    if worst_diff > args.max_diff or worst_cos < args.min_cos:
        raise SystemExit(
            f"model parity outside threshold: diff={worst_diff:.8g}, cos={worst_cos:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
