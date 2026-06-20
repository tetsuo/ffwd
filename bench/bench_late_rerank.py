#!/usr/bin/env python3
"""Benchmark late-interaction MaxSim reranking shapes.

Builds the C bench (bench/bench_late_rerank.c) via the Makefile and runs it
against a late model directory (e.g. pplx-embed-v1-late-0.6b). python3 only.

  bench/bench_late_rerank.py --model-dir DIR
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(cc: str, backend: str) -> Path:
    """Build the late-rerank bench via the Makefile; return its path."""
    target = "bench_late_rerank_mlx" if backend == "mlx" else "bench_late_rerank"
    proc = subprocess.run(["make", "-C", "bench", f"CC={cc}", target], cwd=ROOT,
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)
    return ROOT / "bench" / target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--backend", choices=("cpu", "mlx"), default="cpu")
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--doc-repeat", type=int, default=3)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = ap.parse_args()

    binary = build(args.cc, args.backend)
    return subprocess.run(
        [str(binary), args.model_dir, str(args.candidates),
         str(args.doc_repeat), str(args.runs)],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
