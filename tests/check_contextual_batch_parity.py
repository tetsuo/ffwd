#!/usr/bin/env python3
"""Run the contextual batch-versus-singleton parity check.

The C side is tests/contextual_batch.c, built by the Makefile target
`parity-context-driver`; this script builds it and runs it with the requested
parameters. It compares the single-document path against the batched path in C
(no external reference), so it needs a model directory but no Python ML stack.

  tests/check_contextual_batch_parity.py --model-dir DIR
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_driver(cc: str, backend: str) -> Path:
    """Build the contextual-batch driver via the Makefile; return its path."""
    target = "parity-context-driver-mlx" if backend == "mlx" else "parity-context-driver"
    binary = "contextual_batch_mlx" if backend == "mlx" else "contextual_batch"
    proc = subprocess.run(["make", "-C", "tests", f"CC={cc}", target], cwd=ROOT,
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)
    return ROOT / "tests" / binary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--backend", choices=("cpu", "mlx"), default="cpu")
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--runs", type=int, default=0,
                    help="run N timed contextual document batches after parity")
    ap.add_argument("--max-diff", type=float, default=0.00005)
    ap.add_argument("--min-cosine", type=float, default=0.99999)
    ap.add_argument("--mlx-quant-bits", "--mlx-quantize-bits",
                    dest="mlx_quantize_bits", type=int, default=0,
                    choices=(0, 8))
    ap.add_argument("--mlx-quant-group-size", "--mlx-quantize-group-size",
                    dest="mlx_quantize_group_size", type=int, default=64)
    ap.add_argument("--synthetic-docs", type=int, default=0,
                    help="generate N synthetic documents instead of the smoke set")
    ap.add_argument("--synthetic-chunks", type=int, default=0,
                    help="chunks per synthetic document")
    ap.add_argument("--synthetic-repeats", type=int, default=1,
                    help="repeated text blocks per synthetic chunk")
    ap.add_argument("--synthetic-ragged", action="store_true",
                    help="scale synthetic chunk count by document index")
    args = ap.parse_args()
    if args.mlx_quantize_bits and args.backend != "mlx":
        raise SystemExit("--mlx-quant-bits requires --backend mlx")
    if args.mlx_quantize_group_size <= 0:
        raise SystemExit("--mlx-quant-group-size must be > 0")
    if args.backend == "mlx" and platform.system() != "Darwin":
        raise SystemExit("MLX checks require macOS")

    driver = build_driver(args.cc, args.backend)
    return subprocess.run(
        [
            str(driver),
            args.model_dir,
            str(args.runs),
            str(args.max_diff),
            str(args.min_cosine),
            str(args.mlx_quantize_bits),
            str(args.mlx_quantize_group_size),
            str(args.synthetic_docs),
            str(args.synthetic_chunks),
            str(args.synthetic_repeats),
            "1" if args.synthetic_ragged else "0",
        ],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
