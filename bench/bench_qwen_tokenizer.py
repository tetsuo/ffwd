#!/usr/bin/env python3
"""Benchmark Qwen tokenizer encode paths.

Builds the C tokenizer bench via the Makefile (`make -C bench bench_tokenizer`)
and runs it against a Qwen model's vocab.json with a curated text set.

  bench/bench_qwen_tokenizer.py --model-dir DIR
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital of France.",
    "document: dense retrieval systems need stable tokenization.",
    "unicode: cafe, café, Istanbul, İstanbul, emoji \U0001f50d",
    "mixed whitespace:\n  one\t two   three\n\nfour",
    ("long document: " + "embedding inference should batch tokenization and "
     "model execution carefully. ") * 32,
]


def build(cc: str) -> Path:
    """Build bench/bench_tokenizer via the Makefile; return its path."""
    proc = subprocess.run(["make", "-C", "bench", f"CC={cc}", "bench_tokenizer"],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)
    return ROOT / "bench" / "bench_tokenizer"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--text", action="append",
                    help="text to benchmark; may be repeated")
    args = ap.parse_args()

    binary = build(args.cc)
    vocab = Path(args.model_dir) / "vocab.json"
    texts = args.text if args.text else DEFAULT_TEXTS
    return subprocess.run([str(binary), str(vocab), str(args.runs), *texts],
                          cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
