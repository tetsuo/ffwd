#!/usr/bin/env python3
"""Check CUDA reduced-precision quality against exact CUDA F32.

Covers both reduced-precision axes: compute mode (--mode tf32|bf16|16f,
selecting --gpu-gemm-mode) and weight storage (--weights bf16, selecting
--gpu-weight-dtype). The reference run always uses exact F32 weights and compute.
For BF16 weight storage the int8 byte-change rate is inherently large
(~7-15%) and informational only; cosine is the quality gate, so the int8
threshold defaults to 1.0 in that case unless overridden."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "ffwd-cli"


DEFAULT_TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital of France.",
    "document: Berlin is the capital of Germany.",
    (
        "document: Embedding models turn text into dense vectors for semantic "
        "retrieval. Long document inference stresses bidirectional attention "
        "because every token attends to every other token in the same packed "
        "sequence. "
    ) * 3,
    (
        "document: Embedding models turn text into dense vectors for semantic "
        "retrieval. Long document inference stresses bidirectional attention "
        "because every token attends to every other token in the same packed "
        "sequence. "
    ) * 6,
]
# The longest passage stays under ~512 tokens so short-context encoders
# (BERT/MiniLM/BGE max_position 512, XLM-R 514) accept it; larger 8x/14x inputs
# overflowed their position tables and were (correctly) rejected, which looked
# like a CUDA failure. Still long enough to stress packed-sequence attention.
#
# NOTE on coverage: these defaults are all DIFFERENT lengths, so they exercise
# only the per-sequence (ragged) attention path. The CUDA batched-attention path
# (cublasGemmBatchedEx) engages only when every sequence in the batch is the SAME
# length (see attn_batched_setup). To exercise it, pass >=2 identical texts as
# positional args, e.g. the same sentence four times with --batch-size 4.


def run_cuda(binary: str, model_dir: str, mode: str, weights: str,
             batch_size: int, texts: list[str]) -> list[list[float]]:
    cmd = [
        binary,
        "-d", model_dir,
        "--gpu-gemm-mode", mode,
        "--gpu-weight-dtype", weights,
        "--stream",
        "-b", str(batch_size),
    ]
    proc = subprocess.run(
        cmd,
        input="\n".join(texts) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )

    rows: list[list[float]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            raise RuntimeError(f"stdin returned error: {row}")
        rows.append([float(x) for x in row["embedding"]])
    if len(rows) != len(texts):
        raise RuntimeError(f"expected {len(texts)} rows, got {len(rows)}")
    return rows


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def quantize_int8_tanh(x: float) -> int:
    v = round(math.tanh(x) * 127.0)
    if v < -128:
        return -128
    if v > 127:
        return 127
    return int(v)


def compare(name: str, ref: list[list[float]], got: list[list[float]]) -> tuple[float, float, int]:
    worst_diff = 0.0
    worst_cos = 1.0
    int8_changes = 0
    for i, (a, b) in enumerate(zip(ref, got)):
        if len(a) != len(b):
            raise RuntimeError(f"dimension mismatch at row {i}: {len(a)} vs {len(b)}")
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        cos = cosine(a, b)
        changes = sum(
            quantize_int8_tanh(x) != quantize_int8_tanh(y)
            for x, y in zip(a, b)
        )
        worst_diff = max(worst_diff, max_diff)
        worst_cos = min(worst_cos, cos)
        int8_changes += changes
        print(
            f"{name} [{i}] cosine={cos:.8f} "
            f"max_abs_diff={max_diff:.8g} int8_changes={changes}"
        )
    return worst_diff, worst_cos, int8_changes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mode", choices=("f32", "tf32", "bf16", "16f"),
                    default="tf32")
    ap.add_argument("--weights", choices=("f32", "bf16"), default="f32")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-diff", type=float, default=None)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--max-int8-change-rate", type=float, default=None)
    ap.add_argument("texts", nargs="*")
    args = ap.parse_args()

    if args.batch_size < 2:
        raise SystemExit("--batch-size must be >= 2")
    if args.mode == "f32" and args.weights == "f32":
        raise SystemExit("nothing to check: --mode and --weights are both f32")
    if args.max_diff is None:
        args.max_diff = 0.05 if args.weights == "bf16" else 0.01
    if args.max_int8_change_rate is None:
        args.max_int8_change_rate = 1.0 if args.weights == "bf16" else 0.02

    texts = args.texts or DEFAULT_TEXTS
    ref_batch = run_cuda(args.binary, args.model_dir, "f32", "f32",
                         args.batch_size, texts)
    fast_batch = run_cuda(args.binary, args.model_dir, args.mode,
                          args.weights, args.batch_size, texts)
    fast_single = run_cuda(args.binary, args.model_dir, args.mode,
                           args.weights, 1, texts)

    qdiff, qcos, qchanges = compare("quality", ref_batch, fast_batch)
    sdiff, scos, schanges = compare("shape", fast_single, fast_batch)

    total_values = len(texts) * len(ref_batch[0])
    qrate = qchanges / total_values
    srate = schanges / total_values
    print(
        f"ok: mode={args.mode} weights={args.weights} "
        f"batch_size={args.batch_size} "
        f"quality_diff={qdiff:.8g} quality_cos={qcos:.8f} "
        f"quality_int8_change_rate={qrate:.6f} "
        f"shape_diff={sdiff:.8g} shape_cos={scos:.8f} "
        f"shape_int8_change_rate={srate:.6f}"
    )

    failed = (
        qdiff > args.max_diff or qcos < args.min_cosine or
        sdiff > args.max_diff or scos < args.min_cosine or
        qrate > args.max_int8_change_rate or
        srate > args.max_int8_change_rate
    )
    if failed:
        raise SystemExit(
            f"cuda parity failed (mode={args.mode} weights={args.weights}): "
            f"quality diff={qdiff:g} cos={qcos:g} int8_rate={qrate:g}; "
            f"shape diff={sdiff:g} cos={scos:g} int8_rate={srate:g}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
