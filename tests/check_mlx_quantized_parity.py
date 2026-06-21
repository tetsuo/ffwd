#!/usr/bin/env python3
"""Compare normal MLX embeddings against opt-in quantized MLX weights."""

import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "tools/cli/ffwd-cli"


TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital of France.",
    "document: Berlin is the capital of Germany.",
    "document: Istanbul is a major city in Turkey.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    (
        "Retrieval augmented generation combines embedding models with "
        "external knowledge sources and ranks documents by semantic similarity."
    ),
]


def run(binary: str, model_dir: str, texts: list[str], batch_size: int,
        bits: int, group_size: int) -> list[list[float]]:
    cmd = [binary, "-d", model_dir, "--stream", "-b", str(batch_size)]
    if bits:
        cmd.extend([
            "--gpu-quant-bits", str(bits),
            "--gpu-quant-group-size", str(group_size),
        ])
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
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row:
            raise RuntimeError(f"stdin returned error: {row}")
        rows.append(row["embedding"])
    if len(rows) != len(texts):
        raise RuntimeError(f"expected {len(texts)} rows, got {len(rows)}")
    return rows


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--bits", type=int, default=8, choices=(8,))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--max-diff", type=float, default=0.08)
    ap.add_argument("--min-cosine", type=float, default=0.995)
    args = ap.parse_args()

    ref = run(args.binary, args.model_dir, TEXTS, args.batch_size, 0,
              args.group_size)
    got = run(args.binary, args.model_dir, TEXTS, args.batch_size, args.bits,
              args.group_size)

    worst_diff = 0.0
    worst_cos = 1.0
    for i, (a, b) in enumerate(zip(ref, got)):
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        cos = cosine(a, b)
        worst_diff = max(worst_diff, max_diff)
        worst_cos = min(worst_cos, cos)
        print(f"[{i}] cosine={cos:.8f} max_abs_diff={max_diff:.8g}")

    print(
        f"ok: bits={args.bits} group_size={args.group_size} "
        f"worst_diff={worst_diff:.8g} worst_cos={worst_cos:.8f}"
    )
    if worst_diff > args.max_diff or worst_cos < args.min_cosine:
        raise SystemExit(
            f"quantized parity failed: diff={worst_diff:g}, cos={worst_cos:g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
