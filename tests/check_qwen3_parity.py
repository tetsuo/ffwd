#!/usr/bin/env python3
"""Compare Qwen3-Embedding output with SentenceTransformers.

The CLI runs the real product path: BPE tokenize -> qwen3 forward (causal
attention, last-token pooling) -> L2 normalize. The reference is
SentenceTransformer(dir).encode(). Needs the model weights to run the forward,
so it stays a manual --model-dir check. The reference side needs torch +
sentence-transformers, so run it with uv:

  uv run --python 3.12 --with-requirements requirements-reference.txt \
      tests/check_qwen3_parity.py --model-dir DIR

The default texts carry the Instruct/Query prompt, which holds embedded
newlines, so they are passed as CLI arguments (one text per arg) rather than
through --stream, which reads one text per stdin line. The CLI prints the
cosine matrix, a blank line, then one raw-embedding row per text (with -e), and
the per-text token counts on stderr (with -v).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "ffwd-cli"

DEFAULT_TEXTS = [
    (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: What is the capital of China?"
    ),
    (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery: Explain gravity"
    ),
    "The capital of China is Beijing.",
    (
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects."
    ),
]


def ensure_cli(binary: Path) -> None:
    """Build the CLI if the default binary is missing."""
    if binary == DEFAULT_BINARY and not binary.exists():
        subprocess.run(["make", "cpu"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def local_embeddings(
    binary: str,
    model_dir: Path,
    batch_size: int,
    texts: list[str],
) -> tuple[np.ndarray, list[int]]:
    cmd = [binary, "-d", str(model_dir), "-e", "-v", "-b", str(batch_size), *texts]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"local command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")

    # -e prints the cosine matrix, a blank line, then one raw-embedding row per
    # text. The embedding rows are the last len(texts) non-empty stdout lines.
    nonempty = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(nonempty) < 2 * len(texts):
        raise RuntimeError(f"local output did not contain {len(texts)} embedding rows")
    emb_lines = nonempty[-len(texts):]
    rows = [[float(x) for x in ln.split()] for ln in emb_lines]

    token_counts = [int(x) for x in re.findall(r"tokens \((\d+)\):", proc.stderr)]
    if len(rows) != len(texts) or len(token_counts) != len(texts):
        raise RuntimeError(
            f"expected {len(texts)} rows/tokens, got {len(rows)}/{len(token_counts)}")
    return np.asarray(rows, dtype=np.float32), token_counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-diff", type=float, default=0.01)
    ap.add_argument("--min-cosine", type=float, default=0.9999)
    ap.add_argument("texts", nargs="*")
    args = ap.parse_args()

    texts = args.texts or DEFAULT_TEXTS
    reference = SentenceTransformer(str(args.model_dir), local_files_only=True)
    reference[0].auto_model.to(dtype=torch.float32)
    expected = np.asarray(
        reference.encode(texts, batch_size=args.batch_size), dtype=np.float32)

    ensure_cli(Path(args.binary))
    actual, token_counts = local_embeddings(
        args.binary, args.model_dir, args.batch_size, texts)
    hf_tokens = reference.tokenizer(texts, padding=False, truncation=True)["input_ids"]

    ok = actual.shape == expected.shape
    worst_diff = 0.0
    worst_cosine = 1.0
    for i, (got, want) in enumerate(zip(actual, expected)):
        diff = float(np.max(np.abs(got - want)))
        score = cosine(got, want)
        tokens_ok = token_counts[i] == len(hf_tokens[i])
        passed = tokens_ok and diff <= args.max_diff and score >= args.min_cosine
        ok = ok and passed
        worst_diff = max(worst_diff, diff)
        worst_cosine = min(worst_cosine, score)
        print(
            f"{'ok' if passed else 'FAIL'}: row={i} "
            f"tokens={token_counts[i]}/{len(hf_tokens[i])} "
            f"max_abs_diff={diff:.8g} cosine={score:.8f}")

    print(
        f"{'ok' if ok else 'FAIL'}: rows={len(texts)} "
        f"worst_diff={worst_diff:.8g} worst_cosine={worst_cosine:.8f}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
