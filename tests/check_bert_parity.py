#!/usr/bin/env python3
"""Compare the BERT-family forward against SentenceTransformers.

Builds a small C driver that loads the WordPiece tokenizer and the BERT model
from a sentence-transformers checkpoint, wraps each text with [CLS]/[SEP],
encodes it (BERT forward + mean pool + L2 normalize), and prints the vector.
The reference is SentenceTransformer(model).encode(normalize_embeddings=True),
which runs the same pipeline. Defaults to all-MiniLM-L6-v2.

Needs the real model weights to run the C forward.

  uv run --python 3.12 --with-requirements requirements-reference.txt \
      tests/check_bert_parity.py --model-dir DIR
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests/bert_dump"

DEFAULT_TEXTS = [
    "hello world",
    "The capital of France is Paris.",
    "Dense retrieval systems need stable tokenization.",
    "WordPiece splits unaffable into sub-words.",
    "Numbers like 12345 and dates 2020-2021 appear here.",
    "café résumé naïve coordinate punctuation, test!",
    "a short query",
    "Sentence embeddings map text to a vector space for semantic search.",
]


def build_driver(cc: str) -> Path:
    """Build the BERT parity driver via the Makefile; return the binary path."""
    proc = subprocess.run(["make", "-C", "tests", f"CC={cc}", "parity-bert-driver"],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)
    return DRIVER


def c_encode(driver: Path, model_dir: str, texts: list[str]) -> list[list[float]]:
    proc = subprocess.run([str(driver), model_dir, *texts], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(proc.returncode)
    rows = []
    for line in proc.stdout.splitlines():
        f = line.split()
        rows.append([float(x) for x in f[1:]])
    return rows


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--model-dir", default=None, help="local dir; default downloads --model")
    ap.add_argument("--c-model-dir", default=None,
                    help="dir the C side loads; defaults to the reference dir. Set to a "
                         "BF16-converted copy to gate the BF16 path against the F32 reference.")
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--min-cos", type=float, default=0.9999)
    ap.add_argument("--max-diff", type=float, default=1e-3)
    ap.add_argument("texts", nargs="*")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    if args.model_dir:
        model_dir = args.model_dir
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model)

    c_model_dir = args.c_model_dir or model_dir

    texts = list(args.texts if args.texts else DEFAULT_TEXTS)
    ref = SentenceTransformer(model_dir)
    ref_vecs = ref.encode(texts, normalize_embeddings=True).tolist()

    driver = build_driver(args.cc)
    c_vecs = c_encode(driver, c_model_dir, texts)

    worst_cos, worst_diff, fails = 1.0, 0.0, 0
    for i, (t, c, r) in enumerate(zip(texts, c_vecs, ref_vecs)):
        if len(c) != len(r):
            print(f"dim mismatch {i}: {len(c)} vs {len(r)}", file=sys.stderr)
            fails += 1
            continue
        cos = cosine(c, r)
        diff = max(abs(x - y) for x, y in zip(c, r))
        worst_cos = min(worst_cos, cos)
        worst_diff = max(worst_diff, diff)
        flag = "" if (cos >= args.min_cos and diff <= args.max_diff) else "  <-- FAIL"
        if flag:
            fails += 1
        print(f"[{i}] cos={cos:.8f} max_abs={diff:.3e}{flag}  {t[:48]!r}")

    print(f"worst_cos={worst_cos:.8f} worst_diff={worst_diff:.3e}")
    if fails:
        print(f"failed: {fails}/{len(texts)} BERT parity cases", file=sys.stderr)
        return 1
    print(f"ok: BERT parity for {len(texts)} cases ({args.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
