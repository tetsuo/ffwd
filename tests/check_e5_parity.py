#!/usr/bin/env python3
"""Compare the ffwd CLI against SentenceTransformers for an XLM-R /
SentencePiece model (default: intfloat/multilingual-e5-large-instruct).

The CLI runs the real product path: SentencePiece tokenize -> <s> ... </s> ->
xlm-roberta forward (position offset pad_id+1) -> config-driven mean pool -> L2
normalize. The reference is SentenceTransformer(dir).encode(normalize=True),
which runs the same pipeline. Cosine is scale-invariant; max_abs also checks
magnitude, so it only matches if both sides normalize.

Needs the model weights to run the forward, so it stays a manual --model-dir
check. The reference side needs torch + sentence-transformers, so run it with uv:

  uv run --python 3.12 --with-requirements requirements-reference.txt \
      tests/check_e5_parity.py --model-dir DIR
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "ffwd-cli"

DEFAULT_TEXTS = [
    "The capital of France is Paris.",
    "Dense retrieval maps text to a vector space for semantic search.",
    "Numbers like 12345 and dates 2020-2021 appear here.",
    "café résumé naïve coordinate punctuation, test!",
    "南瓜是一种蔬菜。",
    "Привет мир, как дела сегодня?",
    "مرحبا بالعالم هذا اختبار.",
    "a short query",
]


def ensure_cli(binary: Path) -> None:
    """Build the CLI if the default binary is missing."""
    if binary == DEFAULT_BINARY and not binary.exists():
        subprocess.run(["make", "cpu"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)


def cli_encode(binary, model_dir, texts):
    cmd = [str(binary), "-d", str(model_dir), "--stream"]
    proc = subprocess.run(cmd, input="\n".join(texts) + "\n", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(proc.returncode)
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "error" in obj:
            raise SystemExit(f"CLI error: {obj}")
        rows.append([float(x) for x in obj["embedding"]])
    return rows


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-large-instruct")
    ap.add_argument("--model-dir", default=None, help="local dir; default downloads --model")
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--max-diff", type=float, default=2e-3)
    ap.add_argument("texts", nargs="*")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    if args.model_dir:
        model_dir = args.model_dir
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model)

    texts = list(args.texts if args.texts else DEFAULT_TEXTS)
    ref = SentenceTransformer(model_dir)
    ref_vecs = ref.encode(texts, normalize_embeddings=True).tolist()

    ensure_cli(Path(args.binary))
    c_vecs = cli_encode(args.binary, model_dir, texts)

    if len(c_vecs) != len(ref_vecs):
        print(f"row count mismatch: C={len(c_vecs)} ref={len(ref_vecs)}", file=sys.stderr)
        return 1

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
        bad = cos < args.min_cos or diff > args.max_diff
        fails += bad
        print(f"[{i}] cos={cos:.8f} max_abs={diff:.3e}{'  <-- FAIL' if bad else ''}  {t[:42]!r}")

    print(f"worst_cos={worst_cos:.8f} worst_diff={worst_diff:.3e} dim={len(c_vecs[0])}")
    if fails:
        print(f"failed: {fails}/{len(texts)} e5 parity cases", file=sys.stderr)
        return 1
    print(f"ok: e5/XLM-R parity for {len(texts)} cases ({args.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
