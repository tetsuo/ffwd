#!/usr/bin/env python3
"""Check the C WordPiece tokenizer against stored reference vectors.

The reference ids live in tests/test-vectors/wordpiece/expected.json, generated
from the slow BertTokenizer by devtools/gen_test_vectors.py. This builds the C
WordPiece driver (tests/wordpiece_dump), runs it against the checked-in
vocab.txt + tokenizer_config.json, and compares the ids exactly. It imports no
transformers and downloads no model, so it is hermetic and fast.

  tests/check_wordpiece_parity.py
  tests/check_wordpiece_parity.py --cc clang
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS = ROOT / "tests/test-vectors/wordpiece"
DRIVER = ROOT / "tests/wordpiece_dump"


def build_driver(cc: str) -> None:
    """Build the WordPiece driver via the Makefile so a C change is picked up."""
    proc = subprocess.run(["make", "-C", "tests", f"CC={cc}", "parity-wordpiece-driver"],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)


def c_encode(model_dir: Path, texts: list[str]) -> list[list[int]]:
    proc = subprocess.run([str(DRIVER), str(model_dir), *texts], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    rows: list[list[int]] = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        rows.append([int(x) for x in fields[1:]] if fields else [])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--vectors", default=str(DEFAULT_VECTORS),
                    help="dir holding expected.json + vocab.txt + tokenizer_config.json")
    args = ap.parse_args()

    vectors = Path(args.vectors)
    data = json.loads((vectors / "expected.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    texts = [c["text"] for c in cases]
    expected = [c["ids"] for c in cases]

    build_driver(args.cc)
    c_rows = c_encode(vectors, texts)
    if len(c_rows) != len(texts):
        raise SystemExit(f"C emitted {len(c_rows)} rows for {len(texts)} texts")

    failures = 0
    for i, (text, c_ids, ref_ids) in enumerate(zip(texts, c_rows, expected)):
        if c_ids == ref_ids:
            continue
        failures += 1
        print(f"mismatch {i}: {text!r}", file=sys.stderr)
        print(f"  C  : {c_ids[:80]}{' ...' if len(c_ids) > 80 else ''}", file=sys.stderr)
        print(f"  ref: {ref_ids[:80]}{' ...' if len(ref_ids) > 80 else ''}", file=sys.stderr)

    if failures:
        print(f"failed: {failures}/{len(texts)} WordPiece cases mismatched", file=sys.stderr)
        return 1
    print(f"ok: WordPiece parity for {len(texts)} cases ({data['model']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
