#!/usr/bin/env python3
"""Check the BPE tokenizer against stored reference vectors.

The reference ids live in tests/test-vectors/bpe/expected.json, generated from
the Qwen tokenizer by devtools/gen_test_vectors.py.

This builds the C BPE driver (tests/tokenizer_dump), runs it against the checked-in
vocab.json + merges.txt, and compares the ids exactly.

  tests/check_tokenizer_parity.py
  tests/check_tokenizer_parity.py --cc clang
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTORS = ROOT / "tests/test-vectors/bpe"
DRIVER = ROOT / "tests/tokenizer_dump"


def build_driver(cc: str) -> None:
    """Build the BPE driver via the Makefile so a C change is picked up."""
    proc = subprocess.run(["make", "-C", "tests", f"CC={cc}", "parity-tokenizer-driver"],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)


def c_encode(vocab: Path, texts: list[str]) -> list[list[int]]:
    proc = subprocess.run([str(DRIVER), str(vocab), *texts], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    rows: list[list[int]] = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if not fields:
            rows.append([])
            continue
        n = int(fields[0])
        ids = [int(x) for x in fields[1:]]
        if len(ids) != n:
            raise SystemExit(f"C tokenizer emitted {len(ids)} ids, header says {n}")
        rows.append(ids)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--vectors", default=str(DEFAULT_VECTORS),
                    help="dir holding expected.json + vocab.json + merges.txt")
    args = ap.parse_args()

    vectors = Path(args.vectors)
    data = json.loads((vectors / "expected.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    texts = [c["text"] for c in cases]
    expected = [c["ids"] for c in cases]

    build_driver(args.cc)
    c_rows = c_encode(vectors / "vocab.json", texts)
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
        print(f"failed: {failures}/{len(texts)} BPE tokenizer cases mismatched", file=sys.stderr)
        return 1
    print(f"ok: BPE tokenizer parity for {len(texts)} cases ({data['model']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
