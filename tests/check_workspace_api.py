#!/usr/bin/env python3
"""Run the workspace / model API smoke test against real weights.

The C side is tests/test_workspace.c, the same source `make test` builds with
the synthetic fixture. Given a MODEL_DIR it loads real weights from that
directory instead and exercises the workspace and batch APIs. Needs only
python3 and a model directory (with vocab.json), no reference stack.

The model must be mean-pooled: the test checks that a whole-sequence span pools
to the embedding, which holds for mean pooling (pplx-embed, BERT/BGE) but not
for last-token pooling (Qwen3). Use a pplx-embed model.

  tests/check_workspace_api.py --model-dir DIR   # DIR = a pplx-embed model
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests/test_workspace"


def ensure_driver(cc: str) -> None:
    """Build tests/test_workspace via the Makefile so a C change is picked up."""
    proc = subprocess.run(["make", "-C", "tests", f"CC={cc}", "test_workspace"],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = ap.parse_args()
    ensure_driver(args.cc)
    return subprocess.run([str(DRIVER), args.model_dir], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
