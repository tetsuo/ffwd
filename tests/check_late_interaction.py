#!/usr/bin/env python3
"""Smoke-check late-interaction token vectors and MaxSim ranking.

The C side is tests/late_check.c, built by the Makefile target
`late-check-driver`. Plain mode runs the driver against a real late model and
checks the expected ranking; it needs only python3 and a late model directory
(e.g. pplx-embed-v1-late-0.6b). With --compare-pylate it also compares the C
token vectors against PyLate, which needs the pylate package:

  tests/check_late_interaction.py --model-dir DIR
  uv run --python 3.12 --with pylate tests/check_late_interaction.py \
      --model-dir DIR --compare-pylate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_driver(cc: str, backend: str) -> Path:
    """Build the late-interaction driver via the Makefile; return its path."""
    target = "late-check-driver-mlx" if backend == "mlx" else "late-check-driver"
    binary = "late_check_mlx" if backend == "mlx" else "late_check"
    proc = subprocess.run(["make", "-C", "tests", f"CC={cc}", target], cwd=ROOT,
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        raise SystemExit(proc.returncode)
    return ROOT / "tests" / binary


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def compare_with_pylate(model_dir: str, c_json: dict,
                        max_score_diff: float, min_vector_cosine: float) -> None:
    try:
        from pylate import models
        from pylate.scores import colbert_scores
    except ImportError as e:
        raise SystemExit(
            "missing PyLate dependency; run with: uv run --python 3.12 --with pylate "
            "tests/check_late_interaction.py --compare-pylate ..."
        ) from e

    query = "What motivates scientific discovery?"
    docs = [
        "Scientists explore the universe driven by curiosity.",
        "Children learn through curious exploration.",
        "Historical discoveries began with curious questions.",
    ]

    model = models.ColBERT(
        model_name_or_path=model_dir,
        trust_remote_code=True,
        device="cpu",
    )
    q = model.encode([query], is_query=True, convert_to_tensor=True,
                     show_progress_bar=False)[0].detach().cpu()
    d = [
        x.detach().cpu()
        for x in model.encode(docs, is_query=False, convert_to_tensor=True,
                              show_progress_bar=False)
    ]

    c_query = c_json["query"]
    if len(c_query) != q.shape[0] or len(c_query[0]) != q.shape[1]:
        raise RuntimeError(
            f"query shape mismatch: C={len(c_query)}x{len(c_query[0])} "
            f"PyLate={tuple(q.shape)}"
        )

    worst_cos = 1.0
    for i, row in enumerate(c_query):
        worst_cos = min(worst_cos, cosine(row, q[i].tolist()))

    py_scores = []
    for i, py_doc in enumerate(d):
        c_doc = c_json["documents"][i]["vectors"]
        if len(c_doc) != py_doc.shape[0] or len(c_doc[0]) != py_doc.shape[1]:
            raise RuntimeError(
                f"doc{i} shape mismatch: C={len(c_doc)}x{len(c_doc[0])} "
                f"PyLate={tuple(py_doc.shape)}"
            )
        for j, row in enumerate(c_doc):
            worst_cos = min(worst_cos, cosine(row, py_doc[j].tolist()))
        py_scores.append(float(colbert_scores(q.unsqueeze(0),
                                             py_doc.unsqueeze(0))[0, 0]))

    c_scores = [float(x) for x in c_json["scores"]]
    worst_score_diff = max(abs(a - b) for a, b in zip(c_scores, py_scores))
    py_best = max(range(len(py_scores)), key=lambda i: py_scores[i])
    if c_json["best"] != py_best:
        raise RuntimeError(f"best mismatch: C={c_json['best']} PyLate={py_best}")
    if worst_score_diff > max_score_diff or worst_cos < min_vector_cosine:
        raise RuntimeError(
            f"late PyLate parity failed: worst_score_diff={worst_score_diff:g} "
            f"worst_vector_cosine={worst_cos:g}"
        )

    print(
        "pylate parity: "
        f"worst_score_diff={worst_score_diff:.8g} "
        f"worst_vector_cosine={worst_cos:.8f} best=doc{py_best}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--backend", choices=("cpu", "mlx"), default="cpu")
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    ap.add_argument("--compare-pylate", action="store_true",
                    help="compare all retained C token vectors against PyLate")
    ap.add_argument("--max-score-diff", type=float, default=1e-4)
    ap.add_argument("--min-vector-cosine", type=float, default=0.99999)
    ap.add_argument("--runs", type=int, default=0,
                    help="run N timed packed MaxSim reranking loops")
    args = ap.parse_args()
    binary = build_driver(args.cc, args.backend)

    if args.compare_pylate:
        proc = subprocess.run(
            [str(binary), args.model_dir, "--json"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            return proc.returncode
        c_json = json.loads(proc.stdout)
        compare_with_pylate(args.model_dir, c_json, args.max_score_diff,
                            args.min_vector_cosine)
        return 0
    cmd = [str(binary), args.model_dir]
    if args.runs:
        cmd += ["--runs", str(args.runs)]
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
