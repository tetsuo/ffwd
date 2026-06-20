#!/usr/bin/env python3
"""Rank stdin JSON embeddings by cosine similarity to a query row.

Usage:
  ./tools/ffwd-cli ... --stream <<'EOF' | ./devtools/rank_embeddings.py
  query: what is the capital of France?
  document: Paris is the capital of France.
  document: Berlin is the capital of Germany.
  EOF
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    aa = 0.0
    bb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        aa += x * x
        bb += y * y
    if aa == 0.0 or bb == 0.0:
        return float("nan")
    return dot / math.sqrt(aa * bb)


def read_rows() -> list[dict]:
    rows = []
    for lineno, line in enumerate(sys.stdin, 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"line {lineno}: invalid JSON: {exc}") from exc
        if "error" in row:
            raise SystemExit(f"line {lineno}: stdin error: {row['error']}")
        emb = row.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise SystemExit(f"line {lineno}: missing embedding array")
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-index", type=int, default=0,
                    help="embedding row to use as query, default: 0")
    ap.add_argument("--include-query", action="store_true",
                    help="include the query row in the ranked output")
    ap.add_argument("--labels", nargs="*",
                    help="optional labels for rows, in input order")
    args = ap.parse_args()

    rows = read_rows()
    if not rows:
        raise SystemExit("no embeddings on stdin")
    if args.query_index < 0 or args.query_index >= len(rows):
        raise SystemExit(f"--query-index out of range for {len(rows)} rows")
    if args.labels and len(args.labels) != len(rows):
        raise SystemExit("--labels count must match embedding row count")

    query = rows[args.query_index]["embedding"]
    ranked = []
    for i, row in enumerate(rows):
        if i == args.query_index and not args.include_query:
            continue
        emb = row["embedding"]
        if len(emb) != len(query):
            raise SystemExit(f"row {i}: dimension mismatch")
        ranked.append((cosine(query, emb), i, row))

    ranked.sort(key=lambda item: item[0], reverse=True)

    print(f"query row: {args.query_index}")
    print("rank  row  cosine      tokens  label")
    print("----  ---  ----------  ------  -----")
    for rank, (score, i, row) in enumerate(ranked, 1):
        label = args.labels[i] if args.labels else f"row {i}"
        tokens = row.get("tokens", "-")
        print(f"{rank:>4}  {i:>3}  {score:>10.6f}  {tokens:>6}  {label}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
