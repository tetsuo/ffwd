#!/usr/bin/env python3
"""Compare two benchmark records produced by make bench or make bench-model
(JSON written by bench/bench.h --json).

usage: benchstat.py OLD.json NEW.json [--alpha 0.05] [--threshold 3.0]

Per benchmark:
median ns/op for both sample sets, the delta, and a Mann-Whitney U
significance test.

A delta counts only when it is both:
- significant at --alpha
- at least --threshold percent

Back-to-back runs of an unchanged binary can drift a few percent with machine
state, such as thermals and core scheduling. Smaller shifts are below the
reproducibility floor even when statistically clean.

Everything else prints as "~"; treat it as noise, not improvement or regression.

Exits 1 if any benchmark regresses past both bars.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {b["name"]: b["ns_per_op"] for b in data["benchmarks"]}


def median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mann_whitney_p(a: list[float], b: list[float]) -> float:
    """Two-sided p-value via the normal approximation with tie correction.
    Fine for the sample counts bench.h produces (default 10 per side)."""
    n1, n2 = len(a), len(b)
    pooled = sorted((x, 0) for x in a) + sorted((x, 1) for x in b)
    pooled.sort(key=lambda t: t[0])

    ranks: list[float] = [0.0] * len(pooled)
    tie_term = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        rank = (i + j + 1) / 2.0  # average rank for the tie group (1-based)
        for k in range(i, j):
            ranks[k] = rank
        t = j - i
        tie_term += t * t * t - t
        i = j

    r1 = sum(r for r, (_, grp) in zip(ranks, pooled) if grp == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma_sq = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma_sq <= 0:
        return 1.0  # all values identical
    z = (u1 - mu - (0.5 if u1 > mu else -0.5 if u1 < mu else 0.0)) \
        / math.sqrt(sigma_sq)
    return math.erfc(abs(z) / math.sqrt(2.0))


def fmt_ns(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.1f}ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f}µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f}ms"
    return f"{ns / 1e9:.3f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="minimum |delta| percent to report (default 3)")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    common = [k for k in old if k in new]
    if not common:
        print("no common benchmarks", file=sys.stderr)
        return 2

    width = max(len(k) for k in common)
    print(f"{'benchmark':<{width}}  {'old':>10}  {'new':>10}  delta")
    regressions = []
    for name in common:
        mo, mn = median(old[name]), median(new[name])
        p = mann_whitney_p(old[name], new[name])
        delta = (mn - mo) / mo * 100.0
        if p < args.alpha and abs(delta) >= args.threshold:
            mark = f"{delta:+.1f}% (p={p:.3f})"
            if delta > 0:
                regressions.append((name, delta))
        else:
            mark = "~"
        print(f"{name:<{width}}  {fmt_ns(mo):>10}  {fmt_ns(mn):>10}  {mark}")

    for k in old:
        if k not in new:
            print(f"only in {Path(args.old).name}: {k}")
    for k in new:
        if k not in old:
            print(f"only in {Path(args.new).name}: {k}")

    if regressions:
        worst = max(regressions, key=lambda r: r[1])
        print(f"\n{len(regressions)} significant regression(s); "
              f"worst: {worst[0]} {worst[1]:+.1f}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
