#!/usr/bin/env python3
"""Benchmark /v1/rerank and optionally compare end-to-end with PyLate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any


BASE_DOCUMENTS = [
    "Scientists explore the universe driven by curiosity.",
    "Children learn through curious exploration.",
    "Historical discoveries began with careful questions.",
    "Redis stores data structures in memory for fast access.",
    "SQLite keeps relational data in a local database file.",
    "Embedding models map text into semantic vector spaces.",
    "Paris is the capital of France and a major European destination.",
    "Neural search compares query and document token representations.",
]
QUERY = "What motivates scientific discovery?"


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    rank = (len(values) - 1) * p / 100
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def make_documents(candidates: int, repeat: int) -> list[str]:
    return [
        " ".join(
            BASE_DOCUMENTS[(candidate + offset) % len(BASE_DOCUMENTS)]
            for offset in range(repeat)
        )
        for candidate in range(candidates)
    ]


def post_json(
    base_url: str,
    payload: dict[str, Any],
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/rerank",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text}") from exc


def summarize(name: str, values: list[float], candidates: int) -> None:
    mean = statistics.fmean(values)
    print(
        f"{name}: mean_ms={mean:.2f} p50_ms={percentile(values, 50):.2f} "
        f"p95_ms={percentile(values, 95):.2f} "
        f"candidates/s={candidates * 1000 / mean:.2f}"
    )


def synchronize(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()
    elif device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def pylate_run(model: Any, query: str, documents: list[str], device: str) -> list[float]:
    from pylate.scores import colbert_scores

    query_vectors = model.encode(
        [query],
        is_query=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )[0]
    document_vectors = model.encode(
        documents,
        is_query=False,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    scores = [
        float(
            colbert_scores(
                query_vectors.unsqueeze(0),
                document.unsqueeze(0),
            )[0, 0]
        )
        for document in document_vectors
    ]
    synchronize(device)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--model", default="pplx-embed-v1-late-0.6b")
    ap.add_argument("--model-dir")
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--document-repeat", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--compare-pylate", action="store_true")
    ap.add_argument("--pylate-device", default="mps")
    args = ap.parse_args()
    if min(args.candidates, args.document_repeat, args.runs) <= 0:
        raise SystemExit("candidates, document-repeat, and runs must be positive")
    if args.warmups < 0:
        raise SystemExit("warmups must be non-negative")
    if args.compare_pylate and not args.model_dir:
        raise SystemExit("--model-dir is required with --compare-pylate")

    documents = make_documents(args.candidates, args.document_repeat)
    payload = {
        "model": args.model,
        "query": QUERY,
        "documents": documents,
    }
    for _ in range(args.warmups):
        post_json(args.base_url, payload, args.api_key, args.timeout)
    server_times = []
    server_response: dict[str, Any] = {}
    for _ in range(args.runs):
        start = time.perf_counter()
        server_response = post_json(
            args.base_url, payload, args.api_key, args.timeout
        )
        server_times.append((time.perf_counter() - start) * 1000)
    summarize("pplx-embed-server", server_times, args.candidates)

    if args.compare_pylate:
        try:
            from pylate import models
        except ImportError as exc:
            raise SystemExit(
                "PyLate is missing; run with `uv run --with pylate "
                "bench/bench_late_api.py ...`"
            ) from exc

        model = models.ColBERT(
            model_name_or_path=args.model_dir,
            trust_remote_code=True,
            device=args.pylate_device,
        )
        for _ in range(args.warmups):
            pylate_run(model, QUERY, documents, args.pylate_device)
        pylate_times = []
        pylate_scores: list[float] = []
        for _ in range(args.runs):
            start = time.perf_counter()
            pylate_scores = pylate_run(
                model, QUERY, documents, args.pylate_device
            )
            pylate_times.append((time.perf_counter() - start) * 1000)
        summarize(f"PyLate/{args.pylate_device}", pylate_times, args.candidates)

        server_scores = [0.0] * args.candidates
        for result in server_response["results"]:
            server_scores[result["index"]] = float(result["relevance_score"])
        worst = max(abs(a - b) for a, b in zip(server_scores, pylate_scores))
        server_order = sorted(
            range(args.candidates), key=server_scores.__getitem__, reverse=True
        )
        pylate_order = sorted(
            range(args.candidates), key=pylate_scores.__getitem__, reverse=True
        )
        print(
            f"parity: worst_score_diff={worst:.8g} "
            f"top1_match={server_order[0] == pylate_order[0]}"
        )
        if worst > 1e-4 or server_order[0] != pylate_order[0]:
            raise RuntimeError("server and PyLate score parity failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
