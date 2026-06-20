#!/usr/bin/env python3
"""Benchmark contextual span inference through the local HTTP API."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import statistics
import time
import urllib.error
import urllib.request


CHUNK_TEXT = (
    "Contextual embedding inference processes complete documents before pooling "
    "individual chunk spans. This benchmark keeps the input deterministic while "
    "stressing long bidirectional attention and compact chunk result handling. "
)


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def parse_server_timing(value: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not value:
        return out
    for part in value.split(","):
        fields = [x.strip() for x in part.strip().split(";")]
        if not fields or not fields[0]:
            continue
        name = fields[0]
        for field in fields[1:]:
            if field.startswith("dur="):
                try:
                    out[name] = float(field[4:])
                except ValueError:
                    pass
    return out


def chunk_count(args: argparse.Namespace, doc_index: int) -> int:
    return args.chunks * (doc_index + 1) if args.ragged else args.chunks


def total_chunks_per_request(args: argparse.Namespace) -> int:
    return sum(chunk_count(args, di) for di in range(args.documents))


def make_payload(args: argparse.Namespace, request_id: int) -> dict:
    docs = []
    for di in range(args.documents):
        chunks = [
            f"request {request_id} document {di} chunk {ci}: "
            + CHUNK_TEXT * args.repeats_per_chunk
            for ci in range(chunk_count(args, di))
        ]
        docs.append(chunks)
    return {
        "model": args.model,
        "dimensions": args.dimensions,
        "input": docs,
    }


def post_json(args: argparse.Namespace, request_id: int) -> tuple[float, int, dict[str, float]]:
    payload = make_payload(args, request_id)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/contextualizedembeddings",
        data=body,
        headers=headers,
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            timing = parse_server_timing(resp.headers.get("Server-Timing"))
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {text}") from e
    rows = data["data"]
    if len(rows) != args.documents:
        raise RuntimeError("unexpected contextual document count")
    for di, row in enumerate(rows):
        expected_chunks = chunk_count(args, di)
        if row["index"] != di or len(row["data"]) != expected_chunks:
            raise RuntimeError("unexpected contextual response shape")
        for ci, chunk in enumerate(row["data"]):
            if chunk["index"] != ci:
                raise RuntimeError("unexpected contextual chunk index")
    return (time.perf_counter() - t0) * 1000.0, int(data["usage"]["prompt_tokens"]), timing


def rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        kb = int(out.strip())
        return kb / 1024.0
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--model", default="pplx-embed-context-v1-0.6b")
    ap.add_argument("--dimensions", type=int, default=128)
    ap.add_argument("--documents", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--repeats-per-chunk", type=int, default=8)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--ragged", action="store_true",
                    help="scale chunk count by document index to benchmark padded contextual batches")
    args = ap.parse_args()
    if min(args.dimensions, args.documents, args.chunks, args.repeats_per_chunk,
           args.runs, args.concurrency) <= 0:
        raise SystemExit(
            "dimensions, documents, chunks, repeats-per-chunk, runs, and "
            "concurrency must be positive"
        )
    if args.warmups < 0:
        raise SystemExit("warmups must be >= 0")

    for i in range(args.warmups):
        post_json(args, -i - 1)

    latencies = []
    timing_rows: list[dict[str, float]] = []
    wall_minus_worker: list[float] = []
    total_tokens = 0
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(post_json, args, i) for i in range(args.runs)]
        for future in cf.as_completed(futures):
            elapsed, tokens, timing = future.result()
            latencies.append(elapsed)
            if timing:
                timing_rows.append(timing)
                if "worker" in timing:
                    wall_minus_worker.append(elapsed - timing["worker"])
            total_tokens += tokens
    wall = time.perf_counter() - t0
    p50 = percentile(latencies, 50)
    chunks_per_request = total_chunks_per_request(args)
    print(
        f"server={args.base_url} model={args.model} documents={args.documents} "
        f"chunks/request={chunks_per_request} ragged={args.ragged} "
        f"runs={args.runs} concurrency={args.concurrency}"
    )
    print(
        "latency_ms "
        f"mean={statistics.fmean(latencies):.2f} "
        f"p50={p50:.2f} "
        f"p95={percentile(latencies, 95):.2f} "
        f"max={max(latencies):.2f}"
    )
    total_docs = args.runs * args.documents
    total_chunks = args.runs * chunks_per_request
    print(
        f"wall={wall:.3f}s req/s={args.runs / wall:.2f} "
        f"documents/s={total_docs / wall:.2f} chunks/s={total_chunks / wall:.2f} "
        f"tokens/s={total_tokens / wall:.2f}"
    )
    mem = rss_mb(args.server_pid)
    if mem is not None:
        print(f"rss_mb={mem:.1f}")
    if timing_rows:
        for name in ("queue", "parse", "tokenize", "infer", "encode", "worker"):
            values = [row[name] for row in timing_rows if name in row]
            if values:
                print(
                    f"server_timing_{name}_ms "
                    f"mean={statistics.fmean(values):.3f} "
                    f"p50={percentile(values, 50):.3f} "
                    f"p95={percentile(values, 95):.3f}"
                )
        if wall_minus_worker:
            print(
                "wall_minus_worker_ms "
                f"mean={statistics.fmean(wall_minus_worker):.3f} "
                f"p50={percentile(wall_minus_worker, 50):.3f} "
                f"p95={percentile(wall_minus_worker, 95):.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
