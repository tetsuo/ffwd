#!/usr/bin/env python3
"""Concurrent benchmark for the local ffwd HTTP API."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import http.client
import json
import math
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from typing import Any


TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital of France.",
    "document: Berlin is the capital of Germany.",
    "document: Istanbul is a major city in Turkey.",
    "A small embedding request for throughput testing.",
    "Redis is an in-memory data structure server used for caching.",
    "SQLite stores relational data in a local file.",
    "Machine learning systems often use embeddings for semantic retrieval.",
]

_thread_local = threading.local()


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def decode_int8(s: str) -> list[float]:
    return [float(b - 256 if b > 127 else b) for b in base64.b64decode(s)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def make_input(request_id: int, texts_per_request: int,
               tokens_per_text: int = 0) -> list[str]:
    out = []
    for i in range(texts_per_request):
        if tokens_per_text > 0:
            # One common word per token, with a distinct prefix so batch entries
            # are not identical. Special tokens push the exact count slightly
            # above the target; throughput is reported from usage tokens anyway.
            word = TEXTS[(request_id + i) % len(TEXTS)].split()[0]
            out.append(" ".join([word] + ["hello"] * (tokens_per_text - 1)))
        else:
            out.append(TEXTS[(request_id + i) % len(TEXTS)])
    return out


def keepalive_connection(args: argparse.Namespace) -> tuple[http.client.HTTPConnection, str]:
    url = urllib.parse.urlparse(args.base_url)
    if url.scheme and url.scheme != "http":
        raise RuntimeError("--keepalive supports http URLs only")
    host = url.hostname or "127.0.0.1"
    port = url.port or 80
    prefix = url.path.rstrip("/")
    key = (host, port, prefix)
    cached = getattr(_thread_local, "conn", None)
    cached_key = getattr(_thread_local, "key", None)
    if cached is None or cached_key != key:
        if cached is not None:
            cached.close()
        cached = http.client.HTTPConnection(host, port, timeout=args.timeout)
        _thread_local.conn = cached
        _thread_local.key = key
    return cached, prefix


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


def post_embeddings(args: argparse.Namespace, request_id: int,
                    texts: list[str] | None = None) -> tuple[float, int, int, dict[str, float]]:
    if texts is None:
        texts = make_input(request_id, args.texts_per_request,
                           getattr(args, "tokens_per_text", 0))
    payload = {
        "model": args.model,
        "dimensions": args.dimensions,
        "input": texts,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    t0 = time.perf_counter()
    if args.keepalive:
        conn, prefix = keepalive_connection(args)
        try:
            conn.request("POST", prefix + "/v1/embeddings", body, headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status != 200:
                raise RuntimeError(
                    f"HTTP {resp.status}: {raw.decode('utf-8', errors='replace')}"
                )
            if (resp.getheader("Connection") or "").lower() == "close":
                raise RuntimeError("server closed keep-alive benchmark connection")
            timing = parse_server_timing(resp.getheader("Server-Timing"))
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            conn.close()
            _thread_local.conn = None
            raise
    else:
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/embeddings",
            data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                timing = parse_server_timing(resp.headers.get("Server-Timing"))
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {text}") from e
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    rows = data["data"]
    if len(rows) != len(texts):
        raise RuntimeError(f"expected {len(texts)} embeddings, got {len(rows)}")
    for i, row in enumerate(rows):
        if row["index"] != i:
            raise RuntimeError(f"bad index at request {request_id}: {row['index']} != {i}")
        emb = row["embedding"]
        if isinstance(emb, list):
            # OpenAI-family float output.
            if len(emb) != args.dimensions:
                raise RuntimeError(f"bad embedding length: {len(emb)} != {args.dimensions}")
        else:
            # Base64: int8 (1 byte/dim, pplx) or float32 (4 bytes/dim, OpenAI).
            raw = base64.b64decode(emb)
            if len(raw) not in (args.dimensions, 4 * args.dimensions):
                raise RuntimeError(f"bad decoded length: {len(raw)} != {args.dimensions}")

    return elapsed_ms, len(texts), int(data["usage"]["prompt_tokens"]), timing


def verify_paris_ranking(args: argparse.Namespace) -> bool:
    texts = [
        "query: what is the capital of France?",
        "document: Paris is the capital of France.",
        "document: Berlin is the capital of Germany.",
    ]
    body = json.dumps({
        "model": args.model,
        "dimensions": args.dimensions,
        "input": texts,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/embeddings",
                                 data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = data["data"]
    if len(rows) != len(texts):
        raise RuntimeError(f"ranking probe expected {len(texts)} embeddings, got {len(rows)}")
    vecs = [decode_int8(row["embedding"]) for row in data["data"]]
    return cosine(vecs[0], vecs[1]) > cosine(vecs[0], vecs[2])


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
    ap.add_argument("--model", default="pplx-embed-v1-0.6b")
    ap.add_argument("--dimensions", type=int, default=128)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--texts-per-request", type=int, default=1)
    ap.add_argument("--tokens-per-text", type=int, default=0,
                    help="synthesize inputs of about this many tokens instead of the stock texts")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--verify-ranking", action="store_true")
    ap.add_argument("--keepalive", action="store_true",
                    help="reuse one HTTP/1.1 connection per benchmark worker thread")
    args = ap.parse_args()

    if args.requests <= 0 or args.concurrency <= 0 or args.texts_per_request <= 0:
        raise SystemExit("requests, concurrency, and texts-per-request must be positive")

    t0 = time.perf_counter()
    latencies: list[float] = []
    timing_rows: list[dict[str, float]] = []
    wall_minus_worker: list[float] = []
    total_texts = 0
    total_tokens = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(post_embeddings, args, i) for i in range(args.requests)]
        for fut in cf.as_completed(futs):
            ms, texts, tokens, timing = fut.result()
            latencies.append(ms)
            if timing:
                timing_rows.append(timing)
                if "worker" in timing:
                    wall_minus_worker.append(ms - timing["worker"])
            total_texts += texts
            total_tokens += tokens
    wall = time.perf_counter() - t0
    ranking_ok = verify_paris_ranking(args) if args.verify_ranking else True

    print(f"server: {args.base_url} model={args.model} dim={args.dimensions}")
    mode = "keepalive" if args.keepalive else "native-http"
    print(f"requests={args.requests} concurrency={args.concurrency} texts/request={args.texts_per_request} mode={mode}")
    print(f"wall={wall:.3f}s req/s={args.requests / wall:.2f} texts/s={total_texts / wall:.2f} tokens/s={total_tokens / wall:.2f}")
    print(
        "latency_ms "
        f"mean={statistics.fmean(latencies):.2f} "
        f"p50={percentile(latencies, 50):.2f} "
        f"p95={percentile(latencies, 95):.2f} "
        f"p99={percentile(latencies, 99):.2f} "
        f"max={max(latencies):.2f}"
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
    if args.verify_ranking:
        print(f"ranking={'ok' if ranking_ok else 'FAIL'}")
    return 0 if ranking_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
