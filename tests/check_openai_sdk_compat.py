#!/usr/bin/env python3
"""Verify the local server through the official OpenAI Python SDK.

The server's /v1/embeddings is OpenAI-shaped, so an OpenAI client should work
against it unchanged. This builds and launches ffwd-server with a model, then
drives client.embeddings.create through the OpenAI SDK, checking the response
shape and a retrieval ranking. Pass --base-url to test a running server instead.

The OpenAI SDK is not in requirements-reference.txt, so run with uv:

  uv run --python 3.12 --with openai \
      tests/check_openai_sdk_compat.py --model-dir DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit(
        "missing dependency; run with: uv run --python 3.12 --with openai "
        "tests/check_openai_sdk_compat.py ..."
    ) from e

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
SERVER_BIN = ROOT / "ffwd-server"

DOCS = [
    ("Capitals", "Paris is the capital of France."),
    ("Capitals", "Berlin is the capital of Germany."),
    ("Science", "Photosynthesis converts sunlight into chemical energy in plants."),
    ("Databases", "Redis is an in-memory data structure server used for caching."),
]
QUERY = "What is the capital of France?"


def ensure_server() -> None:
    if not SERVER_BIN.exists():
        subprocess.run(["make", "cpu"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    if not SERVER_BIN.exists():
        sys.exit(f"ffwd-server not found and could not be built: {SERVER_BIN}")


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def die_with_log(proc: subprocess.Popen, log_path: str, msg: str) -> None:
    try:
        with open(log_path) as f:
            tail = "".join(f.readlines()[-20:])
    except OSError:
        tail = "(no server log)"
    try:
        proc.kill()
    except OSError:
        pass
    sys.exit(f"{msg}\n--- server log tail ---\n{tail}")


def wait_ready(base_url: str, api_key: str, model: str,
               proc: subprocess.Popen, log_path: str, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    body = json.dumps({"model": model, "input": "ready?",
                       "encoding_format": "float"}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/embeddings", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(proc, log_path, "server exited during startup")
        try:
            with urllib.request.urlopen(req, timeout=5):
                return
        except (OSError, urllib.error.HTTPError):
            pass
        time.sleep(0.2)
    die_with_log(proc, log_path, "server did not become ready in time")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed(client: OpenAI, model: str, texts: list[str],
          dimensions: int) -> list[list[float]]:
    resp = client.embeddings.create(model=model, input=texts,
                                    encoding_format="float", dimensions=dimensions)
    if len(resp.data) != len(texts):
        raise RuntimeError(f"got {len(resp.data)} rows, expected {len(texts)}")
    rows = []
    for item in sorted(resp.data, key=lambda d: d.index):
        if not isinstance(item.embedding, list):
            raise RuntimeError("embedding is not a float array")
        rows.append([float(x) for x in item.embedding])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=None,
                    help="test a running server instead of launching one")
    ap.add_argument("--model-dir", default=None, help="model dir (for launch)")
    ap.add_argument("--model", default="text-embedding-local")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--dimensions", type=int, default=128)
    args = ap.parse_args()

    proc: subprocess.Popen | None = None
    log_path: str | None = None
    base_url = args.base_url

    if base_url is None:
        if not args.model_dir:
            raise SystemExit("--model-dir is required when launching the server")
        ensure_server()
        port = free_port()
        base_url = f"http://{HOST}:{port}"
        argv = [str(SERVER_BIN), "--host", HOST, "--port", str(port),
                "--api-key", args.api_key, "-b", "2",
                "--model", f"{args.model}={args.model_dir}:api=perplexity"
                           f":min_dim={args.dimensions}"]
        log = open(tempfile.mktemp(prefix="ffwd-openai-sdk-", suffix=".log"), "w")
        log_path = log.name
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        wait_ready(base_url, args.api_key, args.model, proc, log_path)

    try:
        client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=args.api_key)

        print("standard OpenAI SDK call:")
        doc_vecs = embed(client, args.model, [t for _, t in DOCS], args.dimensions)
        if any(len(v) != args.dimensions for v in doc_vecs):
            raise RuntimeError("decoded dimension mismatch")
        q_vec = embed(client, args.model, [QUERY], args.dimensions)[0]

        ranked = sorted(
            ({"title": title, "text": text, "score": cosine(q_vec, vec)}
             for (title, text), vec in zip(DOCS, doc_vecs)),
            key=lambda r: r["score"], reverse=True)

        print(f"  dimensions={args.dimensions} rows={len(doc_vecs)}")
        for row in ranked[:3]:
            print(f"  {row['score']: .4f}  [{row['title']}] {row['text']}")
        ok = ranked[0]["text"] == "Paris is the capital of France."
        print(f"  verdict={'ok' if ok else 'FAIL'}: expected the Paris document on top")
        print(f"overall={'ok' if ok else 'FAIL'}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if log_path:
                try:
                    os.unlink(log_path)
                except OSError:
                    pass

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
