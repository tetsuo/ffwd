#!/usr/bin/env python3
"""Compare the ffwd server against the official model snapshot.

Builds and launches ffwd-server with the given pplx-embed model, then for each
text requests both float and base64_int8 embeddings from /v1/embeddings and
compares them against SentenceTransformer(model_dir).encode(). pplx-embed emits
unnormalized int8, so this checks the server's int8 output against the reference.

Pass --base-url to test an already-running server instead of launching one. The
reference side needs torch + sentence-transformers, so run it with uv:

  uv run --python 3.12 --with-requirements requirements-reference.txt \
      tests/check_reference_parity.py --model-dir DIR
"""

from __future__ import annotations

import argparse
import base64
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

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
API_KEY = "tt-test-key"
SERVER_BIN = ROOT / "ffwd-server"

DEFAULT_TEXTS = [
    "document: Your text string goes here",
    "document: what is the capital of France?",
    "document: Redis is an in-memory data structure server used for caching.",
]


def ensure_server() -> None:
    """Build ffwd-server if its binary is missing."""
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


def post_json(base_url: str, payload: dict, api_key: str | None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/embeddings",
                                 data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {text}") from exc


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
    """Poll until a real embeddings request returns, retrying while the model loads."""
    deadline = time.monotonic() + timeout_s
    warmup = {"model": model, "input": "ready?", "encoding_format": "float"}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(proc, log_path, "server exited during startup")
        try:
            post_json(base_url, warmup, api_key)
            return
        except (OSError, RuntimeError):
            pass
        time.sleep(0.2)
    die_with_log(proc, log_path, "server did not become ready in time")


def decode_int8(value: str) -> list[int]:
    return [b - 256 if b > 127 else b for b in base64.b64decode(value)]


def cosine(a: list[int], b: list[int]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def local_vectors(base_url: str, api_key: str, model: str,
                  text: str) -> tuple[list[int], list[int]]:
    common = {"model": model, "input": text}
    floats = post_json(base_url, {**common, "encoding_format": "float"}, api_key)
    encoded = post_json(base_url, {**common, "encoding_format": "base64_int8"}, api_key)

    float_values = floats["data"][0]["embedding"]
    if not isinstance(float_values, list):
        raise RuntimeError("float response did not contain an array")
    float_int8 = [round(float(x) * 128.0) for x in float_values]

    b64_value = encoded["data"][0]["embedding"]
    if not isinstance(b64_value, str):
        raise RuntimeError("base64_int8 response did not contain a string")
    return float_int8, decode_int8(b64_value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model", default="pplx-embed-v1-0.6b")
    ap.add_argument("--base-url", default=None,
                    help="test a running server instead of launching one")
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--max-int8-diff", type=int, default=1)
    ap.add_argument("--min-cosine", type=float, default=0.9999)
    ap.add_argument("--text", action="append", dest="texts")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    texts = args.texts or DEFAULT_TEXTS
    reference = SentenceTransformer(args.model_dir, trust_remote_code=True,
                                    local_files_only=True)

    proc: subprocess.Popen | None = None
    log_path: str | None = None
    base_url = args.base_url
    api_key = args.api_key

    if base_url is None:
        ensure_server()
        port = free_port()
        base_url = f"http://{HOST}:{port}"
        log = open(tempfile.mktemp(prefix="ffwd-ref-server-", suffix=".log"), "w")
        log_path = log.name
        argv = [str(SERVER_BIN), "--host", HOST, "--port", str(port),
                "--api-key", api_key, "-b", "2",
                "--model", f"{args.model}={args.model_dir}:api=perplexity"]
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        wait_ready(base_url, api_key, args.model, proc, log_path)

    try:
        ok = True
        for text in texts:
            official = [round(float(x)) for x in reference.encode([text])[0]]
            local_float, local_int8 = local_vectors(base_url, api_key, args.model, text)
            if local_float != local_int8:
                raise RuntimeError("local float and base64_int8 responses disagree")
            max_diff = max(abs(a - b) for a, b in zip(local_int8, official))
            equal = sum(a == b for a, b in zip(local_int8, official))
            score = cosine(local_int8, official)
            passed = (len(local_int8) == len(official)
                      and max_diff <= args.max_int8_diff
                      and score >= args.min_cosine)
            ok = ok and passed
            print(f"{'ok' if passed else 'FAIL'}: equal={equal}/{len(official)} "
                  f"max_int8_diff={max_diff} cosine={score:.8f} text={text!r}")
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

    print(f"{'ok' if ok else 'FAIL'}: server reference parity for {len(texts)} "
          f"texts ({args.model})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
