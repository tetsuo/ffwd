#!/usr/bin/env python3
"""Verify the local server through the official Perplexity Python SDK.

Builds and launches ffwd-server with the pplx standard and contextual models,
then drives /v1/embeddings and /v1/contextualizedembeddings through the official
perplexityai SDK, checking response shape and retrieval ranking. Pass --base-url
to test an already-running server instead of launching one.

The SDK and numpy are not in requirements-reference.txt, so run with uv:

  uv run --python 3.12 --with perplexityai --with numpy \
      tests/check_perplexity_sdk_compat.py \
      --model-dir DIR --context-model-dir CTXDIR
"""

from __future__ import annotations

import argparse
import base64
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from perplexity import APIStatusError, Perplexity
except ImportError as e:
    raise SystemExit(
        "missing dependency; run with: uv run --python 3.12 "
        "--with perplexityai --with numpy tests/check_perplexity_sdk_compat.py ..."
    ) from e

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
SERVER_BIN = ROOT / "tools/server/ffwd-server"


STANDARD_DOCS = [
    ("RAG Overview", "Retrieval-augmented generation grounds LLM responses in external data."),
    ("RAG Overview", "RAG reduces hallucinations by providing factual context to the model."),
    ("RAG Overview", "A typical RAG pipeline has three stages: indexing, retrieval, and generation."),
    ("Embedding Models", "Embedding models map text to dense vector representations."),
    ("Embedding Models", "Similar texts produce vectors that are close in the embedding space."),
    ("Embedding Models", "Perplexity offers both standard and contextualized embedding models."),
]

CONTEXT_DOCS = {
    "Introduction to Transformers": [
        "The Transformer architecture replaced recurrent layers with self-attention mechanisms.",
        "Multi-head attention lets a model attend to information from different representation subspaces.",
        "Transformers became the foundation for modern language models including BERT, GPT, and T5.",
    ],
    "Retrieval-Augmented Generation": [
        "Retrieval-Augmented Generation combines information retrieval with text generation.",
        "RAG systems retrieve relevant documents and use them as context.",
        "This reduces hallucination because the model grounds responses in retrieved evidence.",
    ],
}


# -------------------------------------------------------------------------
# Server launch (skipped when --base-url points at a running server)
# -------------------------------------------------------------------------

def ensure_server() -> None:
    if not SERVER_BIN.exists():
        subprocess.run(["make", "-C", "tools/server", "all"], cwd=ROOT, check=True,
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


# -------------------------------------------------------------------------
# SDK checks
# -------------------------------------------------------------------------

def decode_int8(embedding: str) -> np.ndarray:
    raw = base64.b64decode(embedding)
    return np.frombuffer(raw, dtype=np.int8).astype(np.float32)


def decode_binary(embedding: str, dimensions: int) -> np.ndarray:
    raw = base64.b64decode(embedding)
    out = np.empty(dimensions, dtype=np.float32)
    pos = 0
    for byte in raw:
        for bit in range(8):
            out[pos] = 1.0 if byte & (1 << bit) else -1.0
            pos += 1
            if pos == dimensions:
                return out
    if pos != dimensions:
        raise RuntimeError(f"binary embedding decoded to {pos} dimensions, expected {dimensions}")
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def require_embedding(obj: Any, label: str) -> str:
    emb = getattr(obj, "embedding", None)
    if not isinstance(emb, str) or not emb:
        raise RuntimeError(f"{label}: missing embedding string in SDK response")
    return emb


def require_data(resp: Any, label: str) -> list[Any]:
    data = getattr(resp, "data", None)
    if not isinstance(data, list):
        raise RuntimeError(f"{label}: missing data list in SDK response")
    return data


def contextual_chunk_rows(resp: Any, label: str) -> list[tuple[int, int, Any]]:
    rows = []
    for doc_obj in require_data(resp, label):
        di = getattr(doc_obj, "index", None)
        if not isinstance(di, int):
            raise RuntimeError(f"{label}: missing document index")
        for chunk_obj in require_data(doc_obj, f"{label} doc {di}"):
            ci = getattr(chunk_obj, "index", None)
            if not isinstance(ci, int):
                raise RuntimeError(f"{label}: missing chunk index")
            rows.append((di, ci, chunk_obj))
    return rows


def check_standard(client: Perplexity, model: str, dimensions: int,
                   encoding: str) -> bool:
    texts = [text for _, text in STANDARD_DOCS]
    print("standard SDK call:")
    resp = client.embeddings.create(
        input=texts,
        model=model,  # type: ignore[arg-type]
        dimensions=dimensions,
        encoding_format=encoding,  # type: ignore[arg-type]
    )
    rows = require_data(resp, "standard")
    if len(rows) != len(texts):
        raise RuntimeError(f"standard: got {len(rows)} rows, expected {len(texts)}")

    vecs = [decode_int8(require_embedding(row, "standard")) for row in rows]
    if any(len(v) != dimensions for v in vecs):
        raise RuntimeError("standard: decoded dimension mismatch")

    q_resp = client.embeddings.create(
        input=["What are the stages of a RAG pipeline?"],
        model=model,  # type: ignore[arg-type]
        dimensions=dimensions,
        encoding_format=encoding,  # type: ignore[arg-type]
    )
    q_vec = decode_int8(require_embedding(require_data(q_resp, "standard query")[0], "standard query"))

    ranked = sorted(
        (
            {"title": title, "text": text, "score": cosine(q_vec, vec)}
            for (title, text), vec in zip(STANDARD_DOCS, vecs)
        ),
        key=lambda row: row["score"],
        reverse=True,
    )

    print(f"  model={getattr(resp, 'model', None)} dimensions={dimensions} rows={len(rows)}")
    for row in ranked[:3]:
        print(f"  {row['score']: .4f}  [{row['title']}] {row['text']}")

    ok = ranked[0]["title"] == "RAG Overview"
    print(f"  verdict={'ok' if ok else 'FAIL'}: expected RAG Overview near the top")
    return ok


def check_binary(client: Perplexity, model: str, dimensions: int) -> bool:
    print("binary SDK call:")
    resp = client.embeddings.create(
        input=["Matryoshka embeddings allow dimension reduction."],
        model=model,  # type: ignore[arg-type]
        dimensions=dimensions,
        encoding_format="base64_binary",
    )
    vec = decode_binary(require_embedding(require_data(resp, "binary")[0], "binary"), dimensions)
    ok = len(vec) == dimensions and all(v in (-1.0, 1.0) for v in vec)
    print(f"  dimensions={len(vec)} verdict={'ok' if ok else 'FAIL'}")
    return ok


def check_contextual(client: Perplexity, model: str, dimensions: int,
                     encoding: str) -> bool:
    docs = list(CONTEXT_DOCS.values())
    print("contextual SDK call:")
    try:
        resp = client.contextualized_embeddings.create(
            input=docs,
            model=model,  # type: ignore[arg-type]
            dimensions=dimensions,
            encoding_format=encoding,  # type: ignore[arg-type]
        )
    except APIStatusError as e:
        if e.status_code == 503:
            print(f"  skipped: {model} is valid but not loaded by this server")
            return True
        raise

    if len(require_data(resp, "contextual")) != len(docs):
        raise RuntimeError("contextual: response document count mismatch")

    index = []
    titles = list(CONTEXT_DOCS.keys())
    for di, ci, chunk_obj in contextual_chunk_rows(resp, "contextual"):
        emb = decode_int8(require_embedding(chunk_obj, f"contextual doc {di} chunk {ci}"))
        if len(emb) != dimensions:
            raise RuntimeError("contextual: decoded dimension mismatch")
        index.append({"title": titles[di], "text": docs[di][ci], "embedding": emb})

    query = "How does RAG reduce hallucination?"
    q_resp = client.contextualized_embeddings.create(
        input=[[query]],
        model=model,  # type: ignore[arg-type]
        dimensions=dimensions,
        encoding_format=encoding,  # type: ignore[arg-type]
    )
    query_rows = contextual_chunk_rows(q_resp, "contextual query")
    if len(query_rows) != 1:
        raise RuntimeError("contextual query: expected exactly one chunk")
    q_vec = decode_int8(require_embedding(query_rows[0][2], "contextual query chunk"))

    ranked = sorted(
        ({"score": cosine(q_vec, item["embedding"]), **item} for item in index),
        key=lambda row: row["score"],
        reverse=True,
    )

    print(f"  model={getattr(resp, 'model', None)} dimensions={dimensions} chunks={len(index)}")
    for row in ranked[:3]:
        print(f"  {row['score']: .4f}  [{row['title']}] {row['text']}")

    ok = ranked[0]["title"] == "Retrieval-Augmented Generation"
    print(f"  verdict={'ok' if ok else 'FAIL'}: expected Retrieval-Augmented Generation near the top")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=None,
                    help="test a running server instead of launching one")
    ap.add_argument("--model-dir", default=None, help="standard model dir (for launch)")
    ap.add_argument("--context-model-dir", default=None, help="contextual model dir (for launch)")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model", default="pplx-embed-v1-0.6b")
    ap.add_argument("--context-model", default="pplx-embed-context-v1-0.6b")
    ap.add_argument("--dimensions", type=int, default=128)
    ap.add_argument("--skip-contextual", action="store_true")
    ap.add_argument("--skip-binary", action="store_true")
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
                "--model", f"{args.model}={args.model_dir}:api=perplexity:min_dim={args.dimensions}"]
        if not args.skip_contextual and args.context_model_dir:
            argv += ["--contextual-model",
                     f"{args.context_model}={args.context_model_dir}"
                     f":api=perplexity:min_dim={args.dimensions}"]
        log = open(tempfile.mktemp(prefix="ffwd-pplx-sdk-", suffix=".log"), "w")
        log_path = log.name
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        wait_ready(base_url, args.api_key, args.model, proc, log_path)

    try:
        client = Perplexity(base_url=base_url, api_key=args.api_key)
        ok = check_standard(client, args.model, args.dimensions, "base64_int8")
        if not args.skip_binary:
            ok = check_binary(client, args.model, args.dimensions) and ok
        if not args.skip_contextual:
            ok = check_contextual(client, args.context_model, args.dimensions, "base64_int8") and ok
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
