#!/usr/bin/env python3
"""Hermetic end-to-end test of the ffwd-server HTTP API.

Builds the tiny synthetic fixture (the gen_fixture test helper, the same model
the C tests write) and the ffwd-server binary, launches the server on a free
loopback port against the fixture across standard, contextual, and late/rerank
model slots, then drives the full HTTP request path and asserts on it: routing,
auth, malformed input, validation 422s, output encodings, Matryoshka
truncation, text_type prompting, micro-batching, exact token counts, and rerank
ordering.

The fixture weights are deterministic but meaningless, so this checks PROTOCOL
and STRUCTURE, not embedding quality. It is the out-of-process successor to the
in-process C test that used to live in tests/test_server.c; the server's
internal helpers are unit-tested separately under tools/server.

Usage: tests/check_server_api.py [--server PATH] [--gen PATH] [-v]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOST = "127.0.0.1"
API_KEY = "tt-test-key"
STD_DIM = 1024
GTE_DIM = 1536
LATE_TOKEN_DIM = 128

# The retrieval task text. Qwen3 has no trailing space after "Query:"; GTE-Qwen2
# adds one. The instruction holds a newline and colons, so the model-spec parser
# must take query_instruct= as the final key (it does) and unescape \n.
QWEN_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)
GTE_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery: "
)

PORT = 0
VERBOSE = False
failures = 0


def fail(what: str) -> None:
    global failures
    failures += 1
    print(f"FAIL: {what}", file=sys.stderr)


def expect(cond: bool, what: str) -> None:
    if not cond:
        fail(what)


# --------------------------------------------------------------------------
# Build + fixture + launch
# --------------------------------------------------------------------------

def ensure_tool(path: str, make_dir: str, make_target: str) -> None:
    """Build a Make target if its output binary is missing."""
    if not os.path.exists(path):
        subprocess.run(["make", "-C", make_dir, make_target], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL)
    if not os.path.exists(path):
        sys.exit(f"required binary not found and could not be built: {path}")


def write_fixtures(gen: str, root: str) -> dict[str, str]:
    """Write the four fixture model directories and return their paths.

    std-main and ctx-main share one base model; the late model adds a Dense
    projection; qwen3 and gte-qwen2 differ in architecture and hidden size.
    """
    dirs = {
        "base": os.path.join(root, "base"),
        "late": os.path.join(root, "late"),
        "qwen": os.path.join(root, "qwen"),
        "gte": os.path.join(root, "gte"),
    }
    specs = [
        (dirs["base"], ["--model", "base", "--hidden", str(STD_DIM)]),
        (dirs["late"], ["--model", "base", "--hidden", str(STD_DIM),
                        "--late-dim", str(LATE_TOKEN_DIM)]),
        (dirs["qwen"], ["--model", "qwen3", "--hidden", str(STD_DIM)]),
        (dirs["gte"], ["--model", "qwen2", "--hidden", str(GTE_DIM)]),
    ]
    for out, args in specs:
        os.mkdir(out)
        subprocess.run([gen, "--out", out, *args], check=True,
                       stdout=subprocess.DEVNULL)
    return dirs


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def cli_instruct(text: str) -> str:
    """Encode a newline as the \\n the model-spec parser unescapes."""
    return text.replace("\n", "\\n")


def server_argv(server: str, dirs: dict[str, str], port: int) -> list[str]:
    return [
        server,
        "--host", HOST,
        "--port", str(port),
        "--api-key", API_KEY,
        "-b", "2",
        "--batch-wait-us", "1000",
        "--model", f"std-main={dirs['base']}:api=perplexity:min_dim=128",
        "--contextual-model", f"ctx-main={dirs['base']}:api=perplexity:min_dim=128",
        "--late-model", f"late-main={dirs['late']}:api=perplexity:min_dim=128",
        "--model", f"qwen-search={dirs['qwen']}:min_dim=32"
                   f":query_instruct={cli_instruct(QWEN_INSTRUCT)}",
        "--model", f"gte-search={dirs['gte']}:min_dim=1536"
                   f":query_instruct={cli_instruct(GTE_INSTRUCT)}",
    ]


def wait_ready(proc: subprocess.Popen, log_path: str, timeout_s: float = 40.0) -> None:
    """Poll the server until a real embeddings request returns 200.

    The model load happens in the worker thread, so the first request may block
    or be refused while the server comes up; retry until it succeeds.
    """
    deadline = time.monotonic() + timeout_s
    warmup = {"model": "std-main", "input": "ready?", "encoding_format": "float"}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            _die_with_log(proc, log_path, "server exited during startup")
        try:
            status, _ = request("POST", "/v1/embeddings", body=warmup, timeout=5)
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    _die_with_log(proc, log_path, "server did not become ready in time")


def _die_with_log(proc: subprocess.Popen, log_path: str, msg: str) -> None:
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


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def request(method: str, path: str, body=None, auth: str | None = API_KEY,
            timeout: float = 30.0) -> tuple[int, str]:
    """One request over a fresh connection; returns (status, body text).

    body: dict/list -> JSON; str -> sent verbatim (for malformed-JSON checks);
    None -> no body.
    """
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode()
    elif isinstance(body, str):
        data = body.encode()
    else:
        data = None
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    import http.client
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        text = resp.read().decode("utf-8", "replace")
        if VERBOSE:
            print(f"  {method} {path} -> {resp.status}", file=sys.stderr)
        return resp.status, text
    finally:
        conn.close()


def request_raw(raw: bytes, timeout: float = 10.0) -> int:
    """Send raw bytes on the wire and return the parsed status code (or -1)."""
    with socket.create_connection((HOST, PORT), timeout=timeout) as s:
        s.sendall(raw)
        s.settimeout(timeout)
        data = b""
        try:
            while b"\r\n" not in data:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
        except OSError:
            pass
    try:
        return int(data.split(b"\r\n", 1)[0].split()[1])
    except (IndexError, ValueError):
        return -1


def embeddings(model: str, **fields) -> tuple[int, str]:
    return request("POST", "/v1/embeddings", body={"model": model, **fields})


def max_absdiff(a, b) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


# --------------------------------------------------------------------------
# Test groups (ported from tests/test_server.c)
# --------------------------------------------------------------------------

def test_routing_and_auth() -> None:
    expect(request("OPTIONS", "/v1/embeddings", auth=None)[0] == 204,
           "OPTIONS preflight returns 204")
    expect(request("GET", "/v1/embeddings")[0] == 404, "GET on embeddings is 404")
    expect(request("POST", "/nope", body={})[0] == 404, "unknown path is 404")

    ok = {"model": "std-main", "input": "hello world", "encoding_format": "float"}
    expect(request("POST", "/v1/embeddings", body=ok, auth=None)[0] == 401,
           "missing auth is 401")
    expect(request("POST", "/v1/embeddings", body=ok, auth="wrong-key")[0] == 401,
           "wrong key is 401")

    expect(request_raw(b"GARBAGE\r\n\r\n") == 400, "malformed HTTP is 400")
    expect(request("POST", "/v1/embeddings", body="{nope")[0] == 422,
           "malformed JSON is 422")


def test_validation() -> None:
    cases = [
        ({"input": "hi"}, "missing model"),
        ({"model": "foo", "input": "hi"}, "unknown model"),
        ({"model": "ctx-main", "input": "hi"}, "contextual model on /v1/embeddings"),
        ({"model": "std-missing", "input": "hi"}, "unloaded model"),
        ({"model": "std-main", "input": "hi", "encoding_format": "yaml"},
         "bad encoding enum"),
        ({"model": "std-main", "input": "hi", "dimensions": 64},
         "dimensions below min_dim"),
        ({"model": "std-main", "input": []}, "empty input array"),
    ]
    for body, label in cases:
        expect(request("POST", "/v1/embeddings", body=body)[0] == 422,
               f"422 for {label}")

    expect(request("POST", "/v1/contextualizedembeddings",
                   body={"model": "std-main", "input": [["a"]]})[0] == 422,
           "standard model on contextual endpoint is 422")
    expect(request("POST", "/v1/contextualizedembeddings",
                   body={"model": "ctx-missing", "input": [["a"]]})[0] == 422,
           "unknown contextual model is 422")


def test_standard_embeddings() -> None:
    status, text = embeddings("std-main", input="hello world", encoding_format="float")
    expect(status == 200, "standard float request is 200")
    if status == 200:
        root = json.loads(text)
        data = root["data"]
        expect(len(data) == 1, "one embedding for one input")
        emb = data[0]["embedding"]
        expect(isinstance(emb, list) and len(emb) == STD_DIM,
               "float embedding has full dimension")
        expect(data[0]["index"] == 0, "index is 0")
        expect(all(isinstance(v, (int, float)) for v in emb),
               "embedding values are numbers")
        expect(any(v != 0.0 for v in emb), "embedding is not all zeros")
        expect(root["usage"]["total_tokens"] >= 1, "usage reports tokens")

    # Batch: one row per input, ascending indexes, distinct vectors.
    status, text = embeddings("std-main", input=["hello", "world", "held"],
                              encoding_format="float")
    expect(status == 200, "batch request is 200")
    if status == 200:
        data = json.loads(text)["data"]
        expect(len(data) == 3, "three embeddings for three inputs")
        expect([d["index"] for d in data] == [0, 1, 2], "indexes are 0..n-1")
        expect(max_absdiff(data[0]["embedding"], data[1]["embedding"]) > 0.0,
               "different inputs give different vectors")

    # Perplexity default encoding is base64_int8: a string of the int8 base64.
    b64_len = ((STD_DIM + 2) // 3) * 4
    for label, fields in [("default", {}),
                          ("explicit", {"encoding_format": "base64_int8"})]:
        status, text = embeddings("std-main", input="hello", **fields)
        expect(status == 200, f"{label} int8 request is 200")
        if status == 200:
            emb = json.loads(text)["data"][0]["embedding"]
            expect(isinstance(emb, str) and len(emb) == b64_len,
                   f"{label} encoding is base64_int8 string")

    expect(embeddings("std-main", input="hello", encoding_format="base64")[0] == 422,
           "OpenAI base64 is rejected on a Perplexity model")

    # Matryoshka truncation via dimensions.
    status, text = embeddings("std-main", input="hello", dimensions=128,
                              encoding_format="float")
    expect(status == 200, "matryoshka request is 200")
    if status == 200:
        emb = json.loads(text)["data"][0]["embedding"]
        expect(isinstance(emb, list) and len(emb) == 128, "truncated to 128 dims")


def test_qwen_mrl_and_text_type() -> None:
    status, text = embeddings("qwen-search", input="hello", encoding_format="float")
    expect(status == 200, "qwen full request is 200")
    full = json.loads(text)["data"][0]["embedding"] if status == 200 else None

    status, text = embeddings("qwen-search", input="hello", dimensions=32,
                              encoding_format="float")
    expect(status == 200, "qwen MRL-32 request is 200")
    if status == 200:
        root = json.loads(text)
        mrl = root["data"][0]["embedding"]
        expect(isinstance(full, list) and len(full) == STD_DIM, "qwen full is 1024")
        expect(isinstance(mrl, list) and len(mrl) == 32, "qwen MRL is 32")
        if isinstance(full, list) and isinstance(mrl, list):
            expect(max((abs(full[i] - mrl[i]) for i in range(32)), default=0.0) > 0.0,
                   "MRL re-normalizes the truncated prefix")
        # hello plus Qwen3's terminal token = 2 tokens.
        expect(root["usage"]["total_tokens"] == 2, "qwen usage counts the EOT token")

    # text_type=query must equal manually prepending the instruction with
    # text_type=document: same tokens, hence the same vector.
    for model, instruct, dim in [("qwen-search", QWEN_INSTRUCT, STD_DIM),
                                 ("gte-search", GTE_INSTRUCT, GTE_DIM)]:
        sa, ta = embeddings(model, input="hello", text_type="query",
                            encoding_format="float")
        sb, tb = embeddings(model, input=instruct + "hello", text_type="document",
                            encoding_format="float")
        expect(sa == 200 and sb == 200, f"{model} text_type requests are 200")
        if sa == 200 and sb == 200:
            qe = json.loads(ta)["data"][0]["embedding"]
            me = json.loads(tb)["data"][0]["embedding"]
            expect(isinstance(qe, list) and len(qe) == dim, f"{model} query dim {dim}")
            expect(max_absdiff(qe, me) < 1e-6,
                   f"{model} text_type=query equals the manual instruction")

    expect(embeddings("std-main", input="hello", text_type="query")[0] == 422,
           "text_type on a model without a query instruction is 422")
    expect(embeddings("qwen-search", input="hello", text_type="passage")[0] == 422,
           "invalid text_type value is 422")


def test_contextual() -> None:
    expect(request("POST", "/v1/contextualizedembeddings",
                   body={"model": "ctx-main", "input": "hi"})[0] == 422,
           "contextual input must be an array of chunk arrays")
    expect(request("POST", "/v1/contextualizedembeddings",
                   body={"model": "ctx-main", "input": [[""]]})[0] == 422,
           "contextual chunks must be non-empty")

    status, text = request("POST", "/v1/contextualizedembeddings",
                           body={"model": "ctx-main",
                                 "input": [["hello", "world"], ["held"]],
                                 "encoding_format": "float"})
    expect(status == 200, "contextual happy path is 200")
    multi = json.loads(text) if status == 200 else None
    if multi:
        docs = multi["data"]
        expect(len(docs) == 2, "two documents")
        for di, doc in enumerate(docs):
            expect(doc["index"] == di, f"doc {di} index")
            chunks = doc["data"]
            expect(len(chunks) == (2 if di == 0 else 1), f"doc {di} chunk count")
            for ci, chunk in enumerate(chunks):
                expect(chunk["index"] == ci, f"doc {di} chunk {ci} index")
                expect(isinstance(chunk["embedding"], list)
                       and len(chunk["embedding"]) == STD_DIM,
                       f"doc {di} chunk {ci} embedding dim")
        # hello=1, world=3, held=2, one separator in doc 0 -> 7.
        expect(multi["usage"]["total_tokens"] == 7, "contextual token accounting")

    # A single-chunk document covers the whole sequence, so it must match the
    # standard embedding of the same text; a chunk beside a neighbor must not.
    status, text = request("POST", "/v1/contextualizedembeddings",
                           body={"model": "ctx-main", "input": [["hello"]],
                                 "encoding_format": "float"})
    single = json.loads(text) if status == 200 else None
    status, text = embeddings("std-main", input="hello", encoding_format="float")
    std = json.loads(text) if status == 200 else None
    if multi and single and std:
        std_emb = std["data"][0]["embedding"]
        alone = single["data"][0]["data"][0]["embedding"]
        in_doc = multi["data"][0]["data"][0]["embedding"]
        expect(max_absdiff(std_emb, alone) < 1e-4,
               "single-chunk contextual equals standard embedding")
        expect(max_absdiff(alone, in_doc) > 1e-3,
               "whole-document attention changes a chunk beside a neighbor")


def test_rerank() -> None:
    expect(request("POST", "/v1/rerank",
                   body={"model": "std-main", "query": "hello",
                         "documents": ["world"]})[0] == 422,
           "rerank on a standard model is 422")
    expect(request("POST", "/v1/rerank",
                   body={"model": "late-main", "query": "",
                         "documents": ["world"]})[0] == 422,
           "empty query is 422")
    expect(request("POST", "/v1/rerank",
                   body={"model": "late-main", "query": "hello",
                         "documents": []})[0] == 422,
           "empty documents is 422")
    expect(request("POST", "/v1/rerank",
                   body={"model": "late-main", "query": "hello",
                         "documents": ["world"], "top_n": 1, "top_k": 1})[0] == 422,
           "top_n and top_k together is 422")

    status, text = request("POST", "/v1/rerank",
                           body={"model": "late-main", "query": "hello",
                                 "documents": ["hello!", "world",
                                               'quoted "document"'],
                                 "top_n": 2, "return_documents": True})
    expect(status == 200, "rerank happy path is 200")
    if status == 200:
        root = json.loads(text)
        expect(root.get("object") == "list", "rerank object is list")
        expect(root.get("model") == "late-main", "rerank echoes the model")
        results = root["results"]
        expect(len(results) == 2, "top_n=2 returns two results")
        scores = [r["relevance_score"] for r in results]
        expect(scores == sorted(scores, reverse=True), "results sorted by score")
        idxs = [r["index"] for r in results]
        expect(all(0 <= i < 3 for i in idxs) and len(set(idxs)) == 2,
               "result indexes are distinct and in range")
        expect(all(isinstance(r.get("document"), str) for r in results),
               "return_documents includes the document text")
        usage = root["usage"]
        expect(usage["query_tokens"] == 32, "query expands to 32 tokens")
        expect(usage["document_tokens"] > 0, "documents contribute tokens")
        expect(usage["total_tokens"] == usage["query_tokens"] + usage["document_tokens"],
               "rerank token accounting is consistent")

    status, text = request("POST", "/v1/rerank",
                           body={"model": "late-main", "query": "hello",
                                 "documents": ["hello", "world"], "top_k": 1})
    expect(status == 200, "rerank top_k alias is 200")
    if status == 200:
        results = json.loads(text)["results"]
        expect(len(results) == 1, "top_k=1 returns one result")
        expect("document" not in results[0],
               "documents are omitted unless requested")


def test_concurrent() -> None:
    """More in-flight requests than the micro-batch cap (-b 2)."""
    import threading
    n = 6
    results = [None] * n
    body = {"model": "std-main", "input": "hello world", "encoding_format": "float"}

    def worker(i: int) -> None:
        try:
            results[i] = request("POST", "/v1/embeddings", body=body)[0]
        except OSError as e:  # noqa: BLE001
            results[i] = f"error: {e}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    expect(all(r == 200 for r in results),
           f"all {n} concurrent requests return 200 (got {results})")


def test_keepalive() -> None:
    """One HTTP/1.1 connection is reused across requests (no Connection: close)."""
    import http.client
    conn = http.client.HTTPConnection(HOST, PORT, timeout=30)
    body = json.dumps({"model": "std-main", "input": ["hello", "world"],
                       "encoding_format": "float"}).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {API_KEY}"}
    first_sock = None
    try:
        for i in range(2):
            conn.request("POST", "/v1/embeddings", body=body, headers=headers)
            resp = conn.getresponse()
            resp.read()
            expect(resp.status == 200, f"keep-alive request {i} is 200")
            expect((resp.getheader("Connection") or "").lower() != "close",
                   "server does not send Connection: close")
            expect(conn.sock is not None, "connection stays open after response")
            if i == 0:
                first_sock = conn.sock
            else:
                expect(conn.sock is first_sock, "connection is reused, not reconnected")
    finally:
        conn.close()


# --------------------------------------------------------------------------

def main() -> int:
    global PORT, VERBOSE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default=os.path.join(ROOT, "tools", "server",
                                                      "ffwd-server"),
                    help="path to the ffwd-server binary")
    ap.add_argument("--gen", default=os.path.join(ROOT, "tests", "gen_fixture"),
                    help="path to the gen_fixture helper")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose

    ensure_tool(args.server, "tools/server", "all")
    ensure_tool(args.gen, "tests", "gen_fixture")

    with tempfile.TemporaryDirectory(prefix="ffwd-server-check-") as workdir:
        dirs = write_fixtures(args.gen, workdir)
        PORT = free_port()
        log_path = os.path.join(workdir, "server.log")
        with open(log_path, "w") as log:
            proc = subprocess.Popen(server_argv(args.server, dirs, PORT),
                                    stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(proc, log_path)
            test_routing_and_auth()
            test_validation()
            test_standard_embeddings()
            test_keepalive()
            test_qwen_mrl_and_text_type()
            test_contextual()
            test_rerank()
            test_concurrent()
            # Graceful shutdown: SIGTERM must unwind the server to a clean exit
            # code 0 (the old C test's `ctx.rc == 0` check).
            proc.terminate()
            try:
                expect(proc.wait(timeout=10) == 0, "server exits cleanly on SIGTERM")
            except subprocess.TimeoutExpired:
                fail("server did not exit within 10s of SIGTERM")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    if failures:
        print(f"{failures} server API check(s) failed", file=sys.stderr)
        return 1
    print("ok: ffwd-server HTTP request-path checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
