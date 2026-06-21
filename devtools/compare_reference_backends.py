#!/usr/bin/env python3
"""Compare ffwd output with offline reference implementations.

Benchmarks the native CLI (stdin) or server (http) against sentence-transformers
and ONNX, reporting cosine parity, latency, and peak RSS. Needs the reference
stack, so run it with uv (Python 3.12 for the numpy pin):

  uv run --python 3.12 --with-requirements requirements-reference.txt \
      devtools/compare_reference_backends.py --model-dir DIR \
      --native-mode stdin --native-backend cpu --references sentence-transformers
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


DEFAULT_STANDARD = [
    "Scientists explore the universe driven by curiosity.",
    "Children learn through curious exploration.",
    "Historical discoveries began with curious questions.",
    "Redis is an in-memory data structure server often used for caching.",
    "SQLite stores relational data in a single local database file.",
]

DEFAULT_CONTEXTUAL = [
    [
        "Paris is the capital city of France.",
        "The Eiffel Tower is a landmark in Paris.",
        "France is a country in Western Europe.",
    ],
    [
        "Redis stores data in memory.",
        "SQLite stores relational data in a local file.",
    ],
]

RESULT_PREFIX = "EMBED_REFERENCE_RESULT="


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    pos = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[pos]


def cosine(a: list[int], b: list[int]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    aa = sum(x * x for x in a)
    bb = sum(x * x for x in b)
    return dot / math.sqrt(aa * bb) if aa and bb else 0.0


def int8_rows(value: Any) -> list[list[int]]:
    rows = value.tolist() if hasattr(value, "tolist") else value
    return [[int(round(float(x))) for x in row] for row in rows]


def flatten_contextual(value: Any) -> list[list[int]]:
    rows: list[list[int]] = []
    for document in value:
        rows.extend(int8_rows(document))
    return rows


def labels(kind: str, inputs: Any) -> list[str]:
    if kind == "standard":
        return [f"text[{i}]" for i in range(len(inputs))]
    return [
        f"doc[{di}]/chunk[{ci}]"
        for di, document in enumerate(inputs)
        for ci in range(len(document))
    ]


def offline_environment(cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(cache_dir),
            "HF_HUB_CACHE": str(cache_dir / "hub"),
            "HF_MODULES_CACHE": str(cache_dir / "modules"),
            "TRANSFORMERS_CACHE": str(cache_dir / "transformers"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        return int(proc.stdout.strip()) / 1024.0
    except ValueError:
        return None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc.reason}") from exc


def decode_native(response: dict[str, Any], kind: str) -> list[list[int]]:
    if kind == "standard":
        entries = response["data"]
    else:
        entries = [chunk for document in response["data"] for chunk in document["data"]]
    rows = []
    for entry in entries:
        encoded = entry["embedding"]
        raw = base64.b64decode(encoded)
        rows.append([x - 256 if x > 127 else x for x in raw])
    return rows


def quantize_native(rows: list[dict[str, Any]]) -> list[list[int]]:
    vectors = []
    for row in rows:
        if "error" in row:
            raise RuntimeError(f"native stdin returned an error: {row['error']}")
        vectors.append(
            [
                max(-128, min(127, round(math.tanh(float(x)) * 127.0)))
                for x in row["embedding"]
            ]
        )
    return vectors


class RssGuard:
    def __init__(self, proc: subprocess.Popen[str], ceiling_mb: float):
        self.proc = proc
        self.ceiling_mb = ceiling_mb
        self.peak_mb = 0.0
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        while not self.stop_event.wait(0.025):
            current = rss_mb(self.proc.pid) or 0.0
            self.peak_mb = max(self.peak_mb, current)
            if current > self.ceiling_mb:
                self.error = (
                    f"native stdin exceeded RSS ceiling: "
                    f"{current:.1f} > {self.ceiling_mb} MiB"
                )
                self.proc.kill()
                return
            if self.proc.poll() is not None:
                return

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak_mb = max(self.peak_mb, rss_mb(self.proc.pid) or 0.0)


def benchmark_native_stdin(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind != "standard":
        raise RuntimeError("native stdin comparison currently supports standard embeddings only")
    cmd = [
        args.native_binary,
        "-d",
        args.model_dir,
        "--stream",
        "-b",
        str(len(args.inputs)),
    ]
    if args.native_backend == "cpu":
        cmd.extend(["-t", str(args.threads)])
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    guard = RssGuard(proc, args.max_native_rss_mb)
    guard.start()

    def send() -> tuple[list[list[int]], float]:
        if not proc.stdin or not proc.stdout:
            raise RuntimeError("native stdin pipes are unavailable")
        proc.stdin.write("\n".join(args.inputs) + "\n")
        proc.stdin.flush()
        rows = []
        for _ in args.inputs:
            line = proc.stdout.readline()
            if not line:
                errors = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(guard.error or f"native stdin exited early:\n{errors}")
            rows.append(json.loads(line))
        return quantize_native(rows), float(rows[0]["ms"])

    try:
        start = time.perf_counter()
        vectors, _ = send()
        load_ms = (time.perf_counter() - start) * 1000.0
        for _ in range(args.warmups):
            vectors, _ = send()
        elapsed = []
        for _ in range(args.runs):
            vectors, engine_ms = send()
            elapsed.append(engine_ms)
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        guard.stop()
    if guard.error:
        raise RuntimeError(guard.error)
    if proc.returncode != 0:
        errors = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"native stdin failed ({proc.returncode}):\n{errors}")
    return {
        "name": f"native-stdin-{args.native_backend}",
        "vectors": vectors,
        "load_ms": load_ms,
        "p50_ms": statistics.median(elapsed),
        "p95_ms": percentile(elapsed, 95.0),
        "rss_mb": guard.peak_mb,
    }


def benchmark_http(
    name: str,
    url: str,
    payload: dict[str, Any],
    decode: Callable[[Any], list[list[int]]],
    warmups: int,
    runs: int,
    timeout: float,
    pid: int | None,
) -> dict[str, Any]:
    for _ in range(warmups):
        decode(post_json(url, payload, timeout))
    elapsed = []
    vectors = None
    peak_rss = rss_mb(pid)
    for _ in range(runs):
        start = time.perf_counter()
        vectors = decode(post_json(url, payload, timeout))
        elapsed.append((time.perf_counter() - start) * 1000.0)
        current = rss_mb(pid)
        peak_rss = max(peak_rss or 0.0, current or 0.0) or None
    return {
        "name": name,
        "vectors": vectors,
        "load_ms": None,
        "p50_ms": statistics.median(elapsed),
        "p95_ms": percentile(elapsed, 95.0),
        "rss_mb": peak_rss,
    }


def validate_artifacts(model_dir: Path, require_onnx: bool) -> None:
    model_weights = model_dir / "model.safetensors"
    model_index = model_dir / "model.safetensors.index.json"
    required = [
        model_dir / "config.json",
        model_dir / "tokenizer.json",
    ]
    if require_onnx:
        required.append(model_dir / "onnx" / "model.onnx")
    missing = [str(path) for path in required if not path.exists()]
    if not model_weights.exists() and not model_index.exists():
        missing.append(f"{model_weights} or {model_index}")
    if require_onnx and not list((model_dir / "onnx").glob("model.onnx_data*")):
        missing.append(str(model_dir / "onnx" / "model.onnx_data*"))
    if missing:
        raise SystemExit("missing local model artifacts:\n  " + "\n  ".join(missing))


def run_guarded_worker(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    env = offline_environment(cache_dir)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--reference",
        backend,
        "--kind",
        args.kind,
        "--model-dir",
        str(Path(args.model_dir).expanduser().resolve()),
        "--runs",
        str(args.runs),
        "--warmups",
        str(args.warmups),
        "--threads",
        str(args.threads),
        "--input-json",
        json.dumps(args.inputs),
    ]
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(
        mode="w+"
    ) as stderr:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        start = time.monotonic()
        peak = 0.0
        error = None
        while proc.poll() is None:
            current = rss_mb(proc.pid) or 0.0
            peak = max(peak, current)
            if peak > args.max_reference_rss_mb:
                error = (
                    f"{backend} exceeded RSS ceiling: "
                    f"{peak:.1f} > {args.max_reference_rss_mb} MiB"
                )
                proc.send_signal(signal.SIGKILL)
                break
            if time.monotonic() - start > args.timeout:
                error = f"{backend} exceeded timeout: {args.timeout:.1f}s"
                proc.send_signal(signal.SIGKILL)
                break
            time.sleep(0.025)
        proc.wait()
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read()
        errors = stderr.read()
    if error:
        raise RuntimeError(error)
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed ({proc.returncode}):\n{errors or output}")
    marker = output.rfind(RESULT_PREFIX)
    if marker >= 0:
        result = json.loads(output[marker + len(RESULT_PREFIX) :].strip())
        result["rss_mb"] = max(float(result["rss_mb"]), peak)
        return result
    raise RuntimeError(f"{backend} did not emit a result:\n{errors or output}")


def worker_sentencetransformers(args: argparse.Namespace, inputs: Any) -> Any:
    import torch

    torch.set_num_threads(args.threads)
    if args.kind == "standard":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            args.model_dir, trust_remote_code=True, local_files_only=True
        )
        return model, lambda: int8_rows(model.encode(inputs, show_progress_bar=False))

    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    return model, lambda: flatten_contextual(model.encode(inputs, show_progress_bar=False))


def worker_onnx(args: argparse.Namespace, inputs: Any) -> Any:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(
        str(Path(args.model_dir) / "onnx" / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    texts = inputs if args.kind == "standard" else [tokenizer.sep_token.join(x) for x in inputs]
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {
        "input_ids": tokenized["input_ids"].astype(np.int64),
        "attention_mask": tokenized["attention_mask"].astype(np.int64),
    }

    def encode() -> list[list[int]]:
        outputs = session.run(None, ort_inputs)
        if args.kind == "standard":
            return int8_rows(outputs[2])
        rows = []
        for input_ids, mask, hidden in zip(
            ort_inputs["input_ids"], ort_inputs["attention_mask"], outputs[0]
        ):
            start = 0
            end = int(mask.sum())
            for sep in np.flatnonzero((input_ids == tokenizer.sep_token_id) & mask.astype(bool)):
                rows.append(int8_rows([np.rint(np.tanh(hidden[start:sep].mean(axis=0)) * 127)])[0])
                start = int(sep) + 1
            rows.append(int8_rows([np.rint(np.tanh(hidden[start:end].mean(axis=0)) * 127)])[0])
        return rows

    return session, encode


def self_max_rss_mb() -> float:
    import resource

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def worker_main(args: argparse.Namespace) -> int:
    inputs = json.loads(args.input_json)
    start = time.perf_counter()
    if args.reference == "sentence-transformers":
        owner, encode = worker_sentencetransformers(args, inputs)
    elif args.reference == "onnx":
        owner, encode = worker_onnx(args, inputs)
    else:
        raise SystemExit(f"unsupported reference: {args.reference}")
    load_ms = (time.perf_counter() - start) * 1000.0
    for _ in range(args.warmups):
        encode()
    vectors = None
    elapsed = []
    for _ in range(args.runs):
        start = time.perf_counter()
        vectors = encode()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    del owner
    result = {
        "name": args.reference,
        "vectors": vectors,
        "load_ms": load_ms,
        "p50_ms": statistics.median(elapsed),
        "p95_ms": percentile(elapsed, 95.0),
        "rss_mb": self_max_rss_mb(),
    }
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))
    return 0


def print_matrix(name: str, vectors: list[list[int]], row_labels: list[str]) -> None:
    print(f"\n{name} cosine similarity matrix")
    print(" " * 16 + " ".join(f"{label:>14}" for label in row_labels))
    for label, row in zip(row_labels, vectors):
        scores = " ".join(f"{cosine(row, other):14.6f}" for other in vectors)
        print(f"{label:<16}{scores}")


def print_summary(results: list[dict[str, Any]], row_labels: list[str]) -> None:
    print("\nPerformance")
    print(f"{'backend':<24} {'load ms':>10} {'p50 ms':>10} {'p95 ms':>10} {'peak RSS MiB':>14}")
    for result in results:
        load = "-" if result["load_ms"] is None else f"{result['load_ms']:.1f}"
        rss = "-" if result["rss_mb"] is None else f"{result['rss_mb']:.1f}"
        print(
            f"{result['name']:<24} {load:>10} {result['p50_ms']:>10.1f} "
            f"{result['p95_ms']:>10.1f} {rss:>14}"
        )

    print("\nParity")
    baseline = results[0]
    for result in results[1:]:
        if len(result["vectors"]) != len(baseline["vectors"]):
            raise RuntimeError(
                f"{result['name']} returned {len(result['vectors'])} vectors; "
                f"expected {len(baseline['vectors'])}"
            )
        pairs = list(zip(baseline["vectors"], result["vectors"]))
        if any(len(a) != len(b) for a, b in pairs):
            raise RuntimeError(f"{result['name']} returned an unexpected dimension")
        scores = [cosine(a, b) for a, b in pairs]
        diffs = [max(abs(x - y) for x, y in zip(a, b)) for a, b in pairs]
        print(
            f"{baseline['name']} vs {result['name']}: "
            f"min_cosine={min(scores):.8f} mean_cosine={statistics.mean(scores):.8f} "
            f"max_int8_diff={max(diffs)}"
        )


def load_report(path: str, kind: str, inputs: Any) -> list[dict[str, Any]]:
    report = json.loads(Path(path).read_text())
    if report.get("kind") != kind or report.get("inputs") != inputs:
        raise RuntimeError("baseline JSON does not match the selected kind and inputs")
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("baseline JSON does not contain results")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model", default="pplx-embed-v1-0.6b")
    ap.add_argument("--kind", choices=("standard", "contextual"), default="standard")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--native-mode", choices=("http", "stdin", "none"), default="http")
    ap.add_argument("--native-backend", choices=("cpu", "mlx"), default="mlx")
    ap.add_argument("--native-binary", default=str(repo_root() / "ffwd-cli"))
    ap.add_argument("--max-native-rss-mb", type=float, default=32768.0)
    ap.add_argument("--skip-native", action="store_true")
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--tei-url", help="optional TEI base URL; standard mode only")
    ap.add_argument("--tei-pid", type=int)
    ap.add_argument("--references", default="sentence-transformers,onnx")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-reference-rss-mb", type=float, default=6144.0)
    ap.add_argument(
        "--cache-dir", default=str(repo_root() / ".venv" / "reference-cache")
    )
    ap.add_argument("--input-json")
    ap.add_argument("--baseline-json", help="prepend results from an earlier sequential run")
    ap.add_argument("--json-out", help="write inputs and measured results for a later run")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--reference", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.runs < 1 or args.warmups < 0 or args.threads < 1:
        raise SystemExit("runs and threads must be positive; warmups must be non-negative")
    if args.worker:
        return worker_main(args)
    if args.tei_url and args.kind != "standard":
        raise SystemExit("TEI comparison currently supports --kind standard only")
    if args.skip_native:
        args.native_mode = "none"

    args.inputs = json.loads(args.input_json) if args.input_json else (
        DEFAULT_STANDARD if args.kind == "standard" else DEFAULT_CONTEXTUAL
    )
    references = [x.strip() for x in args.references.split(",") if x.strip()]
    validate_artifacts(Path(args.model_dir).expanduser().resolve(), "onnx" in references)
    row_labels = labels(args.kind, args.inputs)
    results = load_report(args.baseline_json, args.kind, args.inputs) if args.baseline_json else []

    if args.native_mode == "stdin":
        results.append(benchmark_native_stdin(args))
    elif args.native_mode == "http":
        endpoint = "/v1/embeddings" if args.kind == "standard" else "/v1/contextualizedembeddings"
        results.append(
            benchmark_http(
                "native-http",
                args.base_url.rstrip("/") + endpoint,
                {"model": args.model, "input": args.inputs},
                lambda response: decode_native(response, args.kind),
                args.warmups,
                args.runs,
                args.timeout,
                args.server_pid,
            )
        )
    for backend in references:
        if backend not in ("sentence-transformers", "onnx"):
            raise SystemExit(f"unsupported reference backend: {backend}")
        results.append(run_guarded_worker(args, backend))
    if args.tei_url:
        results.append(
            benchmark_http(
                "tei-http",
                args.tei_url.rstrip("/") + "/embed",
                {"inputs": args.inputs, "normalize": False},
                int8_rows,
                args.warmups,
                args.runs,
                args.timeout,
                args.tei_pid,
            )
        )
    if not results:
        raise SystemExit("select at least one implementation to measure")

    print(f"kind={args.kind} vectors={len(row_labels)} runs={args.runs} warmups={args.warmups}")
    for result in results:
        print_matrix(result["name"], result["vectors"], row_labels)
    print_summary(results, row_labels)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"kind": args.kind, "inputs": args.inputs, "results": results},
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
