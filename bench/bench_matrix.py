#!/usr/bin/env python3
"""Batch-aware benchmark matrix for the stdin backends."""

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each backend is its own binary now: `make <backend>` builds the CLI into
# build/release/<dir>/ffwd-cli (and refreshes the ./ffwd-cli root symlink).
BUILD_DIR = {"cpu": "blas", "mlx": "mlx", "cuda": "cuda"}


def default_binary(backend):
    return ROOT / "build" / "release" / BUILD_DIR[backend] / "ffwd-cli"


def resolve_binary(backend, args):
    """Pick the CLI for one backend: an explicit --<backend>-binary, then the
    shared --binary override, then the per-backend build-tree default."""
    explicit = {"cpu": args.cpu_binary, "mlx": args.mlx_binary,
                "cuda": args.cuda_binary}.get(backend, "")
    if explicit:
        return Path(explicit)
    if args.binary:
        return Path(args.binary)
    return default_binary(backend)


def ensure_binary(backend, binary):
    """Build the backend's CLI on demand when using the build-tree default."""
    if binary == default_binary(backend) and not binary.exists():
        subprocess.run(["make", backend], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    if not binary.exists():
        raise SystemExit(f"binary not found for backend {backend}: {binary}")


try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


# A case is a target token count (an integer). The input is synthesized to about
# that many common words (~1 token each); the tokenizer reports the actual count
# per run, so the case is an explicit, approximate length target.
_FILLER = (
    "embedding models convert text into dense vectors that capture semantic "
    "meaning for search clustering classification and retrieval augmented "
    "generation across many production systems languages and domains where "
    "latency throughput memory bandwidth and matrix multiplication all matter"
).split()


def make_text(n_tokens):
    n = max(1, int(n_tokens))
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(n))


def parse_csv(value, cast=str):
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            items.append(cast(raw))
    return items


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(values):
    return {
        "min_ms": min(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
    }


def make_batch_texts(n_tokens, batch_size):
    return [make_text(n_tokens) for _ in range(batch_size)]


class StdinRunner:
    def __init__(self, binary, model_dir, backend, batch_size, threads,
                 binary_args=None, backend_args=None):
        self.binary = binary
        self.model_dir = model_dir
        self.backend = backend
        self.batch_size = batch_size
        self.threads = threads
        self.binary_args = list(binary_args or [])
        self.backend_args = list(backend_args or [])
        self.proc = None
        self.ps = None
        self.load_ms = None

    def start(self):
        cmd = [self.binary, "-d", self.model_dir,
               *self.binary_args, *self.backend_args,
               "--stream", "-b", str(self.batch_size)]
        if self.backend == "cpu" and self.threads:
            cmd.extend(["-t", str(self.threads)])

        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if psutil:
            self.ps = psutil.Process(self.proc.pid)

        warmup = make_batch_texts(16, self.batch_size)
        self.send_batch(warmup)
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    def rss_mb(self):
        if self.ps:
            try:
                return self.ps.memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        if not self.proc:
            return None
        try:
            proc = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(self.proc.pid)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            value = proc.stdout.strip() if proc.returncode == 0 else ""
            return (float(value) / 1024.0) if value else None
        except Exception:
            return None

    def send_batch(self, texts):
        if len(texts) != self.batch_size:
            raise ValueError("send_batch requires exactly batch_size texts")
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("stdin is not running")

        t0 = time.perf_counter()
        self.proc.stdin.write("\n".join(texts) + "\n")
        self.proc.stdin.flush()

        rows = []
        for _ in texts:
            line = self.proc.stdout.readline()
            if line == "":
                stderr = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"stdin exited early\n{stderr}")
            row = json.loads(line)
            if "error" in row:
                raise RuntimeError(f"stdin returned error: {row}")
            rows.append(row)

        wall_ms = (time.perf_counter() - t0) * 1000.0
        engine_ms = float(rows[0]["ms"])
        return rows, engine_ms, wall_ms

    def stop(self):
        if not self.proc:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        finally:
            self.proc = None


def read_model_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "intermediate_size": cfg.get("intermediate_size"),
        "vocab_size": cfg.get("vocab_size"),
    }


def run_case(stdin_runner, case_name, runs, warmups):
    texts = make_batch_texts(case_name, stdin_runner.batch_size)

    for _ in range(warmups):
        stdin_runner.send_batch(texts)

    engine_times = []
    wall_times = []
    tokens_per_text = None
    dim = None
    workspace_bytes = None
    for _ in range(runs):
        rows, engine_ms, wall_ms = stdin_runner.send_batch(texts)
        engine_times.append(engine_ms)
        wall_times.append(wall_ms)
        tokens_per_text = [int(row["tokens"]) for row in rows]
        dim = int(rows[0]["dim"])
        row_ws = [int(row.get("workspace_bytes", 0)) for row in rows]
        if row_ws:
            workspace_bytes = max(workspace_bytes or 0, max(row_ws))

    tokens_per_batch = sum(tokens_per_text)
    stats = summarize(engine_times)
    p50 = stats["p50_ms"]
    return {
        "case": case_name,
        "batch_size": stdin_runner.batch_size,
        "tokens_per_text": tokens_per_text,
        "tokens_per_batch": tokens_per_batch,
        "dim": dim,
        "workspace_bytes": workspace_bytes,
        "engine_ms": engine_times,
        "wall_ms": wall_times,
        **stats,
        "wall_p50_ms": percentile(wall_times, 50),
        "texts_per_sec_p50": (1000.0 * stdin_runner.batch_size / p50) if p50 else None,
        "tokens_per_sec_p50": (1000.0 * tokens_per_batch / p50) if p50 else None,
    }


def format_num(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def print_table(results):
    headers = [
        "backend", "B", "case", "tokens", "p50", "p90", "p95",
        "txt/s", "tok/s", "RSS MB", "WS MB",
    ]
    rows = []
    for r in results:
        token_desc = (
            str(r["tokens_per_text"][0])
            if len(set(r["tokens_per_text"])) == 1
            else f"{min(r['tokens_per_text'])}-{max(r['tokens_per_text'])}"
        )
        rows.append([
            r["backend"],
            str(r["batch_size"]),
            str(r["case"]),
            token_desc,
            format_num(r["p50_ms"]),
            format_num(r["p90_ms"]),
            format_num(r["p95_ms"]),
            format_num(r["texts_per_sec_p50"]),
            format_num(r["tokens_per_sec_p50"], 0),
            format_num(r.get("rss_mb"), 0),
            format_num((r["workspace_bytes"] / (1024 * 1024))
                       if r.get("workspace_bytes") else None, 1),
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def markdown_table(results):
    headers = [
        "Backend", "Batch", "Case", "Tokens", "P50 ms", "P90 ms",
        "P95 ms", "Texts/s", "Tokens/s", "RSS MB", "Workspace MB",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        token_desc = (
            str(r["tokens_per_text"][0])
            if len(set(r["tokens_per_text"])) == 1
            else f"{min(r['tokens_per_text'])}-{max(r['tokens_per_text'])}"
        )
        cells = [
            r["backend"],
            str(r["batch_size"]),
            str(r["case"]),
            token_desc,
            format_num(r["p50_ms"]),
            format_num(r["p90_ms"]),
            format_num(r["p95_ms"]),
            format_num(r["texts_per_sec_p50"]),
            format_num(r["tokens_per_sec_p50"], 0),
            format_num(r.get("rss_mb"), 0),
            format_num((r["workspace_bytes"] / (1024 * 1024))
                       if r.get("workspace_bytes") else None, 1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--binary", default="",
                    help="single CLI for every backend; overrides the per-backend default")
    ap.add_argument("--cpu-binary", default="", help="CLI for cpu runs")
    ap.add_argument("--mlx-binary", default="", help="CLI for mlx runs")
    ap.add_argument("--cuda-binary", default="", help="CLI for cuda runs")
    ap.add_argument("--backends", default="cpu",
                    help="Comma-separated list: cpu,mlx,cuda")
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--cases", default="48,80,140,256,384,512",
                    help="target token counts, e.g. 48,256,512")
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU thread count passed to -t; 0 uses binary default")
    ap.add_argument("--binary-args", default="",
                    help="extra arguments passed to every backend binary")
    ap.add_argument("--cpu-binary-args", default="",
                    help="extra arguments passed only to CPU backend runs")
    ap.add_argument("--mlx-binary-args", default="",
                    help="extra arguments passed only to MLX backend runs")
    ap.add_argument("--cuda-binary-args", default="",
                    help="extra arguments passed only to CUDA backend runs")
    ap.add_argument("--json-out")
    ap.add_argument("--markdown-out")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    backends = parse_csv(args.backends)
    batch_sizes = parse_csv(args.batch_sizes, int)
    cases = parse_csv(args.cases, int)
    common_binary_args = shlex.split(args.binary_args)
    backend_binary_args = {
        "cpu": shlex.split(args.cpu_binary_args),
        "mlx": shlex.split(args.mlx_binary_args),
        "cuda": shlex.split(args.cuda_binary_args),
    }

    if args.runs <= 0 or args.warmups < 0:
        raise SystemExit("--runs must be > 0 and --warmups must be >= 0")
    for case in cases:
        if case <= 0:
            raise SystemExit(f"case token count must be a positive integer, got: {case}")
    for backend in backends:
        if backend not in BUILD_DIR:
            raise SystemExit(f"unknown backend: {backend} (expected cpu, mlx, or cuda)")

    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "binary": args.binary or "per-backend (build/release/<backend>/ffwd-cli)",
        "binaries": {b: str(resolve_binary(b, args)) for b in backends},
        "model_dir": os.path.abspath(args.model_dir),
        "model": read_model_config(args.model_dir),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "config": {
            "backends": backends,
            "batch_sizes": batch_sizes,
            "cases": cases,
            "runs": args.runs,
            "warmups": args.warmups,
            "threads": args.threads,
        },
        "results": [],
        "errors": [],
    }

    for backend in backends:
        binary = resolve_binary(backend, args)
        ensure_binary(backend, binary)
        for batch_size in batch_sizes:
            print(f"\n== backend={backend} batch={batch_size} binary={binary} ==")
            stdin_runner = StdinRunner(
                str(binary), args.model_dir, backend, batch_size, args.threads,
                common_binary_args, backend_binary_args.get(backend, []))
            try:
                stdin_runner.start()
                rss_after_warmup = stdin_runner.rss_mb()
                print(
                    f"load+first_batch={stdin_runner.load_ms:.1f} ms"
                    + (f", rss={rss_after_warmup:.0f} MB" if rss_after_warmup else "")
                )
                for case in cases:
                    r = run_case(stdin_runner, case, args.runs, args.warmups)
                    r.update({
                        "backend": backend,
                        "binary": str(binary),
                        "load_ms": stdin_runner.load_ms,
                        "rss_mb": stdin_runner.rss_mb() or rss_after_warmup,
                    })
                    report["results"].append(r)
                    print(
                        f"  {case:<7} tokens={r['tokens_per_batch']:>4} "
                        f"p50={r['p50_ms']:.1f} ms "
                        f"p95={r['p95_ms']:.1f} ms "
                        f"txt/s={r['texts_per_sec_p50']:.1f}"
                        + (f" ws={r['workspace_bytes'] / (1024 * 1024):.1f} MB"
                           if r.get("workspace_bytes") else "")
                    )
            except Exception as exc:
                err = {"backend": backend, "batch_size": batch_size, "error": str(exc)}
                report["errors"].append(err)
                print(f"error: {backend} batch={batch_size}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    stdin_runner.stop()
                    raise
            finally:
                stdin_runner.stop()

    print("\nSummary")
    print_table(report["results"])

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nWrote {args.json_out}")

    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write("# ffwd benchmark matrix\n\n")
            f.write(f"- Created: `{report['created_at']}`\n")
            f.write(f"- Binary: `{report['binary']}`\n")
            f.write(f"- Model: `{report['model_dir']}`\n")
            f.write(f"- Host: `{report['host']['platform']}`\n\n")
            f.write(markdown_table(report["results"]))
            if report["errors"]:
                f.write("\n## Errors\n\n")
                for err in report["errors"]:
                    f.write(f"- `{err['backend']}` batch `{err['batch_size']}`: {err['error']}\n")
        print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        sys.exit(1)
