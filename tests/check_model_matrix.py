#!/usr/bin/env python3
"""Smoke-check supported pplx-embed-v1 model variants by model kind."""

from __future__ import annotations

import argparse
import shlex
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    kind: str
    hub_repo: str
    fallbacks: tuple[Path, ...] = ()


MODEL_SPECS = {
    "v1-0.6b": ModelSpec(
        key="v1-0.6b",
        model_id="pplx-embed-v1-0.6b",
        kind="standard",
        hub_repo="models--perplexity-ai--pplx-embed-v1-0.6b",
    ),
    "v1-4b": ModelSpec(
        key="v1-4b",
        model_id="pplx-embed-v1-4b",
        kind="standard",
        hub_repo="models--perplexity-ai--pplx-embed-v1-4b",
        fallbacks=(Path("/private/tmp/pplx-embed-v1-4b-bf16"),),
    ),
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        model_id="Qwen3-Embedding-0.6B",
        kind="qwen3",
        hub_repo="models--Qwen--Qwen3-Embedding-0.6B",
    ),
    "context-v1-0.6b": ModelSpec(
        key="context-v1-0.6b",
        model_id="pplx-embed-context-v1-0.6b",
        kind="contextual",
        hub_repo="models--perplexity-ai--pplx-embed-context-v1-0.6B",
    ),
    "context-v1-4b": ModelSpec(
        key="context-v1-4b",
        model_id="pplx-embed-context-v1-4b",
        kind="contextual",
        hub_repo="models--perplexity-ai--pplx-embed-context-v1-4B",
        fallbacks=(Path("/private/tmp/pplx-embed-context-v1-4b-bf16"),),
    ),
}

ALIASES = {
    "context-v1-0.6B": "context-v1-0.6b",
    "context-v1-4B": "context-v1-4b",
    "pplx-embed-v1-0.6b": "v1-0.6b",
    "pplx-embed-v1-4b": "v1-4b",
    "pplx-embed-context-v1-0.6b": "context-v1-0.6b",
    "pplx-embed-context-v1-4b": "context-v1-4b",
    "Qwen3-Embedding-0.6B": "qwen3-0.6b",
}

DEFAULT_MODELS = (
    "v1-0.6b",
    "v1-4b",
    "qwen3-0.6b",
    "context-v1-0.6b",
    "context-v1-4b",
)

STANDARD_TEXTS = [
    "query: capital of France?",
    "document: Paris is the capital of France.",
    "document: Berlin is the capital of Germany.",
    "document: Istanbul is a city in Turkey.",
]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def canonical_model_key(key: str) -> str:
    key = ALIASES.get(key, key)
    if key not in MODEL_SPECS:
        known = ", ".join(MODEL_SPECS)
        raise SystemExit(f"unknown model key: {key}; known: {known}")
    return key


def parse_model_overrides(values: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit("--model expects KEY=PATH")
        key, path = value.split("=", 1)
        key = canonical_model_key(key.strip())
        if not path.strip():
            raise SystemExit("--model expects a non-empty PATH")
        overrides[key] = Path(path).expanduser().resolve()
    return overrides


def has_config(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def cached_snapshot(spec: ModelSpec) -> Path | None:
    snapshots = HF_CACHE / spec.hub_repo / "snapshots"
    if not snapshots.is_dir():
        return None
    dirs = sorted(p for p in snapshots.iterdir() if has_config(p))
    return dirs[-1] if dirs else None


def resolve_model_dir(spec: ModelSpec, overrides: dict[str, Path]) -> Path:
    if spec.key in overrides:
        path = overrides[spec.key]
        if not has_config(path):
            raise FileNotFoundError(f"{spec.key}: missing config.json in {path}")
        return path
    for fallback in spec.fallbacks:
        if has_config(fallback):
            return fallback
    path = cached_snapshot(spec)
    if path:
        return path
    raise FileNotFoundError(f"{spec.key}: no cached snapshot or fallback found")


def read_config(model_dir: Path) -> dict[str, int]:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "hidden": int(cfg["hidden_size"]),
        "layers": int(cfg["num_hidden_layers"]),
        "heads": int(cfg["num_attention_heads"]),
        "kv_heads": int(cfg["num_key_value_heads"]),
        "intermediate": int(cfg["intermediate_size"]),
    }


def is_bf16_path(path: Path) -> bool:
    return "bf16" in str(path).lower()


def tolerances(args: argparse.Namespace, model_dir: Path) -> tuple[float, float]:
    if is_bf16_path(model_dir):
        return args.bf16_max_diff, args.bf16_min_cosine
    return args.max_diff, args.min_cosine


def standard_cmd(args: argparse.Namespace, model_dir: Path,
                 backend: str, max_diff: float, min_cosine: float) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tests" / "check_batch_parity.py"),
        "--model-dir", str(model_dir),
        "--binary", args.binary,
        "--backend", backend,
        "--batch-size", str(args.batch_size),
        "--max-diff", str(max_diff),
        "--min-cosine", str(min_cosine),
        *STANDARD_TEXTS,
    ]
    if backend == "mlx" and args.mlx_quantize_bits:
        insert_at = len(cmd) - len(STANDARD_TEXTS)
        cmd[insert_at:insert_at] = [
            "--mlx-quant-bits", str(args.mlx_quantize_bits),
            "--mlx-quant-group-size", str(args.mlx_quantize_group_size),
        ]
    return cmd


def contextual_cmd(args: argparse.Namespace, model_dir: Path,
                   backend: str, max_diff: float,
                   min_cosine: float) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tests" / "check_contextual_batch_parity.py"),
        "--model-dir", str(model_dir),
        "--backend", backend,
        "--runs", str(args.context_runs),
        "--max-diff", str(max_diff),
        "--min-cosine", str(min_cosine),
    ]
    if backend == "mlx" and args.mlx_quantize_bits:
        cmd.extend([
            "--mlx-quant-bits", str(args.mlx_quantize_bits),
            "--mlx-quant-group-size", str(args.mlx_quantize_group_size),
        ])
    return cmd


def qwen3_cmd(args: argparse.Namespace, model_dir: Path,
              backend: str) -> list[str]:
    # check_qwen3_parity.py runs the CLI (cpu); it has no --backend flag.
    return [
        sys.executable,
        str(ROOT / "tests" / "check_qwen3_parity.py"),
        "--model-dir", str(model_dir),
        "--binary", args.binary,
        "--batch-size", str(args.batch_size),
    ]


def run_check(cmd: list[str], args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    if args.dry_run:
        print("  + " + " ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def last_output_line(proc: subprocess.CompletedProcess[str]) -> str:
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def print_failure(backend: str, proc: subprocess.CompletedProcess[str]) -> None:
    print(f"  {backend}: FAIL", file=sys.stderr)
    print("  command: " + shlex.join([str(x) for x in proc.args]), file=sys.stderr)
    if proc.stdout:
        print(proc.stdout, file=sys.stderr)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", default=str(ROOT / "ffwd-cli"))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated model keys")
    ap.add_argument("--model", action="append",
                    help="Override a model path: KEY=PATH")
    ap.add_argument("--backends", default="cpu",
                    help="Comma-separated backends: cpu,mlx")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--context-runs", type=int, default=0)
    ap.add_argument("--max-diff", type=float, default=5e-5)
    ap.add_argument("--min-cosine", type=float, default=0.99999)
    ap.add_argument("--bf16-max-diff", type=float, default=0.03)
    ap.add_argument("--bf16-min-cosine", type=float, default=0.9998)
    ap.add_argument("--mlx-quant-bits", "--mlx-quantize-bits",
                    dest="mlx_quantize_bits", type=int, default=0,
                    choices=(0, 8))
    ap.add_argument("--mlx-quant-group-size", "--mlx-quantize-group-size",
                    dest="mlx_quantize_group_size", type=int, default=64)
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for key, spec in MODEL_SPECS.items():
            print(f"{key:16} {spec.kind:10} {spec.model_id}")
        return 0
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be >= 2")
    if args.context_runs < 0:
        raise SystemExit("--context-runs must be >= 0")
    if args.mlx_quantize_group_size <= 0:
        raise SystemExit("--mlx-quant-group-size must be > 0")

    overrides = parse_model_overrides(args.model)
    model_keys = [canonical_model_key(key) for key in parse_csv(args.models)]
    backends = parse_csv(args.backends)
    for backend in backends:
        if backend not in ("cpu", "mlx"):
            raise SystemExit(f"unknown backend: {backend}")
    if args.mlx_quantize_bits and any(b != "mlx" for b in backends):
        raise SystemExit("--mlx-quant-bits requires --backends mlx")

    ok = 0
    skipped = 0
    failed = 0
    seen: set[str] = set()
    for model_key in model_keys:
        if model_key in seen:
            continue
        seen.add(model_key)
        spec = MODEL_SPECS[model_key]
        try:
            model_dir = resolve_model_dir(spec, overrides)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"\n== {spec.key} {spec.kind}: SKIP ({exc}) ==")
                skipped += len(backends)
                continue
            raise SystemExit(str(exc))

        cfg = read_config(model_dir)
        max_diff, min_cosine = tolerances(args, model_dir)
        print(
            f"\n== {spec.key} {spec.kind} hidden={cfg['hidden']} "
            f"layers={cfg['layers']} heads={cfg['heads']}/{cfg['kv_heads']} =="
        )
        print(model_dir)
        if is_bf16_path(model_dir):
            print(f"tolerance: max_diff={max_diff:g} min_cosine={min_cosine:g}")
        if args.mlx_quantize_bits:
            print(
                f"mlx quantization: q{args.mlx_quantize_bits} "
                f"group_size={args.mlx_quantize_group_size}"
            )

        for backend in backends:
            if spec.kind == "standard":
                cmd = standard_cmd(
                    args, model_dir, backend, max_diff, min_cosine
                )
            elif spec.kind == "qwen3":
                cmd = qwen3_cmd(args, model_dir, backend)
            else:
                cmd = contextual_cmd(
                    args, model_dir, backend, max_diff, min_cosine
                )
            proc = run_check(cmd, args)
            if proc.returncode == 0:
                print(f"  {backend}: {last_output_line(proc)}")
                ok += 1
            else:
                print_failure(backend, proc)
                failed += 1
                if not args.continue_on_error:
                    return 1

    summary = f"\nok: {ok} checks passed"
    if skipped:
        summary += f", skipped={skipped}"
    if failed:
        summary += f", failed={failed}"
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
