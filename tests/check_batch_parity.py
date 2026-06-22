#!/usr/bin/env python3
"""Compare stdin batch_size=1 against true batched stdin execution."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "ffwd-cli"


def ensure_cli(binary: Path) -> None:
    """Build the CLI if the default binary is missing."""
    if binary == DEFAULT_BINARY and not binary.exists():
        subprocess.run(["make", "cpu"], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)


DEFAULT_TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital of France.",
    "document: Berlin is the capital of Germany.",
    "document: Istanbul is a major city in Turkey.",
    "A short string.",
    (
        "Embedding models convert text into dense vectors for retrieval, "
        "clustering, and semantic similarity search."
    ),
]

LONG_TEXTS = [
    "document: " + (
        "Embedding models turn text into dense vectors for semantic retrieval. "
        "Long document inference stresses bidirectional attention because every "
        "token attends to every other token in the same packed sequence. "
    ) * repeats
    for repeats in (8, 10, 12, 14)
]


def detect_pooling_mode(model_dir: str) -> str:
    """Return the sentence pooling mode used by ffwd_config_load, if obvious."""
    root = Path(model_dir)
    pooling_path = root / "1_Pooling" / "config.json"
    if pooling_path.exists():
        try:
            cfg = json.loads(pooling_path.read_text())
        except (OSError, json.JSONDecodeError):
            return "unknown"
        if cfg.get("pooling_mode_lasttoken"):
            return "last-token"
        if cfg.get("pooling_mode_cls_token"):
            return "cls"
        if cfg.get("pooling_mode_mean_tokens"):
            return "mean"
        return "unknown"

    config_path = root / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return "unknown"
        if cfg.get("model_type") == "qwen3":
            return "last-token"

    return "unknown"


def run_stdin(binary, model_dir, backend, batch_size, texts,
              mlx_quantize_bits, mlx_quantize_group_size,
              cuda_weight_dtype=None):
    cmd = [binary, "-d", model_dir, "--stream", "-b", str(batch_size)]
    if backend == "mlx":
        if mlx_quantize_bits:
            cmd.extend([
                "--gpu-quant-bits", str(mlx_quantize_bits),
                "--gpu-quant-group-size", str(mlx_quantize_group_size),
            ])
    elif backend == "cuda":
        # Force exact-F32 weight storage so a BF16 snapshot is judged for kernel
        # exactness (batched vs singleton GEMMs) rather than BF16 rounding.
        if cuda_weight_dtype:
            cmd.extend(["--gpu-weight-dtype", cuda_weight_dtype])

    proc = subprocess.run(
        cmd,
        input="\n".join(texts) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )

    rows = []
    for line in proc.stdout.splitlines():
        if line.strip():
            row = json.loads(line)
            if "error" in row:
                raise RuntimeError(f"stdin returned error: {row}")
            rows.append(row)

    if len(rows) != len(texts):
        raise RuntimeError(f"expected {len(texts)} rows, got {len(rows)}")
    return rows


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--backend", choices=("cpu", "mlx", "cuda"), default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-diff", type=float, default=5e-5)
    ap.add_argument("--min-cosine", type=float, default=0.99999)
    ap.add_argument("--cuda-last-token-max-diff", type=float, default=0.002,
                    help=("effective max-abs gate for CUDA last-token pooling; "
                          "keeps cosine strict while allowing batched-GEMM "
                          "float reordering"))
    ap.add_argument("--cuda-last-token-min-cosine", type=float, default=0.9999,
                    help="effective cosine gate for CUDA last-token pooling")
    ap.add_argument("--long", action="store_true",
                    help="use long inputs that exercise tiled CPU attention")
    ap.add_argument("--mlx-quant-bits", "--mlx-quantize-bits",
                    dest="mlx_quantize_bits", type=int, default=0,
                    choices=(0, 8))
    ap.add_argument("--mlx-quant-group-size", "--mlx-quantize-group-size",
                    dest="mlx_quantize_group_size", type=int, default=64)
    ap.add_argument("--cuda-weight-dtype", choices=("f32", "bf16"), default=None,
                    help="force CUDA weight storage dtype (default: snapshot dtype)")
    ap.add_argument("texts", nargs="*")
    args = ap.parse_args()

    texts = args.texts or (LONG_TEXTS if args.long else DEFAULT_TEXTS)
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be >= 2 for a useful parity check")
    if args.mlx_quantize_bits and args.backend != "mlx":
        raise SystemExit("--mlx-quant-bits requires --backend mlx")
    if args.mlx_quantize_group_size <= 0:
        raise SystemExit("--mlx-quant-group-size must be > 0")
    if args.cuda_last_token_max_diff <= 0:
        raise SystemExit("--cuda-last-token-max-diff must be > 0")
    if not 0.0 < args.cuda_last_token_min_cosine <= 1.0:
        raise SystemExit("--cuda-last-token-min-cosine must be in (0, 1]")

    ensure_cli(Path(args.binary))
    pooling_mode = detect_pooling_mode(args.model_dir)
    effective_max_diff = args.max_diff
    effective_min_cosine = args.min_cosine
    calibrated_cuda_last_token = args.backend == "cuda" and pooling_mode == "last-token"
    if calibrated_cuda_last_token and effective_max_diff < args.cuda_last_token_max_diff:
        effective_max_diff = args.cuda_last_token_max_diff
    if calibrated_cuda_last_token and effective_min_cosine > args.cuda_last_token_min_cosine:
        effective_min_cosine = args.cuda_last_token_min_cosine

    seq_rows = run_stdin(args.binary, args.model_dir, args.backend, 1, texts,
                         args.mlx_quantize_bits,
                         args.mlx_quantize_group_size,
                         args.cuda_weight_dtype)
    bat_rows = run_stdin(args.binary, args.model_dir, args.backend,
                         args.batch_size, texts, args.mlx_quantize_bits,
                         args.mlx_quantize_group_size,
                         args.cuda_weight_dtype)

    worst_diff = 0.0
    worst_cos = 1.0
    for i, (seq, bat) in enumerate(zip(seq_rows, bat_rows)):
        if seq["dim"] != bat["dim"] or seq["tokens"] != bat["tokens"]:
            raise RuntimeError(f"metadata mismatch at row {i}: {seq} vs {bat}")
        a = seq["embedding"]
        b = bat["embedding"]
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        cos = cosine(a, b)
        worst_diff = max(worst_diff, max_diff)
        worst_cos = min(worst_cos, cos)
        print(
            f"[{i}] tokens={seq['tokens']} cosine={cos:.8f} "
            f"max_abs_diff={max_diff:.8g}"
        )

    if worst_diff > effective_max_diff or worst_cos < effective_min_cosine:
        raise RuntimeError(
            f"batch parity failed: worst_diff={worst_diff:g}, worst_cos={worst_cos:g}, "
            f"max_diff_gate={effective_max_diff:g}, min_cosine_gate={effective_min_cosine:g}"
        )

    calibration_note = ""
    if calibrated_cuda_last_token:
        calibration_note = (f", pooling={pooling_mode}, max_diff_gate={effective_max_diff:g}, "
                            f"min_cosine_gate={effective_min_cosine:g}")
    seq_ms = sum(float(row["ms"]) for row in seq_rows)
    batch_ms = sum(float(bat_rows[i]["ms"]) for i in range(0, len(bat_rows), args.batch_size))
    print(
        f"ok: {len(texts)} texts, backend={args.backend}, "
        f"batch_size={args.batch_size}, worst_diff={worst_diff:.8g}, "
        f"worst_cos={worst_cos:.8f}, seq_ms_sum={seq_ms:.1f}, "
        f"batch_chunk_ms_sum={batch_ms:.1f}"
        + calibration_note
        + (
            f", mlx_q{args.mlx_quantize_bits}"
            if args.mlx_quantize_bits else ""
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
