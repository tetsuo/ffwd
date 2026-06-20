#!/usr/bin/env python3
"""Compare f32 -> low-precision int8-output drift across engines.

The native BF16 path changes ~10-15% of int8 output bytes versus F32 (while
keeping cosine ~0.9999 and identical retrieval). This script checks whether that
is normal by measuring the *same* f32 -> low-precision drift in independent
implementations of the same model:

  - ONNX    : released model.onnx (F32) vs model_quantized.onnx (official int8)
  - HF      : sentence-transformers / torch, loaded in F32 vs BF16
  - native  : ffwd binary, an F32 model dir vs a BF16 model dir

For every engine it reports, per low-precision variant against that engine's own
F32 output: the int8 byte-change rate, the int8-vector cosine, and the float
relative-L2 diff. If ONNX-int8 and HF-bf16 drift like ours, the drift is inherent
to low precision, not a backend defect.

Reference engines output int8-scale embeddings (round); native outputs raw
floats quantized with tanh*127 (matching the server's base64_int8 encoder).

Needs the reference stack, so run it with uv (Python 3.12 for the numpy pin):

  uv run --python 3.12 --with-requirements requirements-reference.txt \
      devtools/quant_precision_drift.py --model-dir DIR --engines hf
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "tools/cli/ffwd-cli"


DEFAULT_TEXTS = [
    "query: what is the capital of France?",
    "document: Paris is the capital of France.",
    "document: Berlin is the capital of Germany.",
    "document: The Eiffel Tower is a famous landmark located in Paris, France.",
    "document: Photosynthesis converts sunlight into chemical energy in plants.",
    (
        "document: Embedding models turn text into dense vectors for semantic "
        "retrieval. Long document inference stresses bidirectional attention "
        "because every token attends to every other token in the same packed "
        "sequence. "
    ) * 8,
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return dot / (na * nb)


def rel_l2(a: list[float], b: list[float]) -> float:
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(x * x for x in a))
    return num / den if den else 0.0


def drift(ref: list[list[int]], low: list[list[int]]) -> dict[str, float]:
    if len(ref) != len(low):
        raise RuntimeError(f"row count mismatch: {len(ref)} vs {len(low)}")
    total = 0
    changed = 0
    worst_cos = 1.0
    worst_l2 = 0.0
    for a, b in zip(ref, low):
        if len(a) != len(b):
            raise RuntimeError(f"dim mismatch: {len(a)} vs {len(b)}")
        total += len(a)
        changed += sum(1 for x, y in zip(a, b) if x != y)
        worst_cos = min(worst_cos, cosine([float(x) for x in a],
                                          [float(y) for y in b]))
        worst_l2 = max(worst_l2, rel_l2([float(x) for x in a],
                                        [float(y) for y in b]))
    return {
        "int8_change_rate": changed / total if total else 0.0,
        "worst_int8_cosine": worst_cos,
        "worst_rel_l2": worst_l2,
    }


# --- engines -------------------------------------------------------------

def onnx_int8(model_dir: str, onnx_file: str, texts: list[str]) -> list[list[int]]:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,
                                        local_files_only=True)
    sess = ort.InferenceSession(str(Path(model_dir) / "onnx" / onnx_file),
                                providers=["CPUExecutionProvider"])
    t = tok(texts, padding=True, truncation=True, return_tensors="np")
    feed = {"input_ids": t["input_ids"].astype(np.int64),
            "attention_mask": t["attention_mask"].astype(np.int64)}
    # outputs: [0]=hidden states, [2]=pooled int8-scale embedding (see
    # compare_reference_backends.worker_onnx).
    out = sess.run(None, feed)[2]
    return [[int(round(float(x))) for x in row] for row in out]


def hf_int8(model_dir: str, texts: list[str], dtype: str) -> list[list[int]]:
    import torch
    from sentence_transformers import SentenceTransformer

    td = {"f32": torch.float32, "bf16": torch.bfloat16}[dtype]
    model = SentenceTransformer(model_dir, trust_remote_code=True,
                                local_files_only=True,
                                model_kwargs={"torch_dtype": td})
    out = model.encode(texts, show_progress_bar=False)
    return [[int(round(float(x))) for x in row] for row in out.tolist()]


def native_int8(binary: str, model_dir: str, backend: str,
                texts: list[str]) -> list[list[int]]:
    cmd = [binary, "-d", model_dir, f"--{backend}", "--stream", "-b", "8"]
    proc = subprocess.run(cmd, input="\n".join(texts) + "\n", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"native failed: {' '.join(cmd)}\n{proc.stderr[-800:]}")
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        emb = json.loads(line)["embedding"]
        rows.append([max(-128, min(127, round(math.tanh(float(x)) * 127.0)))
                     for x in emb])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="F32 reference model dir (HF snapshot with onnx/)")
    ap.add_argument("--engines", default="onnx,hf",
                    help="comma list: onnx,hf,native")
    ap.add_argument("--native-binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--native-backend", default="mlx", choices=("cpu", "mlx"))
    ap.add_argument("--native-bf16-dir",
                    help="BF16 model dir for the native engine (required for native)")
    ap.add_argument("--onnx-low", default="model_quantized.onnx",
                    help="quantized ONNX file (vs model.onnx)")
    args = ap.parse_args()

    texts = DEFAULT_TEXTS
    engines = [e for e in args.engines.split(",") if e]
    results: list[tuple[str, str, dict[str, float]]] = []

    for eng in engines:
        try:
            if eng == "onnx":
                ref = onnx_int8(args.model_dir, "model.onnx", texts)
                low = onnx_int8(args.model_dir, args.onnx_low, texts)
                results.append(("ONNX", f"f32 vs {args.onnx_low}", drift(ref, low)))
            elif eng == "hf":
                ref = hf_int8(args.model_dir, texts, "f32")
                low = hf_int8(args.model_dir, texts, "bf16")
                results.append(("HF/torch", "f32 vs bf16", drift(ref, low)))
            elif eng == "native":
                if not args.native_bf16_dir:
                    raise RuntimeError("--native-bf16-dir required for native")
                ref = native_int8(args.native_binary, args.model_dir,
                                  args.native_backend, texts)
                low = native_int8(args.native_binary, args.native_bf16_dir,
                                  args.native_backend, texts)
                results.append((f"native/{args.native_backend}", "f32 vs bf16",
                                drift(ref, low)))
            else:
                print(f"unknown engine: {eng}", file=sys.stderr)
        except Exception as exc:  # one engine failing should not block the rest
            print(f"  [{eng}] SKIPPED: {type(exc).__name__}: {exc}",
                  file=sys.stderr)

    print(f"\nf32 -> low-precision int8 drift ({len(texts)} texts)\n")
    print(f"{'engine':14s} {'comparison':24s} {'int8_change':>11s} "
          f"{'int8_cosine':>12s} {'rel_l2':>8s}")
    print("-" * 74)
    for name, comp, d in results:
        print(f"{name:14s} {comp:24s} {d['int8_change_rate']*100:10.1f}% "
              f"{d['worst_int8_cosine']:12.5f} {d['worst_rel_l2']*100:7.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
