#!/usr/bin/env python3
"""Verify that MLX rejects mismatched model metadata before touching Metal."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "ffwd-cli"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    binary = Path(args.binary).resolve()
    model_dir = Path(args.model_dir).resolve()
    with (model_dir / "config.json").open() as f:
        config = json.load(f)

    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 1:
        raise SystemExit("config.json has no usable vocab_size")
    config["vocab_size"] = vocab_size - 1

    with tempfile.TemporaryDirectory(prefix="ffwd-mlx-invalid-") as td:
        tmp = Path(td)
        (tmp / "config.json").write_text(json.dumps(config))
        (tmp / "vocab.json").symlink_to(model_dir / "vocab.json")
        tensors = list(model_dir.glob("*.safetensors"))
        if not tensors:
            raise SystemExit("model directory has no safetensor files")
        for tensor in tensors:
            (tmp / tensor.name).symlink_to(tensor)

        proc = subprocess.run(
            [str(binary), "-d", str(tmp), "--stream"],
            input="",
            text=True,
            capture_output=True,
            check=False,
        )

    expected = "mlx: bad shape for embed_tokens.weight at dim 0"
    if proc.returncode == 0:
        raise SystemExit("MLX unexpectedly accepted mismatched vocab_size")
    if expected not in proc.stderr:
        raise SystemExit(f"unexpected stderr:\n{proc.stderr}")
    if "MLX error:" in proc.stderr:
        raise SystemExit(f"MLX touched Metal before rejecting the model:\n{proc.stderr}")
    print("ok: MLX rejected mismatched tensor shape before Metal initialization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
