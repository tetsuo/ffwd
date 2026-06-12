#!/usr/bin/env python3
"""Inspect local pplx-embed snapshots."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PRIMARY_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


def human_bytes(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def describe(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    resolved = path.resolve()
    suffix = ""
    if path.is_symlink():
        suffix = f" -> {resolved}"
    return f"{human_bytes(resolved.stat().st_size):>10}{suffix}"


def inspect(model_dir: Path) -> bool:
    model_dir = model_dir.expanduser().resolve()
    print(f"snapshot: {model_dir}")
    ok = True
    for name in PRIMARY_FILES:
        path = model_dir / name
        print(f"  {name:<30} {describe(path)}")
        ok = path.exists() and ok

    model = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if model.exists():
        print(f"  {'model.safetensors':<30} {describe(model)}")
    elif index.exists() and shards:
        print(f"  {'model.safetensors.index.json':<30} {describe(index)}")
        for path in shards:
            print(f"    {path.name:<28} {describe(path)}")
    else:
        print(f"  {'model.safetensors':<30} MISSING")
        ok = False

    onnx_dir = model_dir / "onnx"
    onnx_files = sorted(onnx_dir.glob("*")) if onnx_dir.is_dir() else []
    print("  onnx/")
    if not onnx_files:
        print("    MISSING")
        return False
    for path in onnx_files:
        print(f"    {path.name:<28} {describe(path)}")
        ok = path.exists() and ok
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dirs", nargs="+", metavar="MODEL_DIR")
    args = ap.parse_args()

    ok = True
    for i, name in enumerate(args.model_dirs):
        if i:
            print()
        ok = inspect(Path(os.path.expandvars(name))) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
