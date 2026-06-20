#!/usr/bin/env python3
"""Inspect and detect local model snapshots, across model families.

For each model it detects:
  - the architecture (from config.json: model_type / architectures / dims),
  - the tokenizer family - BPE (vocab.json + merges.txt), WordPiece (vocab.txt),
    or SentencePiece (sentencepiece.bpe.model / spiece.model / tokenizer.model),
  - the weight layout - single safetensors, sharded safetensors, or legacy
    pytorch_model.bin,
  - the serving kind - base encoder, pooled sentence-embedding, or projected /
    late-interaction (1_Dense) - plus the Sentence-Transformers pipeline.

ONNX exports and other extras are reported when present.

A MODEL_DIR argument may be a model directory (one with config.json) or any
directory to search. Point it at a single snapshot, a folder of models, or the
whole Hugging Face hub cache (~/.cache/huggingface/hub) and it finds each model.

A snapshot is "ok" when it has config.json, some weights, and some tokenizer;
everything else is informational.

Usage: inspect_model_artifacts.py MODEL_DIR [MODEL_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.model",
)

# The first family whose required files are all present wins.
TOKENIZER_FAMILIES = (
    ("BPE", ("vocab.json", "merges.txt")),
    ("WordPiece", ("vocab.txt",)),
    ("SentencePiece", ("sentencepiece.bpe.model",)),
    ("SentencePiece", ("spiece.model",)),
    ("SentencePiece", ("tokenizer.model",)),
)

EXTRA_ARTIFACTS = (
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling",
    "1_Dense",
    "onnx",
    "generation_config.json",
)


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def describe(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    if path.is_dir():
        return f"{sum(1 for _ in path.iterdir())} entries"
    return human_bytes(path.resolve().stat().st_size)


def fmt(label: str, path: Path, indent: int = 4, width: int = 32) -> str:
    return f"{' ' * indent}{label:<{width}} {describe(path)}"


def is_model_dir(d: Path) -> bool:
    return (d / "config.json").is_file()


def discover(root: Path) -> list[Path]:
    """Return the model directories at or under `root`.

    A model directory has a config.json. If `root` is itself one, return it;
    otherwise search descendants (e.g. a folder of models or the HF hub cache),
    skipping onnx sub-exports and dropping any dir nested under another model.
    """
    root = root.expanduser().resolve()
    if is_model_dir(root):
        return [root]
    found = {
        cfg.parent
        for cfg in root.rglob("config.json")
        if cfg.parent.name != "onnx"
    }
    out = []
    for d in sorted(found):
        if not any(d != o and str(d).startswith(str(o) + os.sep) for o in found):
            out.append(d)
    return out


def tokenizer_family(d: Path) -> str:
    for name, required in TOKENIZER_FAMILIES:
        if all((d / f).exists() for f in required):
            return name
    if (d / "tokenizer.json").exists():
        return "fast-only (tokenizer.json)"
    return "none"


def config_summary(d: Path) -> str | None:
    cfg = d / "config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text())
    except (OSError, ValueError):
        return "config.json: unreadable"
    parts = []
    if data.get("model_type"):
        parts.append(str(data["model_type"]))
    arch = data.get("architectures") or []
    if arch:
        parts.append(str(arch[0]))
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "vocab_size", "max_position_embeddings"):
        if key in data:
            parts.append(f"{key}={data[key]}")
    return "  ".join(parts) if parts else "config.json: no recognized fields"


def modules_pipeline(d: Path) -> str | None:
    mj = d / "modules.json"
    if not mj.exists():
        return None
    try:
        mods = json.loads(mj.read_text())
    except (OSError, ValueError):
        return None
    names = [(m.get("type") or "").rsplit(".", 1)[-1] or m.get("name") or "?"
             for m in mods]
    return " -> ".join(n for n in names if n) or None


def model_kind(d: Path) -> str:
    if (d / "1_Dense").is_dir():
        return "projected / late-interaction (1_Dense)"
    if (d / "1_Pooling").is_dir() or (d / "modules.json").exists():
        return "pooled sentence-embedding"
    return "base encoder / LM"


def weight_lines(d: Path) -> tuple[list[str], bool]:
    """Return (formatted lines, whether any weights were found)."""
    lines: list[str] = []
    st_index = d / "model.safetensors.index.json"
    st_shards = sorted(d.glob("model-*-of-*.safetensors"))
    pt_index = d / "pytorch_model.bin.index.json"
    pt_shards = sorted(d.glob("pytorch_model-*-of-*.bin"))
    onnx = sorted((d / "onnx").glob("*.onnx")) if (d / "onnx").is_dir() else []
    found = False

    if (d / "model.safetensors").exists():
        lines.append(fmt("model.safetensors", d / "model.safetensors"))
        found = True
    if st_index.exists() and st_shards:
        lines.append(fmt("model.safetensors.index.json", st_index))
        lines += [fmt(s.name, s, indent=6) for s in st_shards]
        found = True
    if (d / "pytorch_model.bin").exists():
        lines.append(fmt("pytorch_model.bin", d / "pytorch_model.bin"))
        found = True
    if pt_index.exists() and pt_shards:
        lines.append(fmt("pytorch_model.bin.index.json", pt_index))
        lines += [fmt(s.name, s, indent=6) for s in pt_shards]
        found = True
    if onnx:
        lines += [fmt(f"onnx/{o.name}", o) for o in onnx]
        found = True
    if not found:
        lines.append("    (no safetensors / pytorch / onnx weights)")
    return lines, found


def inspect(model_dir: Path) -> bool:
    d = model_dir.expanduser().resolve()
    print(f"snapshot: {d}")
    if not d.is_dir():
        print("  NOT A DIRECTORY")
        return False

    print(f"  model:     {config_summary(d) or 'config.json MISSING'}")
    print(f"  kind:      {model_kind(d)}")
    pipeline = modules_pipeline(d)
    if pipeline:
        print(f"  pipeline:  {pipeline}")

    family = tokenizer_family(d)
    print(f"  tokenizer: {family}")
    for name in TOKENIZER_ARTIFACTS:
        if (d / name).exists():
            print(fmt(name, d / name))

    print("  weights:")
    wlines, has_weights = weight_lines(d)
    print("\n".join(wlines))

    extras = [name for name in EXTRA_ARTIFACTS if (d / name).exists()]
    if extras:
        print("  extras:")
        for name in extras:
            print(fmt(name, d / name))

    ok = (d / "config.json").exists() and has_weights and family != "none"
    print(f"  verdict:   {'ok' if ok else 'incomplete'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dirs", nargs="+", metavar="MODEL_DIR")
    args = ap.parse_args()

    ok = True
    first = True
    for name in args.model_dirs:
        root = Path(os.path.expandvars(name)).expanduser().resolve()
        found = discover(root)
        if not found:
            if not first:
                print()
            print(f"snapshot: {root}\n  no config.json found at or under this path")
            ok = first = False
            continue
        if len(found) > 1:
            if not first:
                print()
            print(f"# {root}: {len(found)} models")
            first = False
        for d in found:
            if not first:
                print()
            ok = inspect(d) and ok
            first = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
