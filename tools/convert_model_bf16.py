#!/usr/bin/env python3
"""Convert a local safetensors model directory to BF16.

The output directory keeps non-safetensors files as symlinks by default and
writes converted safetensors files. F32 tensors are converted to BF16 with
round-to-nearest-even. Existing BF16 tensors are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np


COPY_CHUNK = 64 * 1024 * 1024
F32_CHUNK_ELEMS = 16 * 1024 * 1024


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too small for safetensors header")
        header_len = struct.unpack("<Q", raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    return json.loads(header_raw), 8 + header_len


def tensor_nbytes(spec: dict) -> int:
    start, end = spec["data_offsets"]
    return int(end) - int(start)


def converted_tensor_nbytes(spec: dict) -> int:
    if spec["dtype"] == "F32":
        return tensor_nbytes(spec) // 2
    return tensor_nbytes(spec)


def build_output_header(header: dict) -> tuple[dict, dict[str, tuple[int, int]]]:
    out_header = {}
    offsets = {}
    cursor = 0

    for name, spec in header.items():
        if name == "__metadata__":
            out_header[name] = spec
            continue

        out_spec = dict(spec)
        nbytes = converted_tensor_nbytes(spec)
        offsets[name] = (cursor, cursor + nbytes)
        out_spec["data_offsets"] = [cursor, cursor + nbytes]
        if spec["dtype"] == "F32":
            out_spec["dtype"] = "BF16"
        out_header[name] = out_spec
        cursor += nbytes

    return out_header, offsets


def copy_range(src, dst, nbytes: int) -> None:
    left = nbytes
    while left:
        chunk = src.read(min(left, COPY_CHUNK))
        if not chunk:
            raise IOError("unexpected EOF while copying tensor data")
        dst.write(chunk)
        left -= len(chunk)


def write_f32_as_bf16(src, dst, nbytes: int) -> None:
    if nbytes % 4 != 0:
        raise ValueError("F32 tensor byte size is not divisible by 4")

    elems_left = nbytes // 4
    while elems_left:
        count = min(elems_left, F32_CHUNK_ELEMS)
        raw = src.read(count * 4)
        if len(raw) != count * 4:
            raise IOError("unexpected EOF while converting F32 tensor data")

        bits = np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=True)
        lsb = (bits >> np.uint32(16)) & np.uint32(1)
        bf16 = ((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype("<u2")
        dst.write(bf16.tobytes())
        elems_left -= count


def convert_safetensors(src_path: Path, dst_path: Path,
                        verbose: bool = False) -> tuple[int, int]:
    header, data_start = read_safetensors_header(src_path)
    out_header, _offsets = build_output_header(header)
    out_header_raw = json.dumps(out_header, separators=(",", ":")).encode("utf-8")

    tensors = [(name, spec) for name, spec in header.items() if name != "__metadata__"]
    input_bytes = sum(tensor_nbytes(spec) for _, spec in tensors)
    output_bytes = sum(converted_tensor_nbytes(spec) for _, spec in tensors)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    with src_path.open("rb") as src, tmp_path.open("wb") as dst:
        dst.write(struct.pack("<Q", len(out_header_raw)))
        dst.write(out_header_raw)

        for name, spec in tensors:
            start, _end = spec["data_offsets"]
            src.seek(data_start + int(start))
            nbytes = tensor_nbytes(spec)
            dtype = spec["dtype"]
            if dtype == "F32":
                write_f32_as_bf16(src, dst, nbytes)
            elif dtype == "BF16":
                copy_range(src, dst, nbytes)
            else:
                copy_range(src, dst, nbytes)
            if verbose:
                print(f"  {name}: {dtype} {nbytes / (1024 ** 2):.1f} MB", flush=True)

    tmp_path.replace(dst_path)
    return input_bytes, output_bytes


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    else:
        target = src.resolve()
        os.symlink(target, dst, target_is_directory=src.is_dir())


def convert_model(src_dir: Path, dst_dir: Path, copy_files: bool,
                  verbose: bool = False) -> None:
    if not src_dir.is_dir():
        raise SystemExit(f"source model dir does not exist: {src_dir}")
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise SystemExit(f"output dir already exists and is not empty: {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    safetensors = sorted(src_dir.glob("*.safetensors"))
    if not safetensors:
        raise SystemExit(f"no safetensors files found in {src_dir}")

    total_in = 0
    total_out = 0
    for src_path in safetensors:
        dst_path = dst_dir / src_path.name
        print(f"converting {src_path.name}", flush=True)
        n_in, n_out = convert_safetensors(src_path.resolve(), dst_path, verbose)
        total_in += n_in
        total_out += n_out

    for child in src_dir.iterdir():
        if child.name.endswith(".safetensors"):
            continue
        dst_child = dst_dir / child.name
        if dst_child.exists() or dst_child.is_symlink():
            continue
        link_or_copy(child, dst_child, copy_files)

    print(
        f"done: tensor data {total_in / (1024 ** 3):.2f} GiB -> "
        f"{total_out / (1024 ** 3):.2f} GiB",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source model directory")
    ap.add_argument("--dst", required=True, help="output model directory")
    ap.add_argument("--copy-files", action="store_true",
                    help="copy non-safetensors files instead of symlinking")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every tensor as it is converted")
    args = ap.parse_args()

    convert_model(Path(args.src), Path(args.dst), args.copy_files, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
