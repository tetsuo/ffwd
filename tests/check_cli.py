#!/usr/bin/env python3
"""Black-box check of the ffwd CLI surface: argument parsing, mode dispatch,
output shape, and exit codes.

This does NOT check embedding correctness - that is the engine tests' job. It
runs a hermetic synthetic model (written by the gen_fixture test helper, the
same fixture the C tests build) so the CLI has something to load, then drives
the binary through its modes and asserts only exit codes and output shapes.

The binary path is an argument (default tools/cli/ffwd-cli), so this works
against any backend build - CPU, MLX, or CUDA. Backend-specific checks (the CPU
thread-count line) are gated on the binary's reported backend.

Usage: tests/check_cli.py [--bin PATH]
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIDDEN = 4  # fixture hidden_size; a single embedding is this many floats

failures = 0


def fail(what):
    global failures
    failures += 1
    print(f"FAIL: {what}", file=sys.stderr)


def expect(cond, what):
    if not cond:
        fail(what)


def ensure_tool(path, make_dir, make_target):
    """Build a Make target if its output binary is missing."""
    if not os.path.exists(path):
        subprocess.run(["make", "-C", make_dir, make_target], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    if not os.path.exists(path):
        sys.exit(f"required binary not found and could not be built: {path}")


def run_cli(binary, args, stdin_text=None):
    """Run the CLI; return (exit_code, stdout, stderr)."""
    proc = subprocess.run(
        [binary, *args],
        input=stdin_text if stdin_text is not None else "",
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def line_is_embedding(text, dim):
    """First line must hold exactly `dim` finite floats."""
    line = text.splitlines()[0] if text.splitlines() else ""
    try:
        vals = [float(tok) for tok in line.split()]
    except ValueError:
        return False
    return len(vals) == dim and all(math.isfinite(v) for v in vals)


def backend_label(binary):
    """Read the backend from --build-info, e.g. 'CPU' / 'Apple Silicon GPU'."""
    _, out, _ = run_cli(binary, ["--build-info"])
    for line in out.splitlines():
        if line.startswith("backend:"):
            return line.split(":", 1)[1].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "tools", "cli", "ffwd-cli"),
                    help="path to the embed CLI binary")
    args = ap.parse_args()

    binary = args.bin
    ensure_tool(binary, "tools/cli", "all")

    gen = os.path.join(ROOT, "tests", "gen_fixture")
    ensure_tool(gen, "tests", "gen_fixture")

    with tempfile.TemporaryDirectory(prefix="ffwd-cli-check-") as model:
        subprocess.run([gen, "--out", model, "--model", "qwen3"],
                       check=True, stdout=subprocess.DEVNULL)

        # Help and missing-argument handling.
        expect(run_cli(binary, ["-h"])[0] == 0, "-h exits 0")
        expect(run_cli(binary, [])[0] == 1, "no arguments exits 1")
        expect(run_cli(binary, ["-d", "/nonexistent-dir", "hi"])[0] == 1,
               "missing model dir exits 1")

        # -V reports the program's base name, not the path it was invoked by.
        code, out, _ = run_cli(binary, ["-V"])
        expect(code == 0, "-V exits 0")
        prog = out.split()[0] if out.split() else ""
        expect(prog == os.path.basename(binary) and "/" not in prog,
               "-V strips the directory from the program name")

        # Mode 1: one text prints one embedding line.
        code, out, _ = run_cli(binary, ["-d", model, "hello world"])
        expect(code == 0, "single text exits 0")
        expect(line_is_embedding(out, HIDDEN), "single text prints dim floats")

        # Thread count: CPU backend only (GPU backends ignore -t and print a
        # different startup line).
        code, _, err = run_cli(binary, ["-d", model, "-t", "2", "-v", "hello world"])
        expect(code == 0, "-t 2 -v exits 0")
        if backend_label(binary) == "CPU":
            expect("Using 2 CPU thread(s)" in err, "-v reports the thread count")

        # Mode 2+: several texts print a bare numeric matrix, one row per line,
        # with no titles or labels. Two texts give a 2x2 with a 1.0 diagonal.
        code, out, _ = run_cli(binary, ["-d", model, "hello world", "held"])
        expect(code == 0, "two texts exit 0")
        expect(len(out.splitlines()) == 2, "two texts print a 2-row matrix")
        expect("1.000000" in out, "matrix diagonal is 1.000000")
        expect("Cosine" not in out, "matrix has no title line")

        # --json on a matrix emits a bare JSON array of rows, no labels.
        code, out, _ = run_cli(binary, ["-d", model, "--json", "hello world", "held"])
        expect(code == 0, "two texts --json exits 0")
        expect("[[" in out and "1.000000" in out, "--json matrix is a bare array of rows")
        expect("{" not in out, "--json matrix has no object or labels")

        # --json on a single text emits a bare JSON array of floats.
        code, out, _ = run_cli(binary, ["-d", model, "--json", "hello world"])
        expect(code == 0, "single text --json exits 0")
        expect("[" in out and "{" not in out, "--json single text is a bare array")

        # -e appends the raw embeddings (label-free) after the matrix: three
        # matrix rows, a blank separator, then three embedding rows.
        code, out, _ = run_cli(binary, ["-d", model, "-e", "-b", "1",
                                        "hello", "world", "held"])
        expect(code == 0, "-e -b 1 with three texts exits 0")
        expect(len(out.splitlines()) == 7, "-e prints matrix then raw embeddings")

        # Batch mode reads stdin when no texts are given.
        code, out, _ = run_cli(binary, ["-d", model], "hello world\nheld\n")
        expect(code == 0, "stdin batch exits 0")
        expect(len(out.splitlines()) == 2, "stdin batch prints a 2-row matrix")
        expect(run_cli(binary, ["-d", model], "\n\n")[0] == 1, "blank stdin exits 1")

        # Streaming mode writes one JSON object per input line.
        code, out, _ = run_cli(binary, ["-d", model, "--stream"], "hello\nworld\n")
        expect(code == 0, "--stream exits 0")
        expect(len(out.splitlines()) == 2, "--stream writes one line per input")
        expect('{"embedding":[' in out, "--stream writes embedding JSON")

        code, out, _ = run_cli(binary, ["-d", model, "--stream", "-b", "2", "-v"],
                               "hello\nworld\nheld\n")
        expect(code == 0, "--stream -b 2 exits 0")
        expect(len(out.splitlines()) == 3, "--stream -b 2 covers the tail batch")

        # Flag validation: empty model dir must exit 1.
        expect(run_cli(binary, ["-d", "", "t"])[0] == 1, "empty model dir exits 1")

        # Unknown --options fail loudly instead of being silently embedded as
        # text. A bare -- ends option parsing so dash-prefixed text still embeds.
        code, _, err = run_cli(binary, ["-d", model, "--backend", "cuda", "hi"])
        expect(code == 1, "unknown --option exits 1")
        expect("unknown option" in err, "unknown --option names the flag")
        code, out, _ = run_cli(binary, ["-d", model, "--", "-dashy text"])
        expect(code == 0, "-- terminator allows dash-prefixed text")
        expect(line_is_embedding(out, HIDDEN), "dash text after -- embeds as one input")

    if failures:
        print(f"{failures} CLI check(s) failed", file=sys.stderr)
        return 1
    print(f"ok: CLI modes and flag validation against {binary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
