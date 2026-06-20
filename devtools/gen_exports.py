#!/usr/bin/env python3
"""Generate the public-ABI linker export map for libffwd from the header.

The shared library is built with -fvisibility=hidden, so the only symbols that
leave it are the declarations marked FFWD_API in libffwd/ffwd.h. This script
reads those annotations and writes the two linker inputs the build consumes:

  ffwd.exports   Mach-O (-Wl,-exported_symbols_list): one _symbol per line
  ffwd.map       ELF    (-Wl,--version-script):  FFWD { global: ...; local: *; }

Run it as a release step when the libffwd public API changes:

  python3 devtools/gen_exports.py             # write ffwd.exports and ffwd.map
  python3 devtools/gen_exports.py --check     # verify they are in sync (CI), no write
  python3 devtools/gen_exports.py --outdir D  # write somewhere else
"""
import argparse
import os
import re
import sys


def find_symbols(header_text):
    # Drop comments and preprocessor lines so the FFWD_API macro definition
    # (#ifndef/#define FFWD_API ...) and doc comments are not mistaken for
    # declarations.
    text = re.sub(r"/\*.*?\*/", "", header_text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))

    syms = set()
    for m in re.finditer(r"\bFFWD_API\b(.*?)([(;])", text, flags=re.S):
        ids = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(1))
        if ids and ids[-1].startswith("ffwd_"):
            syms.add(ids[-1])
    return sorted(syms)


def render_exports(syms):
    return "".join("_%s\n" % s for s in syms)


def render_map(syms):
    body = "\n".join("    %s;" % s for s in syms)
    return "FFWD {\n  global:\n%s\n  local:\n    *;\n};\n" % body


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--header", default=os.path.join(root, "libffwd", "ffwd.h"))
    ap.add_argument("--outdir", default=root, help="directory for ffwd.exports / ffwd.map")
    ap.add_argument("--check", action="store_true", help="verify files are in sync; do not write")
    args = ap.parse_args()

    with open(args.header) as f:
        syms = find_symbols(f.read())
    if not syms:
        sys.exit("error: no FFWD_API symbols found in %s" % args.header)

    targets = (
        (os.path.join(args.outdir, "ffwd.exports"), render_exports(syms)),
        (os.path.join(args.outdir, "ffwd.map"), render_map(syms)),
    )

    if args.check:
        stale = False
        for path, want in targets:
            try:
                have = open(path).read()
            except FileNotFoundError:
                have = None
            if have != want:
                print("out of sync: %s" % path)
                stale = True
        if stale:
            sys.exit("run: python3 %s" % os.path.relpath(__file__, os.getcwd()))
        print("in sync (%d symbols)" % len(syms))
        return

    for path, want in targets:
        with open(path, "w") as f:
            f.write(want)
    print("wrote ffwd.exports and ffwd.map (%d symbols)" % len(syms))


if __name__ == "__main__":
    main()
