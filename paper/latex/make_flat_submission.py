#!/usr/bin/env python3
r"""Assemble SUBMISSION/ -- a FLAT, self-contained folder the supervisor can zip and hand
to the journal, where manuscript.tex compiles with pdflatex alone.

Consumes the already-flattened MANUSCRIPT/manuscript.tex (make_submission.py owns the
\input expansion and the bibliography embedding; this script does not repeat that work).
The one thing that stops MANUSCRIPT/ from being zippable is \graphicspath pointing at
../figures/main/, so the figures are copied in and the path is rewritten to {{./}}.

Scope: the MAIN MANUSCRIPT ONLY. The supplement is submitted as a finished PDF, so
supplementary.tex and the SF* figures are deliberately absent.

Compile: pdflatex twice. No bibtex (the bibliography is a literal thebibliography block).

Usage:
  python paper/latex/make_flat_submission.py             # build the folder, then zip it
  python paper/latex/make_flat_submission.py --zip-only  # re-zip an existing folder

build.sh uses --zip-only to repackage after test-compiling SUBMISSION/ and deleting the
build artifacts, so the shipped zip is exactly the file set that was proven to compile.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))            # paper/latex
ROOT = os.path.dirname(os.path.dirname(HERE))                # repo root
SRC = os.path.join(ROOT, "MANUSCRIPT", "manuscript.tex")
FIGDIR = os.path.join(ROOT, "figures", "main")
OUT = os.path.join(ROOT, "SUBMISSION")
ZIP = os.path.join(ROOT, "SUBMISSION.zip")

# Non-stock TeX files that must travel with the manuscript. wlscirep.cls RequirePackages
# jabbrv, and jabbrv.sty \InputIfFileExists both .ldf files (hard error if either is
# missing). naturemag-doi.bst is not actually read -- \bibliographystyle only writes
# \bibstyle to the .aux, and bibtex never runs -- but it is 36 kB of insurance against an
# environment that invokes bibtex unprompted. Drop it here for a truly minimal folder.
SUPPORT = (
    "wlscirep.cls",
    "jabbrv.sty",
    "jabbrv-ltwa-all.ldf",
    "jabbrv-ltwa-en.ldf",
    "naturemag-doi.bst",
)

GRAPHICSPATH_RE = re.compile(r"\\graphicspath\{\{[^}]*\}\}")
INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
INPUT_RE = re.compile(r"\\input\{")

# Comment stripping for the shipped copy. The journal should not receive our build
# plumbing (generator names, repo-internal paths, "DO NOT hand-edit" banners).
# A whole-line comment can be deleted outright: TeX swallows the line and its newline.
WHOLE_LINE_COMMENT_RE = re.compile(r"^[ \t]*%.*\n", re.M)
# A TRAILING comment must keep its "%" -- e.g. `\resizebox{\textwidth}{!}{% >>> inlined`
# where the % also suppresses the following newline. Dropping it injects a space into
# the box argument. Only the marker text goes.
INLINE_MARKER_RE = re.compile(r"%[ \t]*(?:>>>|<<<)[^\n]*")


def flatten_graphicspath(text: str) -> str:
    r"""Point \graphicspath at the folder itself. Fail loudly rather than silently ship a
    folder whose figures resolve outside it."""
    text, n = GRAPHICSPATH_RE.subn(r"\\graphicspath{{./}}", text)
    if n != 1:
        raise SystemExit(
            f"expected exactly one \\graphicspath in {SRC}, found {n} -- "
            "did make_submission.py change? refusing to write a folder that may not be flat"
        )
    return text


def strip_comments(text: str) -> str:
    """Remove LaTeX comments from the shipped copy, preserving line-continuation semantics.

    Order matters: collapse trailing markers to a bare "%" first, which turns any
    whole-line marker into a lone "%" that the second pass then deletes along with
    its newline. Doing it the other way round would leave the trailing markers intact.
    """
    text = INLINE_MARKER_RE.sub("%", text)
    return WHOLE_LINE_COMMENT_RE.sub("", text)


def figures_referenced(text: str) -> list:
    r"""Figure filenames actually used, parsed from the source rather than hard-coded, so
    the list cannot go stale when floats are renumbered (F5 already became Table 2 once)."""
    names = []
    for name in INCLUDEGRAPHICS_RE.findall(text):
        if name not in names:
            names.append(name)
    return names


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        text = f.read()

    figs = figures_referenced(text)
    text = flatten_graphicspath(text)
    text = strip_comments(text)

    if INPUT_RE.search(text):
        raise SystemExit(f"{SRC} still has \\input -- run make_submission.py first")

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    # no header banner: the shipped file carries no build provenance at all
    with open(os.path.join(OUT, "manuscript.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    for name in SUPPORT:
        src = os.path.join(HERE, name)
        if not os.path.isfile(src):
            raise SystemExit(f"missing support file: {src}")
        shutil.copy2(src, os.path.join(OUT, name))

    for name in figs:
        src = os.path.join(FIGDIR, name)
        if not os.path.isfile(src):
            raise SystemExit(f"manuscript.tex references {name}, not found in {FIGDIR}")
        shutil.copy2(src, os.path.join(OUT, name))

    write_zip()


def write_zip() -> None:
    """Zip the folder with flat archive names (no nested directory on extraction)."""
    names = sorted(os.listdir(OUT))
    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(os.path.join(OUT, name), arcname=name)

    total = 0
    print(f"SUBMISSION/ ({len(names)} files, flat)")
    for name in names:
        size = os.path.getsize(os.path.join(OUT, name))
        total += size
        print(f"  {name:<32} {size / 1024:8.1f} kB")
    print(f"  {'TOTAL':<32} {total / 1024:8.1f} kB")
    print(f"wrote {os.path.relpath(ZIP, ROOT)} ({os.path.getsize(ZIP) / 1024:.1f} kB)")


if __name__ == "__main__":
    if "--zip-only" in sys.argv:
        if not os.path.isdir(OUT):
            raise SystemExit(f"--zip-only: {OUT} does not exist; run without the flag first")
        write_zip()
    else:
        main()
